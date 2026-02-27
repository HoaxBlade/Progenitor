/*
 * Sparse MLP forward pass using Apple Accelerate SparseBLAS.
 * Compiled as a Python C extension via ctypes.
 *
 * Supports: input -> [MatMul(W_sparse) + Add(bias) + ReLU] x N layers -> output
 * All in one C call to avoid Python-per-node overhead.
 */

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    sparse_matrix_float W;  /* sparse weight handle */
    float *bias;            /* bias vector (dense, owned) */
    int in_dim;
    int out_dim;
    int has_relu;           /* 1 = apply relu after add */
} Layer;

typedef struct {
    int n_layers;
    Layer *layers;
} SparseMLPModel;

/* Create model with n_layers. Caller fills in layers via set_layer. */
SparseMLPModel *smlp_create(int n_layers) {
    SparseMLPModel *m = (SparseMLPModel *)malloc(sizeof(SparseMLPModel));
    m->n_layers = n_layers;
    m->layers = (Layer *)calloc(n_layers, sizeof(Layer));
    return m;
}

/* Set layer i: provide dense weight (row-major, in_dim x out_dim), bias, relu flag.
   Builds sparse matrix internally. */
void smlp_set_layer(SparseMLPModel *m, int idx,
                     float *W_dense, int in_dim, int out_dim,
                     float *bias, int has_relu) {
    Layer *l = &m->layers[idx];
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->has_relu = has_relu;

    /* Build sparse matrix from dense W (in_dim x out_dim) */
    l->W = sparse_matrix_create_float(in_dim, out_dim);
    for (int r = 0; r < in_dim; r++) {
        for (int c = 0; c < out_dim; c++) {
            float v = W_dense[r * out_dim + c];
            if (v != 0.0f) {
                sparse_insert_entry_float(l->W, v, r, c);
            }
        }
    }
    sparse_commit(l->W);

    /* Copy bias */
    l->bias = (float *)malloc(out_dim * sizeof(float));
    if (bias) {
        memcpy(l->bias, bias, out_dim * sizeof(float));
    } else {
        memset(l->bias, 0, out_dim * sizeof(float));
    }
}

/* Forward pass: x (in_dim of layer 0) -> output (out_dim of last layer).
   output must be pre-allocated by caller. */
void smlp_forward(SparseMLPModel *m, float *x, float *output) {
    float *cur = x;
    float *buf = NULL;
    int buf_size = 0;

    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        int out_n = l->out_dim;

        /* Allocate output buffer (reuse if big enough) */
        float *out;
        if (i == m->n_layers - 1) {
            out = output;
        } else {
            if (out_n > buf_size) {
                buf = (float *)realloc(buf, out_n * sizeof(float));
                buf_size = out_n;
            }
            out = buf;
        }
        memset(out, 0, out_n * sizeof(float));

        /* y = W^T @ x  (CblasTrans=CblasTrans) */
        sparse_matrix_vector_product_dense_float(CblasTrans, 1.0f, l->W, cur, 1, out, 1);

        /* Add bias */
        for (int j = 0; j < out_n; j++) {
            out[j] += l->bias[j];
        }

        /* ReLU */
        if (l->has_relu) {
            for (int j = 0; j < out_n; j++) {
                if (out[j] < 0.0f) out[j] = 0.0f;
            }
        }

        cur = out;
    }

    free(buf);
}

void smlp_destroy(SparseMLPModel *m) {
    for (int i = 0; i < m->n_layers; i++) {
        sparse_matrix_destroy(m->layers[i].W);
        free(m->layers[i].bias);
    }
    free(m->layers);
    free(m);
}
