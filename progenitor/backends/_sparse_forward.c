/*
 * Sparse MLP forward pass using Apple Accelerate SparseBLAS.
 * Compiled as a Python C extension via ctypes.
 *
 * Supports: input -> [MatMul(W_sparse) + Add(bias) + ReLU] x N layers -> output
 * All in one C call to avoid Python-per-node overhead.
 *
 * Includes INT8 quantized sparse path for reduced memory bandwidth.
 */

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    sparse_matrix_float W;  /* sparse weight handle (FP32 path) */
    float *bias;            /* bias vector (dense, owned) */
    int in_dim;
    int out_dim;
    int has_relu;           /* 1 = apply relu after add */
} Layer;

typedef struct {
    int n_layers;
    Layer *layers;
} SparseMLPModel;

/* INT8 sparse layer — stores non-zeros as int8 with per-layer scale */
typedef struct {
    int in_dim;
    int out_dim;
    int has_relu;
    float *bias;        /* bias vector (dense, owned) */
    float scale;        /* per-layer dequant scale: float_val = scale * int8_val */

    int nnz;            /* number of non-zeros */
    int *row_idx;       /* row indices of non-zeros */
    int *col_idx;       /* column indices of non-zeros */
    int8_t *values;     /* INT8 quantized values */
} LayerI8;

typedef struct {
    int n_layers;
    LayerI8 *layers;
} SparseMLPModelI8;

/* ========== FP32 Sparse API ========== */

SparseMLPModel *smlp_create(int n_layers) {
    SparseMLPModel *m = (SparseMLPModel *)malloc(sizeof(SparseMLPModel));
    m->n_layers = n_layers;
    m->layers = (Layer *)calloc(n_layers, sizeof(Layer));
    return m;
}

void smlp_set_layer(SparseMLPModel *m, int idx,
                     float *W_dense, int in_dim, int out_dim,
                     float *bias, int has_relu) {
    Layer *l = &m->layers[idx];
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->has_relu = has_relu;

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

    l->bias = (float *)malloc(out_dim * sizeof(float));
    if (bias) {
        memcpy(l->bias, bias, out_dim * sizeof(float));
    } else {
        memset(l->bias, 0, out_dim * sizeof(float));
    }
}

void smlp_forward(SparseMLPModel *m, float *x, float *output) {
    float *cur = x;
    float *buf = NULL;
    int buf_size = 0;

    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        int out_n = l->out_dim;

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

        /* y = W^T @ x */
        sparse_matrix_vector_product_dense_float(CblasTrans, 1.0f, l->W, cur, 1, out, 1);

        for (int j = 0; j < out_n; j++) {
            out[j] += l->bias[j];
        }

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

/* ========== INT8 Quantized Sparse API ========== */

SparseMLPModelI8 *smlp_i8_create(int n_layers) {
    SparseMLPModelI8 *m = (SparseMLPModelI8 *)malloc(sizeof(SparseMLPModelI8));
    m->n_layers = n_layers;
    m->layers = (LayerI8 *)calloc(n_layers, sizeof(LayerI8));
    return m;
}

/*
 * Set layer with FP32 dense weight — internally quantizes to INT8.
 * Quantization: scale = max(|W|) / 127, int8_val = round(W / scale)
 */
void smlp_i8_set_layer(SparseMLPModelI8 *m, int idx,
                        float *W_dense, int in_dim, int out_dim,
                        float *bias, int has_relu) {
    LayerI8 *l = &m->layers[idx];
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->has_relu = has_relu;

    /* Find max abs for scale */
    float max_abs = 0.0f;
    int total = in_dim * out_dim;
    for (int k = 0; k < total; k++) {
        float a = W_dense[k] < 0 ? -W_dense[k] : W_dense[k];
        if (a > max_abs) max_abs = a;
    }
    l->scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    float inv_scale = 1.0f / l->scale;

    /* Count non-zeros and quantize */
    int nnz = 0;
    for (int k = 0; k < total; k++) {
        if (W_dense[k] != 0.0f) nnz++;
    }
    l->nnz = nnz;
    l->row_idx = (int *)malloc(nnz * sizeof(int));
    l->col_idx = (int *)malloc(nnz * sizeof(int));
    l->values = (int8_t *)malloc(nnz * sizeof(int8_t));

    int pos = 0;
    for (int r = 0; r < in_dim; r++) {
        for (int c = 0; c < out_dim; c++) {
            float v = W_dense[r * out_dim + c];
            if (v != 0.0f) {
                l->row_idx[pos] = r;
                l->col_idx[pos] = c;
                float q = v * inv_scale;
                int qi = (int)(q + (q > 0 ? 0.5f : -0.5f));
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                l->values[pos] = (int8_t)qi;
                pos++;
            }
        }
    }

    l->bias = (float *)malloc(out_dim * sizeof(float));
    if (bias) {
        memcpy(l->bias, bias, out_dim * sizeof(float));
    } else {
        memset(l->bias, 0, out_dim * sizeof(float));
    }
}

/*
 * INT8 sparse forward: y = scale * (int8_W^T @ x) + bias [+ relu]
 * Custom sparse matvec with inline INT8->FP32 dequantization.
 */
void smlp_i8_forward(SparseMLPModelI8 *m, float *x, float *output) {
    float *cur = x;
    float *buf = NULL;
    int buf_size = 0;

    for (int i = 0; i < m->n_layers; i++) {
        LayerI8 *l = &m->layers[i];
        int out_n = l->out_dim;

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

        /* Start with bias */
        memcpy(out, l->bias, out_n * sizeof(float));

        /* Sparse INT8 matvec: out[col] += scale * values[k] * x[row]
         * W is stored as (row, col, int8_val) triples.
         * This computes y = W^T @ x in quantized form. */
        float sc = l->scale;
        for (int k = 0; k < l->nnz; k++) {
            out[l->col_idx[k]] += sc * (float)l->values[k] * cur[l->row_idx[k]];
        }

        if (l->has_relu) {
            for (int j = 0; j < out_n; j++) {
                if (out[j] < 0.0f) out[j] = 0.0f;
            }
        }

        cur = out;
    }

    free(buf);
}

void smlp_i8_destroy(SparseMLPModelI8 *m) {
    for (int i = 0; i < m->n_layers; i++) {
        free(m->layers[i].row_idx);
        free(m->layers[i].col_idx);
        free(m->layers[i].values);
        free(m->layers[i].bias);
    }
    free(m->layers);
    free(m);
}
