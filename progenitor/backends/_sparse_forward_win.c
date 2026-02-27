/*
 * Pure C Sparse MLP forward pass for Windows (and non-Apple platforms).
 * Compiled as a shared library (.dll / .so) via gcc.
 *
 * Implements the exact same API signatures as the Apple Accelerate version
 * but relies on pure C COO loops instead of proprietary BLAS.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// -------------------------------------------------------------------------
// FP32 Sparse implementation
// -------------------------------------------------------------------------

typedef struct {
    int in_dim;
    int out_dim;
    int has_relu;
    float *bias;        /* bias vector (dense, owned) */

    int nnz;            /* number of non-zeros */
    int *row_idx;       /* row indices of non-zeros */
    int *col_idx;       /* column indices of non-zeros */
    float *values;      /* fp32 uncompressed values */
} Layer;

typedef struct {
    int n_layers;
    Layer *layers;
} SparseMLPModel;


EXPORT SparseMLPModel *smlp_create(int n_layers) {
    SparseMLPModel *m = (SparseMLPModel *)malloc(sizeof(SparseMLPModel));
    m->n_layers = n_layers;
    m->layers = (Layer *)calloc(n_layers, sizeof(Layer));
    return m;
}

EXPORT void smlp_set_layer(SparseMLPModel *m, int idx,
                     float *W_dense, int in_dim, int out_dim,
                     float *bias, int has_relu) {
    Layer *l = &m->layers[idx];
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->has_relu = has_relu;

    int total = in_dim * out_dim;
    int nnz = 0;
    for (int k = 0; k < total; k++) {
        if (W_dense[k] != 0.0f) nnz++;
    }
    l->nnz = nnz;
    l->row_idx = (int *)malloc(nnz * sizeof(int));
    l->col_idx = (int *)malloc(nnz * sizeof(int));
    l->values = (float *)malloc(nnz * sizeof(float));

    int pos = 0;
    for (int r = 0; r < in_dim; r++) {
        for (int c = 0; c < out_dim; c++) {
            float v = W_dense[r * out_dim + c];
            if (v != 0.0f) {
                l->row_idx[pos] = r;
                l->col_idx[pos] = c;
                l->values[pos] = v;
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

EXPORT void smlp_forward(SparseMLPModel *m, float *x, float *output) {
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

        // Initialize out with bias
        memcpy(out, l->bias, out_n * sizeof(float));

        // y = W^T @ x -> out[c] += W[r,c] * x[r]
        for (int k = 0; k < l->nnz; k++) {
            out[l->col_idx[k]] += l->values[k] * cur[l->row_idx[k]];
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

EXPORT void smlp_destroy(SparseMLPModel *m) {
    for (int i = 0; i < m->n_layers; i++) {
        free(m->layers[i].row_idx);
        free(m->layers[i].col_idx);
        free(m->layers[i].values);
        free(m->layers[i].bias);
    }
    free(m->layers);
    free(m);
}

// -------------------------------------------------------------------------
// INT8 Sparse implementation (Identical logic to the Mac equivalent)
// -------------------------------------------------------------------------

typedef struct {
    int in_dim;
    int out_dim;
    int has_relu;
    float *bias;
    float scale;

    int nnz;
    int *row_idx;
    int *col_idx;
    int8_t *values;
} LayerI8;

typedef struct {
    int n_layers;
    LayerI8 *layers;
} SparseMLPModelI8;

EXPORT SparseMLPModelI8 *smlp_i8_create(int n_layers) {
    SparseMLPModelI8 *m = (SparseMLPModelI8 *)malloc(sizeof(SparseMLPModelI8));
    m->n_layers = n_layers;
    m->layers = (LayerI8 *)calloc(n_layers, sizeof(LayerI8));
    return m;
}

EXPORT void smlp_i8_set_layer(SparseMLPModelI8 *m, int idx,
                        float *W_dense, int in_dim, int out_dim,
                        float *bias, int has_relu) {
    LayerI8 *l = &m->layers[idx];
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->has_relu = has_relu;

    float max_abs = 0.0f;
    int total = in_dim * out_dim;
    for (int k = 0; k < total; k++) {
        float a = W_dense[k] < 0 ? -W_dense[k] : W_dense[k];
        if (a > max_abs) max_abs = a;
    }
    l->scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    float inv_scale = 1.0f / l->scale;

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

EXPORT void smlp_i8_forward(SparseMLPModelI8 *m, float *x, float *output) {
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

        memcpy(out, l->bias, out_n * sizeof(float));

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

EXPORT void smlp_i8_destroy(SparseMLPModelI8 *m) {
    for (int i = 0; i < m->n_layers; i++) {
        free(m->layers[i].row_idx);
        free(m->layers[i].col_idx);
        free(m->layers[i].values);
        free(m->layers[i].bias);
    }
    free(m->layers);
    free(m);
}
