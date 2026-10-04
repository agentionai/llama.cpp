# CUDA: trellis quant types (TQ2_T, TQK6, TQK7)

The trellis types used by the Gyro models run on CUDA through the same paths as
the other quant types:

- **token generation** (one token per step) uses the quantized mat-vec kernels
  (`mmvq`), including the MoE expert path;
- **prompt processing** uses the quantized matrix-multiply kernels (MMQ).

No build flag is needed beyond a normal CUDA build (`-DGGML_CUDA=ON`). The
environment variables below exist for A/B testing and for GPUs where the
defaults are not the best choice.

## Runtime switches

| variable | default | effect |
|---|---|---|
| `GGML_CUDA_TQ_MMVQ_ROWS` | on up to Ada Lovelace, off on compute capability 12.0+ | several matrix rows per thread block in the token-generation mat-vec, so each thread has independent rows in flight |
| `GGML_CUDA_TQ_MMVQ_WIDE` | same as above | 16 rows per thread block when the inner dimension is small (K ≤ 1024, e.g. MoE expert down projections) |
| `GGML_CUDA_TQ_MMQ` | on | `0` sends trellis prompt processing through dequantize + cuBLAS instead of MMQ |

For `GGML_CUDA_TQ_MMVQ_ROWS` and `GGML_CUDA_TQ_MMVQ_WIDE`: unset means the
per-GPU default; `0` forces the layout off and any other value forces it on, on
any GPU. The variables are read once per process.

**The two mat-vec switches are independent.** `GGML_CUDA_TQ_MMVQ_ROWS=0` does
not turn off the wide layout: small-K matrices still take it. To get the
original one-row-per-block kernel everywhere, set **both** to `0`:

```sh
GGML_CUDA_TQ_MMVQ_ROWS=0 GGML_CUDA_TQ_MMVQ_WIDE=0 ./build/bin/llama-server -m model.gguf
```

## Why the defaults differ by GPU

| GPU | layouts on vs both off |
|---|---|
| RTX 3090 (Ampere, sm_86) | expert mat-vec kernel time: gate/up −41 %, down −24 % |
| RTX 4090 (Ada, sm_89) | expert mat-vec kernel time: gate/up −31 % (12.5 → 8.6 µs), down −7 % (13.5 → 12.5 µs) |
| RTX 5090 (Blackwell, sm_120) | 0.3-0.5 % slower end to end, so the default is off |

The cutoff is compute capability 12.0 (RTX 50-series, RTX PRO Blackwell,
DGX Spark). Data-centre Blackwell (compute capability 10.x) keeps the layouts
on by default; it has not been measured, so try both settings there.

## Checking a setting

`test-backend-ops perf -o MUL_MAT -p tqk6` (or `-o MUL_MAT_ID`) shows the
kernel time per shape; `llama-bench -p 0 -n 128` shows the end-to-end effect on
token generation. Run each setting in a fresh process.
