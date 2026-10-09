"""Baseline de rendimiento: matmul en Triton con @triton.autotune.

Para cada tamano, el autotuner prueba todas las configuraciones (BLOCK_SIZE_M/N/K,
GROUP_SIZE_M, num_warps, num_stages) y se queda con la mas rapida. Despues se mide
esa configuracion con triton.testing.do_bench y se compara con cuBLAS (torch.matmul).

Uso (en hennessy):
    docker run --rm --gpus all -v ~/tfm:/workspace/tfm tfm:ngc-arm64 \
        python /workspace/tfm/baseline_matmul.py [--sizes 4096 8192] [--dtype fp16|bf16]
Resultados: results/baseline_matmul_<dtype>.csv y .json
"""
import argparse
import csv
import itertools
import json
import os
import time

import torch
import triton
import triton.language as tl


def configs_autotune():
    """Espacio de busqueda: 3 x 3 x 2 x 2 x 2 = 72 combinaciones, menos las podadas."""
    configs = []
    for bm, bn, bk, warps, stages in itertools.product(
        [64, 128, 256],  # BLOCK_SIZE_M
        [64, 128, 256],  # BLOCK_SIZE_N
        [32, 64],        # BLOCK_SIZE_K
        [4, 8],          # num_warps
        [3, 4],          # num_stages
    ):
        if bm * bn >= 256 * 256:  # 256x256 no cabe en registros/memoria compartida
            continue
        configs.append(triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
            num_warps=warps, num_stages=stages))
    return configs


# Las configuraciones que no caben en memoria compartida (OutOfResources) las
# descarta el propio autotuner. key: se re-afina para cada (M, N, K).
@triton.autotune(configs=configs_autotune(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Reordenacion en grupos de GROUP_SIZE_M filas de bloques para mejorar la reutilizacion en L2.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Acumulacion en fp32 para precision.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "dimensiones incompatibles"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c


def medir(fn, M, N, K, elem_bytes):
    """Devuelve (ms mediana, TFLOP/s, GB/s efectivos)."""
    ms, ms_min, ms_max = triton.testing.do_bench(fn, warmup=100, rep=500, quantiles=[0.5, 0.2, 0.8])
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    gbs = (M * K + K * N + M * N) * elem_bytes / (ms * 1e-3) / 1e9
    return ms, tflops, gbs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192], help="tamanos N (matrices N x N)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu} (sm_{cap[0]}{cap[1]}) | torch {torch.__version__} | triton {triton.__version__} | dtype {args.dtype}")
    print(f"Espacio de autotuning: {len(matmul_kernel.configs)} configuraciones\n")

    torch.manual_seed(0)
    filas = []
    for n in args.sizes:
        M = N = K = n
        a = torch.randn((M, K), device="cuda", dtype=dtype)
        b = torch.randn((K, N), device="cuda", dtype=dtype)

        # 1a llamada: dispara el autotuning (compila y mide todas las configuraciones).
        t0 = time.perf_counter()
        c = matmul(a, b)
        torch.cuda.synchronize()
        t_autotune = time.perf_counter() - t0

        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)  # tolerancia para fp16/bf16 con K grande

        cfg = matmul_kernel.best_config
        ms_t, tf_t, gb_t = medir(lambda: matmul(a, b), M, N, K, elem_bytes)
        ms_c, tf_c, gb_c = medir(lambda: torch.matmul(a, b), M, N, K, elem_bytes)

        print(f"== {M}x{K} @ {K}x{N} ({args.dtype}) — autotuning {t_autotune:.1f} s")
        print(f"   Mejor config: {cfg.kwargs}, num_warps={cfg.num_warps}, num_stages={cfg.num_stages}")
        print(f"   Triton : {ms_t:8.3f} ms  {tf_t:7.2f} TFLOP/s  {gb_t:8.1f} GB/s")
        print(f"   cuBLAS : {ms_c:8.3f} ms  {tf_c:7.2f} TFLOP/s  {gb_c:8.1f} GB/s")
        print(f"   Triton/cuBLAS: {tf_t / tf_c:.1%}\n")

        filas.append({
            "gpu": gpu, "sm": f"{cap[0]}{cap[1]}", "dtype": args.dtype, "M": M, "N": N, "K": K,
            "BLOCK_SIZE_M": cfg.kwargs["BLOCK_SIZE_M"], "BLOCK_SIZE_N": cfg.kwargs["BLOCK_SIZE_N"],
            "BLOCK_SIZE_K": cfg.kwargs["BLOCK_SIZE_K"], "GROUP_SIZE_M": cfg.kwargs["GROUP_SIZE_M"],
            "num_warps": cfg.num_warps, "num_stages": cfg.num_stages, "autotune_s": round(t_autotune, 1),
            "triton_ms": round(ms_t, 4), "triton_tflops": round(tf_t, 2), "triton_gbs": round(gb_t, 1),
            "cublas_ms": round(ms_c, 4), "cublas_tflops": round(tf_c, 2), "cublas_gbs": round(gb_c, 1),
            "torch": torch.__version__, "triton": triton.__version__,
        })

    mejor = max(filas, key=lambda f: f["triton_tflops"])
    print(f"*** BASELINE (rival a batir): {mejor['triton_tflops']} TFLOP/s "
          f"en {mejor['M']}x{mejor['N']}x{mejor['K']} {args.dtype} "
          f"(cuBLAS: {mejor['cublas_tflops']} TFLOP/s) ***")

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, f"baseline_matmul_{args.dtype}")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas)
    with open(base + ".json", "w") as f:
        json.dump({"baseline": mejor, "resultados": filas}, f, indent=2)
    print(f"Resultados guardados en {base}.csv y .json")


if __name__ == "__main__":
    main()
