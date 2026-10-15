# Kernels: procedencia y resultados

*Generado por `benchmarks/kernels.py`. Para cada DSL: de dónde sale el kernel (detalle en [procedencia_kernels.md](../procedencia_kernels.md)) y su último resultado en cada arquitectura (el mismo que el resumen: [resumen.md](resumen.md), con la configuración aplicada). Cada resultado: rendimiento en el mayor tamaño medido · **% del pico** · entorno y job. Una celda sin dato dice por qué falta; n/a = la variante no existe para ese algoritmo.*

*% del pico: en compute-bound, sobre el pico de `mma.sync` medido en esa GPU con `run_pico_mma` (acumulación FP32; en GB10, 122.7 TFLOP/s fp16 y 243.3 FP8, en ráfaga a ~2.5 GHz) o, si no se ha medido, sobre la ficha de NVIDIA (RTX 5090: 209.5 fp16 y 419 FP8 con acumulación FP32, al boost de 2 407 MHz). Por encima del 100 % de la ficha, la GPU va más rápida que el boost o no aplica el recorte de la acumulación FP32 (pendiente: `run_pico_mma` en pascal). En memory-bound, sobre el ancho de banda de la ficha. n/a = la variante no existe para ese algoritmo.*

## Matmul

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 FP8 | RTX 5090 fp16 | RTX 5090 FP8 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | *fp16*: `triton/matmul.py` (baseline)<br>*FP8*: `triton/matmul_fp8.py` | Adaptado | *fp16*: Tutorial oficial de Triton `03-matrix-multiplication.py` [1]<br>*FP8*: Esqueleto de `matmul.py`; el tutorial 03 ya muestra entradas FP8 [1] | **89.89 TFLOP/s** · **73 %**<br>_hennessy, job 21058_ | **143.4 TFLOP/s** · **59 %**<br>_hennessy, job 21058_ | **217.89 TFLOP/s** · **104 %**<br>_pascal, job 20499_ | **394.76 TFLOP/s** · **94 %**<br>_pascal, job 20499_ |
| **Triton (block-ptr)** | `triton/matmul_blockptr.py` | Adaptado | Esqueleto de `matmul.py` + API `tl.make_block_ptr` / `tl.advance` de Triton [1] | **91.71 TFLOP/s** · **75 %**<br>_hennessy, job 21058_ | n/a | **217.6 TFLOP/s** · **104 %**<br>_pascal, job 20499_ | n/a |
| **Triton + TMA** | *fp16*: `triton/matmul_tma.py`<br>*FP8*: `triton/matmul_fp8_tma.py` | *fp16*: Adaptado<br>*FP8*: Propio (combinación) | *fp16*: Tutorial oficial de Triton `09-persistent-matmul.py`, variantes con TMA [1]<br>*FP8*: `matmul_fp8.py` + descriptores TMA de `matmul_tma.py` | **94.62 TFLOP/s** · **77 %**<br>_hennessy, job 21058_ | **150.05 TFLOP/s** · **62 %**<br>_hennessy, job 21058_ | **223.6 TFLOP/s** · **107 %**<br>_pascal, job 20499_ | **394.47 TFLOP/s** · **94 %**<br>_pascal, job 20499_ |
| **Triton + TLX** | — | sin kernel | — | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX |
| **Gluon** | `gluon/matmul.py` | Propio | Piezas oficiales de Triton v3.8.0: `mma_v2` (`test_core.py::test_mma_v2`) y pipeline `cp.async` (`gluon/03-async-copy.py`) [1] | **77.45 TFLOP/s** · **63 %**<br>_hennessy, job 21058_ | — *sin kernel*: no hay matmul FP8 en este DSL | — *no medido*: no se ha ejecutado en esta GPU | — *sin kernel*: no hay matmul FP8 en este DSL |
| **Helion** | `helion/matmul.py` | Adaptado | Ejemplo oficial de Helion `examples/matmul.py` (`matmul`) [3] | **61.94 TFLOP/s** · **50 %** (solo medido con caché fría)<br>_hennessy-frio, job 20841_ | — *sin kernel*: no hay matmul FP8 en este DSL | — *no medido*: no se ha ejecutado en esta GPU | — *sin kernel*: no hay matmul FP8 en este DSL |
| **CUTLASS** | `cutlass/matmul_cute.py` | Oficial sin modificar | `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py` (`Sm120GemmKernel`) [2] | **23.59 TFLOP/s** · **19 %** (solo medido con caché fría)<br>_hennessy-frio, job 20841_ | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | **165.83 TFLOP/s** · **79 %**<br>_pascal, job 20638_ | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) |
| **Referencia** | *fp16*: `torch.matmul`<br>*FP8*: `torch._scaled_mm` | Referencia (no es kernel propio) | *fp16*: cuBLAS (fp16/bf16; fp32 en TF32)<br>*FP8*: cuBLASLt en FP8 | **92.96 TFLOP/s** · **76 %**<br>_hennessy, job 21058_ | **184.97 TFLOP/s** · **76 %**<br>_hennessy, job 21058_ | **209.14 TFLOP/s** · **100 %**<br>_pascal, job 20499_ | **598.67 TFLOP/s** · **143 %**<br>_pascal, job 20499_ |

## FlashAttention (forward)

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 causal | RTX 5090 fp16 | RTX 5090 causal |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `triton/fused_attention.py` | Oficial sin modificar (copia literal) | Tutorial oficial `06-fused-attention.py` [1] · ¿corre en sm_120/121? ✅ | **87.96 TFLOP/s** · **72 %**<br>_hennessy, job 21058_ | **84.81 TFLOP/s** · **69 %**<br>_hennessy, job 21058_ | **162.82 TFLOP/s** · **78 %**<br>_pascal, job 20648_ | **157.2 TFLOP/s** · **75 %**<br>_pascal, job 20648_ |
| **Triton + TLX** | — | no hay oficial ejecutable | tutoriales `tlx/tutorials/{hopper,blackwell}_fa_ws*.py` [4] · ¿corre en sm_120/121? ❌ ver procedencia_kernels.md | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) |
| **Gluon** | — | no hay oficial ejecutable | `python/examples/gluon/01-attention-forward.py` [1] usa `tcgen05` · ¿corre en sm_120/121? ❌ `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st |
| **Helion** | `helion/attention.py` → `examples/attention.py` | Oficial sin modificar | `attention_output` / `causal_attention_output` [3] · ¿corre en sm_120/121? ✅ | **67.48 TFLOP/s** · **55 %**<br>_hennessy, job 21058_ | **43.9 TFLOP/s** · **36 %**<br>_hennessy, job 21058_ | **149.88 TFLOP/s** · **72 %**<br>_pascal, job 20648_ | **75.85 TFLOP/s** · **36 %**<br>_pascal, job 20648_ |
| **CUTLASS** | `cutlass/attention_fa2.py` → `flash_attention_v2.py` | Oficial sin modificar | `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py` (`FlashAttentionForwardAmpere`) [2] · ¿corre en sm_120/121? ✅ | **91.39 TFLOP/s** · **74 %**<br>_hennessy, job 20649_ | **90.8 TFLOP/s** · **74 %**<br>_hennessy, job 20649_ | **164.64 TFLOP/s** · **79 %**<br>_pascal, job 20648_ | **162.18 TFLOP/s** · **77 %**<br>_pascal, job 20648_ |
| **Referencia** | `torch.nn.functional.scaled_dot_product_attention` | Referencia (no es kernel propio) | FlashAttention-2 incluido en PyTorch (`pytorch_flash::flash_fwd_kernel`); también es la referencia de corrección | **82.19 TFLOP/s** · **67 %**<br>_hennessy, job 21058_ | **84.21 TFLOP/s** · **69 %**<br>_hennessy, job 21058_ | **152.93 TFLOP/s** · **73 %**<br>_pascal, job 20648_ | **152.36 TFLOP/s** · **73 %**<br>_pascal, job 20648_ |

## RMSNorm

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 bf16 | RTX 5090 fp16 | RTX 5090 bf16 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `triton/rmsnorm_baseline.py` | Adaptado | Tutorial oficial de Triton `05-layer-norm.py` (`_layer_norm_fwd_fused`) [1] | **242.7 GB/s** · **89 %**<br>_hennessy, job 21058_ | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **1521.4 GB/s** · **85 %**<br>_pascal, job 20499_ | **1513.4 GB/s** · **84 %**<br>_pascal, job 20499_ |
| **Triton + TLX** | `triton_tlx/rmsnorm.py` | Propio | API de µTLX (`triton-lang/triton-ext`, `extensions/utlx`) [4] | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>_pascal, job 20499_ | **1456.4 GB/s** · **81 %**<br>_pascal, job 20499_ |
| **Gluon** | `gluon/rmsnorm.py` | Propio | API de Gluon, tutorial oficial `gluon/02-layouts.py` [1] | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **1521.2 GB/s** · **85 %**<br>_pascal, job 20499_ | **1513.4 GB/s** · **84 %**<br>_pascal, job 20499_ |
| **Helion** | `helion/rmsnorm.py` | Adaptado | Ejemplo oficial de Helion `examples/rms_norm.py` (`rms_norm_fwd`) [3] | **230.3 GB/s** · **84 %**<br>_patterson, job 20435_ | **226.6 GB/s** · **83 %**<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — *no medido*: no se ha ejecutado en esta GPU |
| **CUTLASS** | `cutlass/rmsnorm_ext.cu` | Oficial sin modificar | `cutlass::rmsnorm`, `tools/util/include/cutlass/util/device_rmsnorm.h` [2] | **230.5 GB/s** · **84 %**<br>_patterson, job 20435_ | **211.6 GB/s** · **78 %**<br>_patterson, job 20435_ | **1438.7 GB/s** · **80 %**<br>_pascal, job 20638_ | **1400.1 GB/s** · **78 %**<br>_pascal, job 20638_ |
| **Referencia** | `torch.nn.functional.rms_norm` | Referencia (no es kernel propio) | kernel RMSNorm de PyTorch; también es la referencia de corrección | **217.1 GB/s** · **80 %**<br>_hennessy, job 21058_ | **216.8 GB/s** · **79 %**<br>_hennessy, job 21058_ | **1299.3 GB/s** · **72 %**<br>_pascal, job 20638_ | **1302.8 GB/s** · **73 %**<br>_pascal, job 20638_ |

## Fuentes

| # | proyecto | versión | licencia |
|:---|:---|:---|:---|
| [1] | Triton, `triton-lang/triton`, `python/tutorials/` | tag v3.8.0 (`c01b677`) | MIT |
| [2] | CUTLASS, `NVIDIA/cutlass` | tag v4.8.0 (`098de2a`) | BSD-3-Clause |
| [3] | Helion, `pytorch/helion`, `examples/rms_norm.py`, `examples/attention.py` | `main` (`0f3b242`, 2026-09-30) | BSD-3-Clause |
| [4] | µTLX, `triton-lang/triton-ext`, `extensions/utlx` (paquete `triton-utlx` 3.8.0.post1) | `main` (`3c2bb3c`) | MIT |
| [5] | T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691 (ICLR 2024) | — | — |

Las cuatro licencias permiten reutilizar y modificar el código conservando el aviso de copyright.
