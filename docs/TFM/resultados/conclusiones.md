# Conclusiones por kernel y DSL: GB10 frente a RTX 5090

*Generado por `benchmarks/conclusiones.py`: una sola vista con lo que reparten [resumen.md](resumen.md), [ncu.md](ncu.md), [energia.md](energia.md) y [productividad.md](productividad.md). Cada celda: **rendimiento · % del pico** (mayor tamaño medido) · error numérico (o la validación que pasó, si el benchmark no guarda el error) · métricas de Nsight Compute (SM y memoria = *Speed of Light*; tensor = ciclos con el pipe de tensor ocupado; ocup. = ocupación lograda) y **qué limita** al kernel (≥ 60 % en un pipe = cerca de su techo; si ninguno, latencia y su espera principal) · eficiencia energética · entorno y job. Productividad: líneas de código y 1ª llamada en frío (compilación + autotuning) en GB10.*

*% del pico: en compute-bound, sobre el pico de `mma.sync` medido en esa GPU con `run_pico_mma` (acumulación FP32; en GB10, 122.7 TFLOP/s fp16 y 243.3 FP8, en ráfaga a ~2.5 GHz) o, si no se ha medido, sobre la ficha de NVIDIA (RTX 5090: 209.5 fp16 y 419 FP8 con acumulación FP32, al boost de 2 407 MHz). Por encima del 100 % de la ficha, la GPU va más rápida que el boost o no aplica el recorte de la acumulación FP32 (pendiente: `run_pico_mma` en pascal). En memory-bound, sobre el ancho de banda de la ficha. n/a = la variante no existe para ese algoritmo.*

> **Faltan las métricas de Nsight Compute**: aún no hay ejecuciones con `TFM_NCU=1` (`TFM_NCU=1 bash ejecutar.sh encolar todo`). Al terminar, este documento se regenera solo.

## Matmul fp16 (CB)

- **GB10**: mejor DSL **Triton + TMA** (94.62 TFLOP/s, 77 % del pico mma.sync medido), 102% de la referencia (76 %); el más lento, CUTLASS (19 %).
- **RTX 5090**: mejor DSL **Triton + TMA** (223.6 TFLOP/s, 107 % del pico ficha NVIDIA), 107% de la referencia (100 %); el más lento, CUTLASS (79 %).
- RTX 5090/GB10: de ×2.25 (Referencia) a ×7.03 (CUTLASS).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **89.89 TFLOP/s** · **73 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>1019.6 GFLOP/J (×0.97 ref.)<br>_hennessy, job 21058_ | **217.89 TFLOP/s** · **104 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.42 | 67 LoC · 34.1 s<br>autotune (triton.autotune) |
| **Triton (block-ptr)** | **91.71 TFLOP/s** · **75 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>1036 GFLOP/J (×0.99 ref.)<br>_hennessy, job 21058_ | **217.6 TFLOP/s** · **104 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.37 | — |
| **Triton + TMA** | **94.62 TFLOP/s** · **77 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>1064.1 GFLOP/J (×1.01 ref.)<br>_hennessy, job 21058_ | **223.6 TFLOP/s** · **107 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.36 | 68 LoC · 96.1 s<br>autotune (triton.autotune) |
| **Triton + TLX** | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — | — |
| **Gluon** | **77.45 TFLOP/s** · **63 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>860.6 GFLOP/J (×0.82 ref.)<br>_hennessy, job 21058_ | — *no medido*: no se ha ejecutado en esta GPU | — | 101 LoC · 3.9 s<br>barrido propio (sin autotuning) |
| **Helion** | **61.94 TFLOP/s** · **50 %** (solo medido con caché fría)<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_hennessy-frio, job 20841_ | — *no medido*: no se ha ejecutado en esta GPU | — | 25 LoC · 4.0 min<br>autotune Helion (quick) |
| **CUTLASS** | **23.59 TFLOP/s** · **19 %** (solo medido con caché fría)<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>573.2 GFLOP/J (×0.55 ref.)<br>_hennessy-frio, job 20841_ | **165.83 TFLOP/s** · **79 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×7.03 | 713 LoC · 1.0 s<br>fija (tile del ejemplo) |
| **Referencia** | **92.96 TFLOP/s** · **76 %**<br>es la referencia<br>ncu: *sin perfil*<br>1051.1 GFLOP/J<br>_hennessy, job 21058_ | **209.14 TFLOP/s** · **100 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.25 | — |

*Tamaño: GB10 8192×8192×8192; RTX 5090 8192×8192×8192.*

## Matmul FP8 (CB)

- **GB10**: mejor DSL **Triton + TMA** (150.05 TFLOP/s, 62 % del pico mma.sync medido), 81% de la referencia (76 %); el más lento, Triton (59 %).
- **RTX 5090**: mejor DSL **Triton** (394.76 TFLOP/s, 94 % del pico ficha NVIDIA), 66% de la referencia (143 %); el más lento, Triton + TMA (94 %).
- RTX 5090/GB10: de ×2.63 (Triton + TMA) a ×3.24 (Referencia).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **143.4 TFLOP/s** · **59 %**<br>error 3.75%<br>ncu: *sin perfil*<br>1585.8 GFLOP/J (×0.78 ref.)<br>_hennessy, job 21058_ | **394.76 TFLOP/s** · **94 %**<br>error 3.75%<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.75 | — |
| **Triton + TMA** | **150.05 TFLOP/s** · **62 %**<br>error 3.75%<br>ncu: *sin perfil*<br>1629.2 GFLOP/J (×0.8 ref.)<br>_hennessy, job 21058_ | **394.47 TFLOP/s** · **94 %**<br>error 3.75%<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.63 | — |
| **Triton + TLX** | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — | — |
| **Gluon** | — *sin kernel*: no hay matmul FP8 en este DSL | — *sin kernel*: no hay matmul FP8 en este DSL | — | — |
| **Helion** | — *sin kernel*: no hay matmul FP8 en este DSL | — *sin kernel*: no hay matmul FP8 en este DSL | — | — |
| **CUTLASS** | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | — | — |
| **Referencia** | **184.97 TFLOP/s** · **76 %**<br>error 3.75%<br>ncu: *sin perfil*<br>2036.5 GFLOP/J<br>_hennessy, job 21058_ | **598.67 TFLOP/s** · **143 %**<br>error 3.75%<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×3.24 | — |

*Tamaño: GB10 8192×8192×8192; RTX 5090 8192×8192×8192.*

## FlashAttention fp16 (CB)

- **GB10**: mejor DSL **CUTLASS** (91.39 TFLOP/s, 74 % del pico mma.sync medido), 111% de la referencia (67 %); el más lento, Helion (55 %).
- **RTX 5090**: mejor DSL **CUTLASS** (164.64 TFLOP/s, 79 % del pico ficha NVIDIA), 108% de la referencia (73 %); el más lento, Helion (72 %).
- RTX 5090/GB10: de ×1.80 (CUTLASS) a ×2.22 (Helion).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **87.96 TFLOP/s** · **72 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>913.6 GFLOP/J (×1.08 ref.)<br>_hennessy, job 21058_ | **162.82 TFLOP/s** · **78 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.85 | 220 LoC · 15.2 s<br>autotune (triton.autotune) |
| **Triton + TLX** | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — | — |
| **Gluon** | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — | — |
| **Helion** | **67.48 TFLOP/s** · **55 %**<br>error 4.5e-04<br>ncu: *sin perfil*<br>875.2 GFLOP/J (×1.03 ref.)<br>_hennessy, job 21058_ | **149.88 TFLOP/s** · **72 %**<br>error 4.5e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×2.22 | 63 LoC · 2.1 min<br>autotune Helion (quick) |
| **CUTLASS** | **91.39 TFLOP/s** · **74 %**<br>error 1.1e-04<br>ncu: *sin perfil*<br>930.3 GFLOP/J (×1.1 ref.)<br>_hennessy, job 20649_ | **164.64 TFLOP/s** · **79 %**<br>error 1.1e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.80 | 778 LoC · 7.7 s<br>barrido propio (sin autotuning) |
| **Referencia** | **82.19 TFLOP/s** · **67 %**<br>es la referencia<br>ncu: *sin perfil*<br>846.6 GFLOP/J<br>_hennessy, job 21058_ | **152.93 TFLOP/s** · **73 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.86 | — |

*Tamaño: GB10 B=4 H=32 N=16384 D=128; RTX 5090 B=4 H=32 N=16384 D=128.*

## FlashAttention causal fp16 (CB)

- **GB10**: mejor DSL **CUTLASS** (90.8 TFLOP/s, 74 % del pico mma.sync medido), 108% de la referencia (69 %); el más lento, Helion (36 %).
- **RTX 5090**: mejor DSL **CUTLASS** (162.18 TFLOP/s, 77 % del pico ficha NVIDIA), 106% de la referencia (73 %); el más lento, Helion (36 %).
- RTX 5090/GB10: de ×1.73 (Helion) a ×1.85 (Triton).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **84.81 TFLOP/s** · **69 %**<br>error 2.8e-04<br>ncu: *sin perfil*<br>853.6 GFLOP/J (×1 ref.)<br>_hennessy, job 21058_ | **157.2 TFLOP/s** · **75 %**<br>error 2.8e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.85 | — |
| **Triton + TLX** | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — | — |
| **Gluon** | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — | — |
| **Helion** | **43.9 TFLOP/s** · **36 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>425.2 GFLOP/J (×0.5 ref.)<br>_hennessy, job 21058_ | **75.85 TFLOP/s** · **36 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.73 | — |
| **CUTLASS** | **90.8 TFLOP/s** · **74 %**<br>error 1.3e-04<br>ncu: *sin perfil*<br>899.8 GFLOP/J (×1.06 ref.)<br>_hennessy, job 20649_ | **162.18 TFLOP/s** · **77 %**<br>error 2.0e-05<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.79 | — |
| **Referencia** | **84.21 TFLOP/s** · **69 %**<br>es la referencia<br>ncu: *sin perfil*<br>851 GFLOP/J<br>_hennessy, job 21058_ | **152.36 TFLOP/s** · **73 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.81 | — |

*Tamaño: GB10 B=4 H=32 N=16384 D=128; RTX 5090 B=4 H=32 N=16384 D=128.*

## RMSNorm fp16 (MB)

- **GB10**: mejor DSL **Gluon** (242.8 GB/s, 89 % del pico de memoria), 112% de la referencia (80 %); el más lento, Helion (84 %).
- **RTX 5090**: mejor DSL **Triton** (1521.4 GB/s, 85 % del pico de memoria), 117% de la referencia (72 %); el más lento, CUTLASS (80 %).
- RTX 5090/GB10: de ×5.98 (Referencia) a ×6.27 (Triton).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **242.7 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.23 GB/J (×1.05 ref.)<br>_hennessy, job 21058_ | **1521.4 GB/s** · **85 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.27 | 38 LoC · 0.6 s<br>heurística (warps por N) |
| **Triton + TLX** | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | — | 52 LoC<br>heurística + --stages a mano |
| **Gluon** | **242.8 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.37 GB/J (×1.07 ref.)<br>_hennessy, job 21058_ | **1521.2 GB/s** · **85 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.27 | 35 LoC · 0.6 s<br>heurística (warps y layout por N) |
| **Helion** | **230.3 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.5 GB/J (×1.09 ref.)<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — | 16 LoC · 77.3 s<br>autotune Helion (quick) |
| **CUTLASS** | **230.5 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6 GB/J (×1.01 ref.)<br>_patterson, job 20435_ | **1438.7 GB/s** · **80 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×6.24 | 203 LoC · 55.5 s<br>fija (heurística de CUTLASS) |
| **Referencia** | **217.1 GB/s** · **80 %**<br>es la referencia<br>ncu: *sin perfil*<br>5.95 GB/J<br>_hennessy, job 21058_ | **1299.3 GB/s** · **72 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×5.98 | — |

*Tamaño: GB10 16384×8192; RTX 5090 16384×8192.*

## RMSNorm bf16 (MB)

- **GB10**: mejor DSL **Triton** (242.8 GB/s, 89 % del pico de memoria), 112% de la referencia (79 %); el más lento, CUTLASS (78 %).
- **RTX 5090**: mejor DSL **Triton** (1513.4 GB/s, 84 % del pico de memoria), 116% de la referencia (73 %); el más lento, CUTLASS (78 %).
- RTX 5090/GB10: de ×6.01 (Referencia) a ×6.62 (CUTLASS).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **242.8 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.27 GB/J (×1.04 ref.)<br>_hennessy, job 21058_ | **1513.4 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.23 | — |
| **Triton + TLX** | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | — | — |
| **Gluon** | **242.8 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.37 GB/J (×1.06 ref.)<br>_hennessy, job 21058_ | **1513.4 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.23 | — |
| **Helion** | **226.6 GB/s** · **83 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.47 GB/J (×1.07 ref.)<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — | — |
| **CUTLASS** | **211.6 GB/s** · **78 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>4.68 GB/J (×0.78 ref.)<br>_patterson, job 20435_ | **1400.1 GB/s** · **78 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×6.62 | — |
| **Referencia** | **216.8 GB/s** · **79 %**<br>es la referencia<br>ncu: *sin perfil*<br>6.02 GB/J<br>_hennessy, job 21058_ | **1302.8 GB/s** · **73 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×6.01 | — |

*Tamaño: GB10 16384×8192; RTX 5090 16384×8192.*

