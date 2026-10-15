# Comparativa de kernels y DSLs en Blackwell GeForce

*Documento único generado por `benchmarks/comparativa.py` a partir de los resultados (`results/`) y de [procedencia_kernels.md](procedencia_kernels.md); se regenera tras cada ejecución. Reúne la metodología, la procedencia de cada kernel, su rendimiento y % del pico en cada GPU, su perfil de Nsight Compute, la eficiencia energética y la productividad.*

## Índice

1. [Alcance](#1-alcance)
2. [Metodología](#2-metodología)
3. [Resultados por algoritmo](#3-resultados-por-algoritmo) (procedencia, comparación y conclusiones, perfil de Nsight Compute)
4. [Vista global](#4-vista-global)
5. [Fuentes](#5-fuentes)

## 1. Alcance

| | |
|:---|:---|
| **Algoritmos** | Matmul (fp16 y FP8), FlashAttention forward (fp16, causal y no causal), RMSNorm (fp16 y bf16) |
| **DSLs** | Triton (con sus variantes block-ptr y TMA), Triton + TLX, Gluon, Helion, CUTLASS (CuTe DSL y C++); referencia: PyTorch (cuBLAS, cuBLASLt, SDPA, `F.rms_norm`) |
| **GPUs** | NVIDIA GB10 (sm_121), NVIDIA GeForce RTX 5090 (sm_120) |

## 2. Metodología

### 2.1 Principio de la comparación

Para cada algoritmo, **todos los DSLs pasan por el mismo banco de medida**
(`benchmarks/banco_matmul.py`, `banco_attention.py`, `banco_rmsnorm.py`): mismas formas, misma
validación, misma medida y mismo guardado. Cada DSL solo aporta cómo preparar su kernel. Así, una
diferencia de rendimiento entre DSLs se debe al kernel y no a cómo se midió. En cada GPU, la
**referencia** es la implementación de PyTorch del mismo algoritmo (cuBLAS, cuBLASLt, SDPA,
`F.rms_norm`), medida en la misma ejecución.

### 2.2 Plataforma y picos de referencia

| GPU | nodos | pico CB fp16 | pico CB FP8 | pico MB (memoria) |
|:---|:---|:---|:---|:---|
| NVIDIA GB10 (sm_121) | hennessy, patterson | 122.83 TFLOP/s (mma.sync medido, hennessy, job 21058) | 243.31 TFLOP/s (mma.sync medido, hennessy, job 21058) | 273 GB/s (ficha NVIDIA) |
| NVIDIA GeForce RTX 5090 (sm_120) | pascal | 209.5 TFLOP/s (ficha NVIDIA) | 419 TFLOP/s (ficha NVIDIA) | 1792 GB/s (ficha NVIDIA) |

Los dos chips son Blackwell GeForce: no tienen `wgmma` ni `tcgen05`, así que todos los DSLs
acaban emitiendo `mma.sync` y **el techo de los compute-bound lo pone esa instrucción**. Por eso,
en GB10 el pico de cómputo es el de `mma.sync` medido con un microbenchmark propio
(`run_pico_mma`, acumulación en FP32, § [pico_mma.md](pico_mma.md)); NVIDIA no publica pico de
fp16 ni de FP8 para GB10. En RTX 5090 se usa la ficha de NVIDIA hasta que se mida
(`run_pico_mma` en pascal). En memory-bound, el pico es el ancho de banda de la ficha. Detalle
del hardware y del software: [plataforma.md](plataforma.md).

### 2.3 Kernels y su procedencia

Cada kernel se clasifica según cuánto código propio lleva (§ [procedencia_kernels.md](procedencia_kernels.md)):
**oficial sin modificar** (el de la fuente, con solo la envoltura para medirlo), **adaptado** (parte
de un ejemplo público, con los cambios documentados) o **propio** (escrito sobre la API documentada).
Siempre que existe, se usa el kernel oficial del proyecto. Así se compara lo que cada DSL ofrece de
serie. Una celda sin kernel dice por qué: no existe en ese DSL, o el oficial no compila en sm_120/121.

### 2.4 Cargas de trabajo

| algoritmo | formas | trabajo (FLOP) / tráfico (bytes) |
|:---|:---|:---|
| Matmul fp16 y FP8 | M = N = K ∈ {4096, 8192} | 2·M·N·K FLOP |
| FlashAttention (forward) | B = 4, H = 32, N ∈ {1024, 4096, 16384}, D ∈ {64, 128}, causal y no causal | 4·B·H·N²·D FLOP (×0.5 si causal), como el tutorial de Triton |
| RMSNorm fp16 y bf16 | M ∈ {4096, 16384} filas × N ∈ {2048, 4096, 8192} | (2·M·N + N)·bytes por elemento (leer x y w, escribir y) |

Las tablas de resultados dan el **mayor tamaño medido** de cada celda (matmul 8192³, atención
N = 16384 con D = 128, RMSNorm 16384×8192); los informes de cada experimento tienen todos.

### 2.5 Corrección antes de medir

Ningún kernel se mide sin validarlo antes contra PyTorch (`benchmarks/validation.py`): matmul con
atol 0.1 y rtol 0.01 (acumular en otro orden que cuBLAS con K grande da ~0.1 absoluto), atención
con atol 0.01 frente a SDPA, RMSNorm con atol y rtol 0.02 frente a `F.rms_norm` en fp32. En FP8 no
se exige tolerancia: se informa del error relativo de Frobenius. En los kernels Triton se comprueba
además en el PTX que el cálculo va a **tensor cores** (`mma.sync`/`wgmma`/`tcgen05.mma`); si no,
la ejecución aborta.

### 2.6 Medida del rendimiento

- **Tiempo:** `triton.testing.do_bench`, 100 ms de calentamiento y 500 ms de repetición; se toma la
  **mediana** (y los percentiles 20 y 80). `do_bench` vacía la L2 entre repeticiones.
- **Compilación y autotuning:** fuera de la medida. La primera llamada se cronometra aparte
  (`t_compilacion_s`). Los DSLs sin autotuning (Gluon, CuTe) barren unas pocas configuraciones y
  usan la más rápida.
- **Relojes:** no se fijan (GB10 no lo permite); se **registran** el reloj SM y la potencia
  durante cada medida (NVML cada 20 ms, `reloj_mhz` y `potencia_w`).
- **Aislamiento:** cada experimento corre en un contenedor efímero con la GPU entera; la cola
  `-benchmark` de cada nodo ejecuta un trabajo cada vez. Antes de medir, `ejecutar.sh` para los
  contenedores que hayan quedado de intentos anteriores y avisa si hay otros procesos en la GPU.
  Las ejecuciones con la GPU compartida se apartan a `descartados/` con su motivo (job 21058).

### 2.7 Métricas y normalización

| tipo | métrica | % del pico |
|:---|:---|:---|
| **CB** (compute-bound: matmul, atención) | TFLOP/s = FLOP / tiempo | sobre el pico de `mma.sync` medido en esa GPU y precisión (si no hay, el de la ficha) |
| **MB** (memory-bound: RMSNorm) | GB/s = bytes / tiempo | sobre el ancho de banda de la ficha |

El % del pico es la medida principal de la comparación: dice cuánto del hardware aprovecha cada
kernel y permite comparar GPUs de tamaño muy distinto. Un % por encima de 100 de la ficha indica
que la GPU va por encima del reloj boost de la ficha o que no aplica el recorte de la acumulación
en FP32.

### 2.8 Qué dato se publica

Cada celda es la **última ejecución** de ese benchmark en esa arquitectura, en el mayor tamaño
medido (si hay dos nodos con la misma GPU, como hennessy y patterson en GB10, cuenta la más
reciente). No entran en las tablas de rendimiento los entornos especiales: `*-frio` (caché de
compilación vacía; solo si no hay otro dato), `*-energia` (régimen sostenido), `*-ncu` (bajo
Nsight Compute) ni las pruebas en dos nodos (`run_dist_matmul`). Una celda vacía dice su clase:
*no aplica* (la variante no existe en ese algoritmo, «n/a»), *sin kernel*, *no compila* o *no
medido*, con el motivo.

### 2.9 Perfil con Nsight Compute (por qué rinde lo que rinde)

Con `TFM_NCU=1`, cada benchmark se ejecuta bajo `ncu` y `comun.medir()` perfila **una llamada más**
de cada kernel guardado, ya compilado y caliente, marcada con un rango NVTX; la compilación, el
autotuning y las repeticiones de `do_bench` quedan fuera del perfil. Las métricas se unen a la fila
de su medida (`ncu_*`) y van a un entorno aparte (`<nodo>-ncu`), porque `ncu` repite cada kernel
varias veces para leer todos los contadores y su tiempo no sustituye al de `do_bench`. Se usan
las mismas métricas en las dos GPUs, salvo la memoria principal: GB10 no tiene contadores de DRAM
(memoria unificada LPDDR5X) y se mide como bytes que fallan en L2.

| métrica | contador de ncu | qué indica |
|:---|:---|:---|
| SM % | `sm__throughput` | *Speed of Light* de cómputo: uso de la unidad más cargada del SM |
| tensor % | `sm__pipe_tensor_cycles_active` | ciclos con los tensor cores ocupados: lo que separa un kernel CB bueno de uno malo |
| memoria % | `gpu__compute_memory_throughput` | *Speed of Light* de memoria (L1, L2 y DRAM) |
| emisión % | `smsp__issue_active` | ciclos en que el planificador emite una instrucción |
| ocupación % (lograda / teórica) | `sm__warps_active` / `sm__maximum_warps_per_active_cycle_pct` | warps activos frente al máximo; «limita» = recurso que la fija (registros, memoria compartida, warps, bloques) |
| regs/hilo, smem KB/bloque | `launch__registers_per_thread`, `launch__shared_mem_per_block` | recursos que pide el kernel |
| acierto L2 % | `lts__t_sector_hit_rate` | reutilización en L2 |
| memoria principal MB | `dram__bytes` (RTX 5090) / fallos de L2 (GB10) | tráfico real a memoria |
| conflictos smem | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | conflictos de bancos de memoria compartida |
| espera principal | `smsp__average_warps_issue_stalled_*_per_issue_active` | motivo con más ciclos de espera por instrucción (long scoreboard = memoria global, barrier = sincronización, mio throttle = memoria compartida, math pipe throttle = tensor cores saturados...) |

### 2.10 Energía y productividad

- **Eficiencia energética** (`TFM_ENERGIA=1`): potencia instantánea de la GPU en régimen sostenido
  (6 s en bucle, descartando los 2 primeros) → GFLOP/J en CB y GB/J en MB, también descontando la
  potencia en reposo. Método completo: [resultados/energia.md](resultados/energia.md).
- **Productividad** (`TFM_CACHE_FRIA=1`): líneas de código lógicas del kernel, coste del ajuste
  (tamaño del espacio y primera llamada con las cachés vacías) y rendimiento de esa misma
  ejecución. Método: [productividad.md](productividad.md).

### 2.11 Reproducibilidad

Cada resultado guarda su contexto en `meta.json`: GPU, driver, kernel de Linux, imagen (con su
id), versiones de torch/triton/helion/CUDA, job de Slurm y comando. Desde el login (ibsen):

```bash
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo                 # todos los benchmarks en hennessy
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar ambos todo           # hennessy y patterson a la vez
TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo       # con métricas de Nsight Compute
TFM_ENERGIA=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo   # con eficiencia energética
python3.11 ~/hennessy/tfm_entorno/benchmarks/comparativa.py          # regenerar este documento
```

## 3. Resultados por algoritmo

*Cada resultado: rendimiento en el mayor tamaño medido · **% del pico** · entorno y job. Una celda sin dato dice por qué falta; n/a = la variante no existe para ese algoritmo. Configuración aplicada en cada celda (bloques, warps, etapas...): [resultados/resumen.md](resultados/resumen.md).*

### 3.1 Matmul

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

#### Comparación y conclusiones: Matmul

*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al kernel según ncu · eficiencia energética · entorno y job; productividad en GB10 (líneas de código y 1ª llamada en frío). Generado con `benchmarks/conclusiones.py`.*

##### Matmul fp16 (CB)

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

##### Matmul FP8 (CB)

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

#### Perfil de Nsight Compute: Matmul

*Sin perfil todavía: `TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo` (y lo mismo en pascal para la RTX 5090).*

### 3.2 FlashAttention (forward)

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 causal | RTX 5090 fp16 | RTX 5090 causal |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `triton/fused_attention.py` | Oficial sin modificar (copia literal) | Tutorial oficial `06-fused-attention.py` [1] · ¿corre en sm_120/121? ✅ | **87.96 TFLOP/s** · **72 %**<br>_hennessy, job 21058_ | **84.81 TFLOP/s** · **69 %**<br>_hennessy, job 21058_ | **162.82 TFLOP/s** · **78 %**<br>_pascal, job 20648_ | **157.2 TFLOP/s** · **75 %**<br>_pascal, job 20648_ |
| **Triton + TLX** | — | no hay oficial ejecutable | tutoriales `tlx/tutorials/{hopper,blackwell}_fa_ws*.py` [4] · ¿corre en sm_120/121? ❌ ver procedencia_kernels.md | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) |
| **Gluon** | — | no hay oficial ejecutable | `python/examples/gluon/01-attention-forward.py` [1] usa `tcgen05` · ¿corre en sm_120/121? ❌ `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st |
| **Helion** | `helion/attention.py` → `examples/attention.py` | Oficial sin modificar | `attention_output` / `causal_attention_output` [3] · ¿corre en sm_120/121? ✅ | **67.48 TFLOP/s** · **55 %**<br>_hennessy, job 21058_ | **43.9 TFLOP/s** · **36 %**<br>_hennessy, job 21058_ | **149.88 TFLOP/s** · **72 %**<br>_pascal, job 20648_ | **75.85 TFLOP/s** · **36 %**<br>_pascal, job 20648_ |
| **CUTLASS** | `cutlass/attention_fa2.py` → `flash_attention_v2.py` | Oficial sin modificar | `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py` (`FlashAttentionForwardAmpere`) [2] · ¿corre en sm_120/121? ✅ | **91.39 TFLOP/s** · **74 %**<br>_hennessy, job 20649_ | **90.8 TFLOP/s** · **74 %**<br>_hennessy, job 20649_ | **164.64 TFLOP/s** · **79 %**<br>_pascal, job 20648_ | **162.18 TFLOP/s** · **77 %**<br>_pascal, job 20648_ |
| **Referencia** | `torch.nn.functional.scaled_dot_product_attention` | Referencia (no es kernel propio) | FlashAttention-2 incluido en PyTorch (`pytorch_flash::flash_fwd_kernel`); también es la referencia de corrección | **82.19 TFLOP/s** · **67 %**<br>_hennessy, job 21058_ | **84.21 TFLOP/s** · **69 %**<br>_hennessy, job 21058_ | **152.93 TFLOP/s** · **73 %**<br>_pascal, job 20648_ | **152.36 TFLOP/s** · **73 %**<br>_pascal, job 20648_ |

#### Comparación y conclusiones: FlashAttention (forward)

*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al kernel según ncu · eficiencia energética · entorno y job; productividad en GB10 (líneas de código y 1ª llamada en frío). Generado con `benchmarks/conclusiones.py`.*

##### FlashAttention fp16 (CB)

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

##### FlashAttention causal fp16 (CB)

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

#### Perfil de Nsight Compute: FlashAttention (forward)

*Sin perfil todavía: `TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo` (y lo mismo en pascal para la RTX 5090).*

### 3.3 RMSNorm

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 bf16 | RTX 5090 fp16 | RTX 5090 bf16 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `triton/rmsnorm_baseline.py` | Adaptado | Tutorial oficial de Triton `05-layer-norm.py` (`_layer_norm_fwd_fused`) [1] | **242.7 GB/s** · **89 %**<br>_hennessy, job 21058_ | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **1521.4 GB/s** · **85 %**<br>_pascal, job 20499_ | **1513.4 GB/s** · **84 %**<br>_pascal, job 20499_ |
| **Triton + TLX** | `triton_tlx/rmsnorm.py` | Propio | API de µTLX (`triton-lang/triton-ext`, `extensions/utlx`) [4] | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>_pascal, job 20499_ | **1456.4 GB/s** · **81 %**<br>_pascal, job 20499_ |
| **Gluon** | `gluon/rmsnorm.py` | Propio | API de Gluon, tutorial oficial `gluon/02-layouts.py` [1] | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **1521.2 GB/s** · **85 %**<br>_pascal, job 20499_ | **1513.4 GB/s** · **84 %**<br>_pascal, job 20499_ |
| **Helion** | `helion/rmsnorm.py` | Adaptado | Ejemplo oficial de Helion `examples/rms_norm.py` (`rms_norm_fwd`) [3] | **230.3 GB/s** · **84 %**<br>_patterson, job 20435_ | **226.6 GB/s** · **83 %**<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — *no medido*: no se ha ejecutado en esta GPU |
| **CUTLASS** | `cutlass/rmsnorm_ext.cu` | Oficial sin modificar | `cutlass::rmsnorm`, `tools/util/include/cutlass/util/device_rmsnorm.h` [2] | **230.5 GB/s** · **84 %**<br>_patterson, job 20435_ | **211.6 GB/s** · **78 %**<br>_patterson, job 20435_ | **1438.7 GB/s** · **80 %**<br>_pascal, job 20638_ | **1400.1 GB/s** · **78 %**<br>_pascal, job 20638_ |
| **Referencia** | `torch.nn.functional.rms_norm` | Referencia (no es kernel propio) | kernel RMSNorm de PyTorch; también es la referencia de corrección | **217.1 GB/s** · **80 %**<br>_hennessy, job 21058_ | **216.8 GB/s** · **79 %**<br>_hennessy, job 21058_ | **1299.3 GB/s** · **72 %**<br>_pascal, job 20638_ | **1302.8 GB/s** · **73 %**<br>_pascal, job 20638_ |

#### Comparación y conclusiones: RMSNorm

*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al kernel según ncu · eficiencia energética · entorno y job; productividad en GB10 (líneas de código y 1ª llamada en frío). Generado con `benchmarks/conclusiones.py`.*

##### RMSNorm fp16 (MB)

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

##### RMSNorm bf16 (MB)

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

#### Perfil de Nsight Compute: RMSNorm

*Sin perfil todavía: `TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo` (y lo mismo en pascal para la RTX 5090).*

## 4. Vista global

### 4.1 Rendimiento por arquitectura

*Barras: rendimiento de cada DSL; encima, % del pico; línea: pico de referencia. Los huecos llevan «✕ (n)» y su motivo debajo de la figura.*

![Compute-bound — NVIDIA GB10 (sm_121)](resultados/figuras/resumen_compute_bound_nvidia_gb10_sm_121.png)

![Memory-bound — NVIDIA GB10 (sm_121)](resultados/figuras/resumen_memory_bound_nvidia_gb10_sm_121.png)

![Compute-bound — NVIDIA GeForce RTX 5090 (sm_120)](resultados/figuras/resumen_compute_bound_nvidia_geforce_rtx_5090_sm_120.png)

![Memory-bound — NVIDIA GeForce RTX 5090 (sm_120)](resultados/figuras/resumen_memory_bound_nvidia_geforce_rtx_5090_sm_120.png)

### 4.2 Eficiencia energética

*GFLOP/J (CB) y GB/J (MB) en régimen sostenido; «×» = relativo a la referencia de la misma GPU. Método: [resultados/energia.md](resultados/energia.md).*

#### NVIDIA GB10 (sm_121)

| DSL | Matmul fp16 (CB) | Matmul FP8 (CB) | FlashAttention fp16 (CB) | FlashAttention causal fp16 (CB) | RMSNorm fp16 (MB) | RMSNorm bf16 (MB) |
|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | **1019.6 GFLOP/J (×0.97 ref.)**<br>88.4 W (reposo 16.9 W) · 1260.6 GFLOP/J dinámica · 90.13 TFLOP/s sost. (73 % del pico) · 2158 MHz · 1078 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **1585.8 GFLOP/J (×0.78 ref.)**<br>90.1 W (reposo 16.1 W) · 1930.8 GFLOP/J dinámica · 142.88 TFLOP/s sost. (59 % del pico) · 2138 MHz · 693.4 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **913.6 GFLOP/J (×1.08 ref.)**<br>94.7 W (reposo 16.6 W) · 1107.8 GFLOP/J dinámica · 86.52 TFLOP/s sost. (70 % del pico) · 2171 MHz · 1204 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **853.6 GFLOP/J (×1 ref.)**<br>92.2 W (reposo 16.6 W) · 1041 GFLOP/J dinámica · 78.7 TFLOP/s sost. (64 % del pico) · 2203 MHz · 644.1 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **6.23 GB/J (×1.05 ref.)**<br>39.2 W (reposo 14.9 W) · 10.04 GB/J dinámica · 244 GB/s sost. (89 % del pico) · 2502 MHz · 86.24 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.27 GB/J (×1.04 ref.)**<br>38.9 W (reposo 14.9 W) · 10.16 GB/J dinámica · 243.9 GB/s sost. (89 % del pico) · 2483 MHz · 85.61 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **Triton (block-ptr)** | **1036 GFLOP/J (×0.99 ref.)**<br>87.9 W (reposo 17 W) · 1284.4 GFLOP/J dinámica · 91.07 TFLOP/s sost. (74 % del pico) · 2125 MHz · 1061 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | n/a | n/a | n/a | n/a | n/a |
| **Triton + TMA** | **1064.1 GFLOP/J (×1.01 ref.)**<br>88 W (reposo 17 W) · 1318.8 GFLOP/J dinámica · 93.64 TFLOP/s sost. (76 % del pico) · 2047 MHz · 1033 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **1629.2 GFLOP/J (×0.8 ref.)**<br>90.4 W (reposo 16.1 W) · 1982.2 GFLOP/J dinámica · 147.28 TFLOP/s sost. (61 % del pico) · 2158 MHz · 674.9 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | n/a | n/a | n/a | n/a |
| **Gluon** | **860.6 GFLOP/J (×0.82 ref.)**<br>88.3 W (reposo 12.7 W) · 1005.1 GFLOP/J dinámica · 75.99 TFLOP/s sost. (62 % del pico) · 2158 MHz · 1278 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | — *sin kernel*: no hay matmul FP8 en este DSL | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | **6.37 GB/J (×1.07 ref.)**<br>38.3 W (reposo 14.3 W) · 10.16 GB/J dinámica · 243.9 GB/s sost. (89 % del pico) · 2502 MHz · 84.32 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.37 GB/J (×1.06 ref.)**<br>38.3 W (reposo 14.3 W) · 10.16 GB/J dinámica · 243.9 GB/s sost. (89 % del pico) · 2502 MHz · 84.32 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **Helion** | — *no medido*: sin medida de energía (TFM_ENERGIA=1) en esta GPU | — *sin kernel*: no hay matmul FP8 en este DSL | **875.2 GFLOP/J (×1.03 ref.)**<br>94.7 W (reposo 16.1 W) · 1054.5 GFLOP/J dinámica · 82.88 TFLOP/s sost. (67 % del pico) · 2132 MHz · 1256 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **425.2 GFLOP/J (×0.5 ref.)**<br>94 W (reposo 16.1 W) · 513.1 GFLOP/J dinámica · 39.97 TFLOP/s sost. (33 % del pico) · 2164 MHz · 1293 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **6.5 GB/J (×1.09 ref.)**<br>38.3 W (reposo 15.8 W) · 11.07 GB/J dinámica · 249 GB/s sost. (91 % del pico) · 2483 MHz · 82.59 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.47 GB/J (×1.07 ref.)**<br>38.5 W (reposo 15.8 W) · 10.97 GB/J dinámica · 249.1 GB/s sost. (91 % del pico) · 2483 MHz · 82.99 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **CUTLASS** | **573.2 GFLOP/J (×0.55 ref.)**<br>42.7 W (reposo 11.9 W) · 794.6 GFLOP/J dinámica · 24.47 TFLOP/s sost. (20 % del pico) · 2427.5 MHz · 1918 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | **930.3 GFLOP/J (×1.1 ref.)**<br>94.3 W (reposo 14.7 W) · 1102.1 GFLOP/J dinámica · 87.73 TFLOP/s sost. (71 % del pico) · 2125 MHz · 1182 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **899.8 GFLOP/J (×1.06 ref.)**<br>92.1 W (reposo 14.7 W) · 1070.7 GFLOP/J dinámica · 82.87 TFLOP/s sost. (67 % del pico) · 2119 MHz · 611 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **6 GB/J (×1.01 ref.)**<br>41.4 W (reposo 13.7 W) · 8.97 GB/J dinámica · 248.6 GB/s sost. (91 % del pico) · 2502 MHz · 89.41 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **4.68 GB/J (×0.78 ref.)**<br>49 W (reposo 13.7 W) · 6.5 GB/J dinámica · 229.3 GB/s sost. (84 % del pico) · 2502 MHz · 114.7 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **Referencia** | **1051.1 GFLOP/J**<br>87.6 W (reposo 16.9 W) · 1302.4 GFLOP/J dinámica · 92.08 TFLOP/s sost. (75 % del pico) · 2086 MHz · 1046 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **2036.5 GFLOP/J**<br>89.5 W (reposo 16.1 W) · 2483.2 GFLOP/J dinámica · 182.27 TFLOP/s sost. (75 % del pico) · 2112 MHz · 539.9 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **846.6 GFLOP/J**<br>94.6 W (reposo 16.6 W) · 1026.8 GFLOP/J dinámica · 80.09 TFLOP/s sost. (65 % del pico) · 2216 MHz · 1299 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **851 GFLOP/J**<br>92.7 W (reposo 16.6 W) · 1036.6 GFLOP/J dinámica · 78.88 TFLOP/s sost. (64 % del pico) · 2216 MHz · 646 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **5.95 GB/J**<br>36.5 W (reposo 14.9 W) · 10.06 GB/J dinámica · 217.2 GB/s sost. (80 % del pico) · 2437 MHz · 90.23 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.02 GB/J**<br>36.2 W (reposo 14.9 W) · 10.23 GB/J dinámica · 217.8 GB/s sost. (80 % del pico) · 2431 MHz · 89.22 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |

### 4.3 Productividad frente a rendimiento

| algoritmo | DSL | LoC kernel | LoC propias | LoC total | ajuste | configs | 1a llamada en frío | rendimiento | % del pico |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Matmul fp16 | Triton | 67 | 67 | 67 | autotune (triton.autotune) | 64 | 34.1 s | 92.96 TFLOP/s | 75.7 |
| Matmul fp16 | Triton + TMA | 68 | 68 | 68 | autotune (triton.autotune) | 128 | 96.1 s | 93.31 TFLOP/s | 76 |
| Matmul fp16 | Gluon | 85 | 101 | 101 | barrido propio (sin autotuning) | 7 | 3.9 s | 79.26 TFLOP/s | 64.6 |
| Matmul fp16 | Helion | 25 | 25 | 25 | autotune Helion (quick) | 251 | 4.0 min | 61.94 TFLOP/s | 50.5 |
| Matmul fp16 | CUTLASS | 672 | 41 | 713 | fija (tile del ejemplo) | 1 | 1.0 s | 23.59 TFLOP/s | 19.2 |
| FlashAttention fp16 | Triton | 220 | 220 | 220 | autotune (triton.autotune) | 36 | 15.2 s | 88.97 TFLOP/s | 72.5 |
| FlashAttention fp16 | Helion | 45 | 18 | 63 | autotune Helion (quick) | 135 | 2.1 min | 84.76 TFLOP/s | 69.1 |
| FlashAttention fp16 | CUTLASS | 713 | 65 | 778 | barrido propio (sin autotuning) | 6 | 7.7 s | 90.92 TFLOP/s | 74.1 |
| RMSNorm fp16 | Triton | 38 | 38 | 38 | heurística (warps por N) | 1 | 0.6 s | 242.3 GB/s | 88.8 |
| RMSNorm fp16 | Triton + TLX | 52 | 52 | 52 | heurística + --stages a mano | 1 | — | no compila en GB10 (triton-utlx 3.8.0.post1, ver procedencia_kernels.md) | — |
| RMSNorm fp16 | Gluon | 35 | 35 | 35 | heurística (warps y layout por N) | 1 | 0.6 s | 242.1 GB/s | 88.7 |
| RMSNorm fp16 | Helion | 16 | 16 | 16 | autotune Helion (quick) | 134 | 77.3 s | 247.2 GB/s | 90.5 |
| RMSNorm fp16 | CUTLASS | 135 | 68 | 203 | fija (heurística de CUTLASS) | 1 | 55.5 s | 247 GB/s | 90.5 |

*Entorno en frío `hennessy-frio` (jobs 20841, 20842, 20843); pico de mma.sync f16→f32 = 122.72 TFLOP/s (`hennessy`/run_pico_mma). LoC = líneas lógicas sin blancos, comentarios ni docstrings.*

*Análisis: [productividad.md](productividad.md).*

## 5. Fuentes

| # | proyecto | versión | licencia |
|:---|:---|:---|:---|
| [1] | Triton, `triton-lang/triton`, `python/tutorials/` | tag v3.8.0 (`c01b677`) | MIT |
| [2] | CUTLASS, `NVIDIA/cutlass` | tag v4.8.0 (`098de2a`) | BSD-3-Clause |
| [3] | Helion, `pytorch/helion`, `examples/rms_norm.py`, `examples/attention.py` | `main` (`0f3b242`, 2026-09-30) | BSD-3-Clause |
| [4] | µTLX, `triton-lang/triton-ext`, `extensions/utlx` (paquete `triton-utlx` 3.8.0.post1) | `main` (`3c2bb3c`) | MIT |
| [5] | T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691 (ICLR 2024) | — | — |

Las cuatro licencias permiten reutilizar y modificar el código conservando el aviso de copyright.
