# Bitácora del TFM

*Notas de trabajo para recapitular al redactar la memoria. Cada entrada lleva la fecha y los
jobs de Slurm que la respaldan (logs en `logs/`).*

## 2026-10-05 — Principales desafíos de las pruebas (sobre todo FlashAttention)

**1. La arquitectura: Blackwell GeForce no es el Blackwell de los ejemplos.**
RTX 5090 (sm_120) y GB10 (sm_121) no tienen `wgmma` (Hopper) ni `tcgen05` ni memoria de
tensor (Blackwell de centro de datos, sm_100). Casi todos los kernels oficiales optimizados
están escritos para esas dos arquitecturas, así que aquí no se pueden ejecutar:
- El ejemplo de atención de Gluon aborta en la GPU con
  `LLVM ERROR: Cannot select: intrinsic tcgen05.wait.st` (jobs 20639 pascal, 20640 hennessy).
- Los tutoriales de atención de TLX, tanto los de µTLX (`triton-ext`) como los del fork de Meta
  (`facebookexperimental/triton`), usan `wgmma`/`tcgen05` (el del fork de Meta, comprobado
  solo en su código fuente, no ejecutado).
- En CuTe DSL solo sirve el FlashAttention-2 de **Ampere** (`mma.sync`); los FMHA de Hopper y
  Blackwell no.
- Por eso en estas GPUs el techo lo pone `mma.sync`, lo mismo que ya se vio con TMA en el
  matmul (§ tma.md).

**2. Versiones que no encajan entre sí.**
- El tutorial actual de triton-ext (`7d2619e`) no compila con el `triton-utlx` 3.8.0.post1
  instalado (`async_tasks.__init__() got an unexpected keyword argument '_semantic'`, job
  20644). Y en GB10, TLX ya generaba IR inválido.
- Los tutoriales no vienen en los paquetes pip. Para usar código oficial sin tocarlo hubo que
  fijar commits exactos (Triton `c01b677`, CUTLASS `098de2a`, Helion `0f3b242`) y comprobar
  que la copia es idéntica (`cmp`) o que el fichero no cambia entre versiones (`git diff`).
- Imágenes desactualizadas: la imagen `cutlass` de pascal no tenía CuTe DSL y falló hasta
  reconstruirla (job 20637 → 20638).

**3. Que la comparación entre DSLs sea justa.**
- Triton y Helion hacen autotuning, pero el ejemplo de CuTe no. Con su configuración por
  defecto (128/128/128 hilos), a D=128 se quedaba en el ~60 % de SDPA (job 20645). Se añadió
  un barrido de sus parámetros públicos, sin modificar su código: con él pasa al 85–110 % de
  SDPA (job 20646, GB10, N ≤ 2048).
- Cada implementación usa un orden de dimensiones distinto: CuTe espera (B, N, H, D) y el
  resto (B, H, N, D). La transposición se hace fuera de la medida.
- El kernel causal oficial de Helion no se salta el triángulo superior, solo lo enmascara:
  hace el doble de trabajo útil. Hay que tenerlo en cuenta al leer sus TFLOP/s causales.
- `warp_specialize` en Triton solo está pensado para capability 9 y 10, así que aquí va
  desactivado, igual que en el propio benchmark del tutorial.

**4. Validar con tamaños reales.**
La referencia "ingenua" del tutorial necesitaría 137 GB con B=4, H=32, N=16384, así que la
referencia es SDPA de PyTorch (FlashAttention-2, `pytorch_flash::flash_fwd_kernel`). Se usa
la tolerancia del propio tutorial (atol 1e-2, rtol 0).

**5. Demostrar que de verdad se usan tensor cores.**
Cada DSL esconde el PTX de una forma distinta:
- El autotuner de Triton no devuelve el kernel compilado; se saca de su caché interna
  (`JITFunction.device_caches`).
- Helion genera funciones Triton por dentro y hay que localizarlas. Además, al autotunear
  activa `TRITON_STORE_BINARY_ONLY=1` (`helion/autotuner/base_search.py`) y Triton no guarda
  el PTX: hay que fijar la variable a 0 o mirar el SASS (`HMMA`).
- CuTe solo guarda el PTX si se activa `CUTE_DSL_KEEP=ptx`.

Confirmado `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` en las tres: Triton y CuTe
(jobs 20645 y 20646, GB10) y Helion (job 20647, GB10, con `TRITON_STORE_BINARY_ONLY=0`).

**6. Infraestructura.**
- La GPU solo es accesible por Slurm, y cada nodo tiene su copia del repo (pascal tiene la
  suya), así que hay que sincronizar resultados.
- Los drivers son distintos (hennessy pasó de 580 a 610), lo que obliga a separar entornos
  con `TFM_MAQUINA`.
- El autotuning de Helion es lento: más de 1 minuto por forma.

**7. Variabilidad entre ejecuciones.**
En la RTX 5090, cuBLAS dio 168 TFLOP/s en un job (20638) y 209 en otro (20499), y la
referencia de RMSNorm pasó de 1389 a 1299 GB/s. Las comparaciones fiables son las que se
hacen dentro del mismo job; además, convendría fijar los relojes de la GPU.

## 2026-10-05 — FlashAttention medido en las dos GPUs

Jobs 20648 (pascal, RTX 5090) y 20649 + 20665 (hennessy, GB10). B=4, H=32, fp16, N ∈ {1024,
4096, 16384}, D ∈ {64, 128}. Todas las formas validan contra SDPA (atol 1e-2).
- **Helion en GB10 falló en el 1er intento** (job 20649) con N=16384, D=128: la configuración
  inicial del autotuning tardaba > 30 s en medirse (`benchmark timeout after 30.0s`, límite por
  defecto de `HELION_AUTOTUNE_BENCHMARK_TIMEOUT`) y acabó en `NoConfigFound`. Con 300 s (job
  20665) completa, pero el autotuning de esa forma tarda 24–31 min.
- El banco guardaba solo al final: el fallo hizo perder las 8 formas ya medidas. Desde
  entonces una forma que falla se anota en el CSV (columna `error`) y se siguen las demás.
- **Causal de Helion ≈ mitad que su no causal** (RTX 5090: 75.9 vs 149.9 TFLOP/s; GB10: 42.1 vs
  66.5), coherente con que su kernel causal no se salta el triángulo superior.

## 2026-10-06 — Matmul FP8 en CUTLASS: por qué no está en el resumen

Se intentó rellenar la celda *CUTLASS / Matmul FP8* reutilizando el ejemplo oficial
`dense_gemm.py` (Sm120GemmKernel) con entradas FP8 e4m3. No es posible con ese kernel:

1. **dlpack no exporta float8** (job 20778 pascal): `from_dlpack` falla con `BufferError: float8
   types are not supported by dlpack`. Se resuelve usando `cutlass.torch.cute_tensor_like(t,
   cutlass.Float8E4M3FN, ...)` en vez de `from_dlpack`.
2. **El MMA del ejemplo es fp16/bf16-only** (job 20780 pascal): aunque el chequeo de tipos
   admite anchura 8 ("a_dtype should be float16 or float8"), el Op real es
   `cute.nvgpu.warp.MmaF16BF16Op(...)` (dense_gemm.py:261) y aborta con *"expects the 'ab_dtype'
   Op parameter to be one of Float16 or BFloat16"*.
3. **La única vía CuTe para FP8 en Blackwell GeForce es block-scaled (MXFP8)** (diagnóstico
   job 20781): `examples/.../blackwell_geforce/kernel/blockscaled_gemm/`
   (`dense_blockscaled_gemm_persistent_cooperative/pingpong.py`). Su dispatcher mapea
   `(FP8, FP8, *, Float8E8M0FNU, 32) -> MmaMXF8Op`: FP8 **microscaling**, con un factor de
   escala E8M0 por cada 32 elementos (`sf_vec_size=32`). Es un formato distinto del FP8 plano
   (single-scale) de Triton y cuBLASLt, con metadatos de escala adicionales.

**Conclusión:** no hay un matmul FP8 de CUTLASS *comparable* al de la columna "Matmul FP8"
(FP8 plano) con los ejemplos oficiales; `dense_gemm` es fp16/bf16. La referencia FP8 sigue
siendo cuBLASLt (`torch._scaled_mm`). Un CUTLASS FP8 real exigiría portar el kernel
block-scaled MXFP8 (otro formato), que quedaría en una columna propia, no en "Matmul FP8".

## 2026-10-08 — Pico real de `mma.sync` (microbenchmark)

Jobs 20833 (fallido; errores de ptxas resumidos en
[resultados/sondas/run_pico_mma-20833.txt](resultados/sondas/run_pico_mma-20833.txt)) y 20834
(hennessy, GB10). Detalle en [pico_mma.md](pico_mma.md).
- Kernel CUDA propio (`src/Microbench/pico_mma_ext.cu`): bucles de `mma.sync` sobre registros,
  sin memoria, barriendo warps/SM × ILP. Se compila con nvcc a una `.so` (ctypes), no con
  `torch.utils.cpp_extension`, para controlar la arquitectura.
- **Trampa de nvcc** (job 20833): con `-arch=sm_121a -shared` el PTX sale para `compute_121`
  (sin la `a`) y ptxas rechaza las MMA block-scaled (*"Feature '.kind::mxf4nvf4' not supported
  on .target 'sm_121'"*), aunque `-cubin -arch=sm_121a` sí compila. Se resuelve con
  `-gencode=arch=compute_121a,code=sm_121a` explícito.
- Picos: **f16/bf16 122.7 TFLOP/s** (1 024 FLOP/ciclo/SM), tf32 61.4, **fp8/int8 243**,
  **MXFP4/NVFP4 487** (`OMMA.SF.16864`); acumular en f16 no acelera; el escalado por bloques es gratis.
- Latencia ~28 ciclos; un warp no emite más de 1 MMA cada ~17 ciclos → hacen falta ≥ 8 warps/SM
  (con 4: ~94 % como máximo).
- Los mejores kernels del TFM (Triton+TMA, cuBLAS, cuBLASLt FP8, CuTe FA) se quedan en el
  **74–78 % de este pico**: el límite común no es la MMA.
- *(Corregido en la entrada siguiente: las cifras de reloj/potencia de este punto y del siguiente
  vienen de medidas cortas con potencia promedio y no son fiables.)*
- `comun.medir()` registra desde hoy la mediana del reloj SM y de la potencia durante cada medida
  (`reloj_mhz`, `potencia_w`; NVML, o nvidia-smi si no hay pynvml). Matmul repetidos con reloj
  (jobs 20835–20837): bajo GEMM el reloj cae a 2 275–2 340 MHz con 80–91 W. Corrigiendo el pico
  por el reloj, cuBLAS queda en el 85 %, cuBLASLt FP8 en el 82 % y Triton en el 78–81 %: la
  potencia explica solo una parte del hueco; el resto se pierde por ciclo.
- **TMA ahorra potencia** (55 W frente a 81 W del baseline, reloj 2 470 frente a 2 275 MHz): ese
  es su +4.7 % en 8192³, no una mejora por ciclo. Una sola ejecución; hay que repetirlo.
  **→ Falso: artefacto de medida (ver entrada siguiente).**
- Validación de la imagen triton-tlx: el paso 5 (TLX `async_load`) falla con *"error encountered
  during parsing"* en todos estos jobs; ya pasaba antes y no afecta a los benchmarks.

## 2026-10-08 — Corrección: reloj y potencia se miden en régimen sostenido

Sondas en hennessy (jobs 20844–20847) y `run_sostenido` (job 20848). Detalle en [pico_mma.md](pico_mma.md); salidas de las sondas archivadas en [resultados/sondas/](resultados/sondas/) (índice en `.sonda/README.md`).
- **La potencia de `nvmlDeviceGetPowerUsage` es un promedio retrasado ~1 s.** Al empezar un
  GEMM, la instantánea (`NVML_FI_DEV_POWER_INSTANT`) marca 100 W y la promedio aún 20 W; cuBLAS a
  95 TFLOP/s llegó a leer 10.8 W (sonda 20844). `comun.Muestreador` usa ahora la instantánea
  (`potencia_fuente` en `meta.json`).
- **NVML actualiza cada ~0.5 s y el reloj tarda ~0.5 s en bajar bajo carga** (2 410 → 2 180 MHz
  con cuBLAS continuo, sonda 20845). Una medida de `do_bench` ve sobre todo el transitorio.
- **"TMA ahorra potencia" era falso.** Con la potencia instantánea (job 20847), TMA consume 88 W y
  block-ptr 90 W.
- **Régimen sostenido** (`run_sostenido`: 6 s por kernel, los 4 últimos cuentan, job 20848):
  - La MMA pura sostiene 2 424 MHz con 100 W (118.7 TFLOP/s f16, 99.6 % de 1 024 FLOP/ciclo/SM).
  - Los GEMM bajan a 2 080–2 180 MHz con ~90 W: el límite no es la potencia de GPU según NVML.
  - % del pico a su propio reloj: cuBLAS 89.8, Triton + TMA 91.7, Triton 84.6, cuBLASLt FP8 86.7
    y Triton FP8 67.9. Sobre el pico nominal: 75–77 %. Aproximadamente la mitad del hueco es
    reloj y la otra mitad, pérdida por ciclo. TMA sí mejora por ciclo, pero sostiene menos reloj.
- `run_matmul_tma --sizes 8192` repetido (job 20847) para dejar `ultimo` con la nueva medida de
  potencia.

## 2026-10-08 — Productividad frente a rendimiento por DSL

Jobs 20841–20843 (hennessy, entorno `hennessy-frio`) y sondas 20844 y 20846 (salidas en
[resultados/sondas/](resultados/sondas/)). Detalle en [productividad.md](productividad.md).
- **Los tiempos de 1ª llamada guardados hasta ahora no eran comparables**: `TRITON_CACHE_DIR`
  era persistente, y el FlashAttention de Triton completo tardó 17 s (job 20682). Nuevo modo
  `TFM_CACHE_FRIA=1` en `ejecutar.sh`: la caché de Triton y la de extensiones (`TFM_CACHE`) van al
  `/tmp` del contenedor; `meta.json` lo registra (`cache_fria`). Se usa con
  `TFM_MAQUINA=<nodo>-frio`, entorno que `resumen.py` ignora (si no, sus pocas formas
  sustituirían celdas del resumen).
- `benchmarks/productividad.py` (sin GPU): LoC lógicas por (algoritmo, DSL), con el código
  oficial externo descargado en su versión fijada (`.cache/fuentes/`), el espacio de
  configuraciones y los tiempos en frío.
- Resultados (forma única por algoritmo, fp16): Helion 16–45 LoC pero 1.3–4 min de autotuning
  (134–251 configs) y solo el 50 % del pico en matmul; Triton 38–220 LoC, 15–96 s y 72–89 %;
  CuTe 670–780 LoC (oficiales); Gluon 101 LoC y 65 % en matmul.
- **El matmul oficial de CuTe (`dense_gemm`, `Sm120GemmKernel`) rinde el 19 % del pico en GB10**
  (23.6 TFLOP/s, el 24 % de cuBLAS), frente al 99 % de cuBLAS en la RTX 5090 (job 20638). No es
  `max_active_clusters`: forzando 48, 96 o 192 sigue igual (sonda 20844; además
  `HardwareInfo.get_max_active_clusters` falla con `CUDA_ERROR_INVALID_CONTEXT` si se llama antes
  de crear el contexto). Barriendo el tile (sonda 20846), el mejor es 128×256×64 con 37.8 TFLOP/s;
  el ejemplo calcula `ab_stage` de 1 a 5 según el tile. Causa abierta.

## 2026-10-08 — Documentación y reproducibilidad

- **Evidencia de las sondas**: `logs/` no se versiona, así que las salidas que respaldan
  conclusiones (jobs 20833, 20844–20846) se archivan en [resultados/sondas/](resultados/sondas/),
  con una cabecera que explica qué se probó. Índice de todas las sondas en `.sonda/README.md`.
- **Etiquetas LaTeX únicas**: las tablas generadas usaban `tab:<experimento>`, repetida entre
  entornos (hennessy, pascal…), así que incluir la misma tabla de dos GPUs rompía las referencias.
  Ahora son `tab:<maquina>-<experimento>` (`informe.etiqueta_tex`); las ejecuciones antiguas sin
  `contexto.maquina` la toman de su ruta. Regeneradas todas las tablas; en las figuras con datos
  sin cambios se ha conservado la versión del último commit.
- `run_sostenido` tiene informe propio: el genérico suponía tamaños y `do_bench`, y su pie era
  falso para esta medida.
- Resumen por arquitectura y sus gráficas regenerados desde el login (la imagen triton-tlx no
  tiene seaborn, así que las gráficas no se actualizaban desde los jobs). Tablas del README
  actualizadas desde `results/resumen.csv`.
- **Pendiente:** el commit `25ac1d0` borró `run_matmul_sparsity.py` con sus resultados y figuras,
  pero `sparsity.md`/`.tex` siguen enlazando `resultados/hennessy-580/figuras/run_matmul_sparsity.*`.

## 2026-10-09 — Picos de la ficha de NVIDIA en las gráficas

Documentos de referencia (cifras en `benchmarks/fichas.py`):
- **RTX 5090:** *NVIDIA RTX Blackwell GPU Architecture* (whitepaper), tabla 3, págs. 46–47,
  [PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
  Tensor cores en denso (sin sparsity) y con acumulación en FP32, a la frecuencia boost de 2 407 MHz:
  FP16/BF16 209.5 TFLOP/s, FP8 419 TFLOP/s y TF32 104.8 TFLOP/s. Con acumulación en FP16, el doble
  (FP16 419, FP8 838). Memoria: 1 792 GB/s (GDDR7, 512 bits a 28 Gbps). TGP: 575 W.
- **GB10 (DGX Spark):** *DGX Spark User Guide*, "Hardware Overview",
  [docs.nvidia.com/dgx/dgx-spark/hardware.html](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
  Solo publica 1 PFLOP en FP4 **con sparsity**, 273 GB/s (LPDDR5X a 8 533 MT/s, 256 bits) y un
  TDP de 140 W para todo el SoC. **No hay pico oficial de FP16 ni de FP8.** Los agregadores web dan
  cifras contradictorias entre sí, así que no se usan.

Cambios:
- Todas las gráficas de rendimiento dibujan el pico de la ficha como línea roja discontinua: el
  resumen por arquitectura (un tramo por algoritmo, porque la precisión cambia entre columnas), el
  informe de cada benchmark y el de Triton frente a cuBLAS. En GB10 solo hay línea en memoria; en
  cómputo se indica "sin pico oficial". Las gráficas de `run_sostenido`, `run_pico_mma` y
  productividad no la llevan: su eje es un % del pico medido, no TFLOP/s.
- Se han regenerado las figuras de todas las ejecuciones (los datos y las tablas no cambian).

Observación abierta: **en la RTX 5090 hay medidas por encima de la ficha con acumulación en
FP32.** Triton fp16 llega a 220–224 TFLOP/s (ficha: 209.5) y cuBLASLt FP8 a 599–624 TFLOP/s
(ficha: 419; job 20499). Lo de fp16 lo puede explicar un reloj real por encima del boost de
2 407 MHz, pero lo de FP8 no (×1.43). Una de dos: o `torch._scaled_mm` acumula a la velocidad de
FP16, o en sm_120 la acumulación en FP32 no va a mitad de velocidad. Para saberlo, falta medir el
pico de `mma.sync` e4m3 con acumulación en FP32 y en FP16 con `run_pico_mma` en pascal.
- Descartado que sea solo el reloj: para 599 TFLOP/s a la velocidad de la ficha haría falta
  ~3 440 MHz (×1.43 sobre el boost).
- A favor de que la acumulación FP32 vaya a velocidad completa: el error de cuBLASLt es idéntico
  al de Triton, que acumula en FP32 (3.753 % en 4096³ y 8192³). No es concluyente, porque una
  acumulación FP16 por bloques de K con paso a FP32 entre bloques casi no cambia el error.
- Pendiente en pascal: `run_pico_mma` (job 20838; `e4m3·f32acc` frente a `e4m3·f16acc`) y la
  sonda `sonda_fp8_cublas.py` (job 20852): kernel de cuBLASLt, reloj, potencia, límite de
  potencia y `use_fast_accum`.

## 2026-10-09 — RTX 5090: la ejecución de CUTLASS fue un 20 % más lenta, también en cuBLAS

- En el resumen de la 5090, CUTLASS matmul fp16 sale a 165.8 TFLOP/s frente a 209–224 de cuBLAS
  y Triton. Pero en esa misma ejecución (`run_matmul_cutlass`, job 20638, 4 oct) **cuBLAS dio
  167.9** con el mismo kernel (`cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3`) que en el
  job 20499 (2 oct) dio 209.1. CUTLASS rinde el 98.8 % de cuBLAS en su ejecución: lo lento fue la
  GPU, no CUTLASS.
- Descartado: otros trabajos en pascal (Slurm no muestra ninguno) y GPU ocupada (en reposo al
  empezar: 0 %, 2 MiB, 43 °C).
- Sin poder comprobarlo: esas ejecuciones son anteriores al registro de reloj y potencia. Cambió
  el driver (610.43.02 el 4 oct; el del 2 oct no consta). Candidatos: el driver o un límite de
  potencia por debajo de los 575 W.
- Repetición en pascal con reloj y potencia: `run_matmul_cutlass --sizes 8192` (job 20853) y
  `run_matmul` (job 20839, control con cuBLAS). Hasta entonces, **la celda de CUTLASS de la 5090
  no es comparable** con las demás.

## 2026-10-09 — Gráficas: etiquetas del eje x

- En las gráficas por benchmark con muchas formas (las 12 de atención) las etiquetas del eje x
  se pisaban: el ancho de la figura solo dependía del número de series. `informe.grafica` lo
  calcula ahora también con el número de formas. Regeneradas todas las figuras; los datos no
  cambian.

## 2026-10-09 — Eficiencia energética por DSL (`TFM_ENERGIA=1`)

- Con `TFM_ENERGIA=1`, `comun.medir()` ejecuta además cada kernel en régimen sostenido: 3 s de
  reposo y luego 6 s en bucle, de los que se descartan los 2 primeros. Anota la potencia
  (instantánea de NVML, solo GPU), la potencia en reposo, el reloj, los mJ por llamada y la
  eficiencia (GFLOP/J en CB, GB/J en MB), también en versión "dinámica" (sin la potencia de
  reposo). Entorno por defecto `<nodo>-energia`, que el resumen de rendimiento ignora.
  `benchmarks/energia.py` lo reúne en `docs/TFM/resultados/energia.{md,tex}`.
- El bucle de régimen sostenido pasa a `comun.sostenido()` y `run_sostenido` lo reutiliza.
- Jobs 20849–20851 (hennessy). Matmul 8192³ en GB10: cuBLASLt FP8 da 2 037 GFLOP/J frente a 1 051
  de cuBLAS fp16, con la misma potencia (~90 W): FP8 casi duplica el trabajo por julio. Triton +
  TMA llega a 1 064 GFLOP/J (×1.01 cuBLAS), Gluon a 861 (×0.82) y CuTe a 573 (×0.55): con 24 TFLOP/s
  solo consume 43 W y es el menos eficiente.
- `run_matmul_helion` falló en el autotuning (`helion.exc.NoTensorArgs` al precompilar una
  configuración con `TensorDescriptor`), antes de medir. El día 8 funcionó (job 20841).
- La potencia de reposo varía entre 12 y 17 W según el script, porque se mide después del
  autotuning. La eficiencia total es fiable; la dinámica depende de esa cifra.
- Attention (job 20850; B=4 H=32 N=4096 D=128) y RMSNorm (job 20851; 16384×8192) en GB10:
  - Attention no causal: CuTe 930 GFLOP/J (×1.10 SDPA), Triton 914 (×1.08), Helion 875 (×1.03),
    SDPA 847, todos a ~94 W. Causal: CuTe 900, Triton 854, SDPA 851 y Helion 425 (×0.50; rinde la
    mitad con la misma potencia).
  - RMSNorm consume 38–49 W (las SM esperan a la memoria). fp16: Helion 6.50 GB/J, Gluon 6.37,
    Triton 6.23, CuTe 6.00 y `F.rms_norm` 5.95. En bf16, CuTe baja a 4.68 GB/J (×0.78): su kernel
    escalar `_e1` consume 49 W para menos ancho de banda.
  - En régimen sostenido, RMSNorm llega a 244–249 GB/s (89–91 % de los 273 GB/s de la ficha).
  - TLX falla en GB10, como siempre.
- `docs/` sale del repositorio: ahora vive en `~/hennessy/tfm/docs` (`TFM_DOCS`; `ejecutar.sh`
  lo monta en el contenedor en `/workspace/docs`). El historial de git se reinicia en un único commit.

## Enlaces de origen de cada kernel

Fuente exacta (repo @ versión + ruta) de cada kernel; detalle de elaboración en
[procedencia_kernels.md](procedencia_kernels.md). Versiones: Triton tag v3.8.0 (`c01b677`),
CUTLASS tag v4.8.0 (`098de2a`), Helion `0f3b242`, µTLX `3c2bb3c`.

**Compute-bound — matmul:**
- `triton/matmul.py` (baseline): https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/03-matrix-multiplication.py
- `triton/matmul_tma.py`: https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/09-persistent-matmul.py
- `triton/matmul_blockptr.py` y `triton/matmul_fp8.py`: mismo tutorial 03 (API `tl.make_block_ptr`; el 03 ya muestra FP8)
- `triton/matmul_fp8_tma.py`: propio (combina `matmul_fp8` + `matmul_tma`)
- `gluon/matmul.py`: propio, con piezas oficiales de Triton — `mma_v2` en https://github.com/triton-lang/triton/blob/v3.8.0/python/test/unit/language/test_core.py y `cp.async` de https://github.com/triton-lang/triton/blob/v3.8.0/python/examples/gluon/03-async-copy.py
- `helion/matmul.py`: https://github.com/pytorch/helion/blob/0f3b242/examples/matmul.py
- `cutlass/matmul_cute.py`: https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
- *(FP8 CUTLASS, no viable aquí — MXFP8 block-scaled):* https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py

**Compute-bound — FlashAttention (forward):**
- `triton/fused_attention.py` (copia literal): https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/06-fused-attention.py
- `helion/attention.py`: https://github.com/pytorch/helion/blob/0f3b242/examples/attention.py
- `cutlass/attention_fa2.py`: https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py
- *(Gluon attention, no corre — usa `tcgen05`):* https://github.com/triton-lang/triton/blob/v3.8.0/python/examples/gluon/01-attention-forward.py

**Memory-bound — RMSNorm:**
- `triton/rmsnorm_baseline.py`: https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/05-layer-norm.py
- `gluon/rmsnorm.py`: API de layouts de https://github.com/triton-lang/triton/blob/v3.8.0/python/examples/gluon/02-layouts.py
- `helion/rmsnorm.py`: https://github.com/pytorch/helion/blob/0f3b242/examples/rms_norm.py
- `triton_tlx/rmsnorm.py`: propio sobre µTLX https://github.com/triton-lang/triton-ext/tree/3c2bb3c/extensions/utlx
- `cutlass/rmsnorm_ext.cu`: https://github.com/NVIDIA/cutlass/blob/v4.8.0/tools/util/include/cutlass/util/device_rmsnorm.h

*(Rutas verificadas en procedencia_kernels.md; algunos tutoriales de Gluon pueden estar bajo `python/examples/gluon/` o `python/tutorials/gluon/` según la versión.)*

