# Resumen: optimización de matmul en tensor cores (GB10)

*Resumen consolidado de todos los experimentos del TFM. Hardware: **NVIDIA GB10 (sm_121)**, memoria LPDDR5X unificada; torch 2.9, triton 3.8.0, CUDA 13.0. Metodología: `triton.testing.do_bench` (mediana); tensor cores verificados en el PTX; desde el 2026-10-08, reloj SM y potencia registrados en cada medida. Detalle por técnica en `docs/TFM/{tma,fp8,sparsity,pico_mma}.md`.*

## Objetivo

Partir del matmul denso en Triton (baseline, ~90–100 TFLOP/s en FP16) e intentar **superar la barrera de los ~100 TFLOP/s** acercándose a los límites del silicio, aplicando las técnicas específicas de la arquitectura Blackwell/Hopper: TMA, pipelining, swizzling L2, FP8 y sparsity 2:4.

## Tabla resumen (matmul 8192³)

| técnica | TFLOP/s | vs FP16 | ¿sube el techo? | evidencia clave |
|:---|---:|---:|:---:|:---|
| **FP16 baseline** (Triton) | ~92 | — | referencia | `mma.sync.m16n8k16`, `carga=cp.async` |
| TMA en FP16 (descriptores) | ~93 | +0 % | ❌ | activa `cp.async.bulk.tensor` pero no acelera |
| **FP8 denso** (Triton) | **150** | **+62 %** | ✅ | `mma.sync.m16n8k32.e4m3`, error 3.75 % |
| FP8 denso (cuBLASLt) | **192** | **+108 %** | ✅ | `torch._scaled_mm` |
| FP8 + TMA (Triton) | 150 | +62 % | ❌ (vs FP8) | TMA real, solo +1.5 % sobre FP8 |
| Sparsity 2:4 (cuSPARSELt) | 32 | **−65 %** | ❌ | kernel `sm80_xmma_sparse` (Ampère) |
| *Pico de `mma.sync` FP16 / FP8 / MXFP4* | *122.7 / 243 / 487* | — | techo físico | microbenchmark sin memoria (`run_pico_mma`) |

*Los mejores kernels sostienen el 75–77 % del pico nominal de su precisión, y el 87–92 % del pico al reloj que el chip mantiene bajo carga (§ 5).*

## Qué se descubrió, técnica a técnica

### 1. El baseline no está limitado por la memoria, sino por la ruta hacia la MMA
El matmul denso FP16 se estanca en ~90–100 TFLOP/s. El ancho de banda efectivo es bajísimo (~34 GB/s en 8192³) porque los datos se reutilizan en SRAM: **no es memory-bound**. Pero tampoco agota la unidad MMA: el microbenchmark de § 5 mide 122.7 TFLOP/s para `mma.sync` FP16, y el matmul se queda en el ~75 %. El cuello está en alimentar la MMA (y, en parte, en el reloj; ver § 5).

### 2. TMA: se activa pero no acelera → *ver [tma.md](tma.md)*
- `tl.make_block_ptr` **no** activa TMA (baja a `cp.async`) y está **deprecado** en Triton 3.8.
- `tl.make_tensor_descriptor` **sí** activa TMA (`cp.async.bulk.tensor`, verificado en PTX).
- **Pero apenas acelera** (+2–5 %): coherente con que el matmul denso no sea memory-bound. En régimen sostenido sí mejora la eficiencia por ciclo (del 85 % al 92 % del pico de `mma.sync`, al nivel de cuBLAS), pero su kernel sostiene un reloj algo menor y el neto se queda en +3.5 % (§ 5). *Lección: activar el acelerador de hardware correctamente no implica ganar rendimiento si el cuello está en otro sitio.*
- En sm_121 Triton emite `mma.sync` (no `wgmma` ni `tcgen05.mma`): la MMA es la de sm_80.

### 3. FP8: la única palanca que rompe la barrera → *ver [fp8.md](fp8.md)*
- FP8 e4m3 alcanza **150 TFLOP/s** (Triton) y **192** (cuBLASLt), superando con holgura los ~100 de FP16.
- El throughput viene de la MMA: pasa a `m16n8k32` (K doble → doble ritmo).
- **Coste:** error relativo **3.75 %** (vs 0.02 % en FP16), acotado y estable.
- El kernel Triton llega a ~83 % de cuBLASLt; el hueco restante es de **scheduling** (kernels persistentes), no de datos: **TMA sobre FP8 solo aporta +1.5 %**.

### 4. Sparsity 2:4: regresión en este hardware/stack → *ver [sparsity.md](sparsity.md)*
- El 2:4 (cuSPARSELt vía PyTorch) rinde el **34–62 % del denso**: va **más lento**.
- Causa: cuSPARSELt despacha a `sm80_xmma_sparse_gemm` (**kernel de Ampère**), mientras el denso usa un kernel nativo `nvjet_sm121_..._tmaAB` (Blackwell + TMA). No hay kernel sparse nativo para sm_121.
- **FP8 + 2:4 no está soportado** por la API. La cifra de catálogo (~838 TFLOPS, sparse + baja precisión) **no es reproducible** con este stack.

### 5. El techo real: pico de `mma.sync` → *ver [pico_mma.md](pico_mma.md)*
- Microbenchmark sin memoria (cadenas de `mma.sync` en registros): **1 024 / 2 048 / 4 096 FLOP/ciclo/SM** en FP16/BF16, FP8/INT8 y MXFP4/NVFP4, es decir, **122.7 / 243 / 487 TFLOP/s** en ráfaga a ~2.5 GHz. La MMA pura sostiene 2.42 GHz (118.7 TFLOP/s en FP16). Acumular en FP16 no acelera, y el escalado por bloques no cuesta nada.
- Saturarlo exige **≥ 8 warps/SM**: un warp no emite más de una MMA cada ~17 ciclos.
- En régimen sostenido (`run_sostenido`), los mejores kernels (cuBLAS, Triton + TMA, cuBLASLt FP8) dan el **75–77 % del pico nominal**. Aproximadamente la mitad del hueco es **reloj**: un GEMM real baja GB10 a 2.08–2.18 GHz, aunque la potencia de GPU que da NVML sea menor que con la MMA pura. La otra mitad se pierde **por ciclo**: a su propio reloj, rinden el 87–92 %. Triton sin TMA llega al 85 %, y Triton FP8 al 68 %.
- **FP4 block-scaled** (`kind::mxf4nvf4`, solo con `sm_121a`) duplica el techo de FP8 y es la palanca que no se ha explotado.

### 6. Eficiencia energética → *ver [resultados/energia.md](resultados/energia.md)*
- En régimen sostenido a 8192³ (job 20849), **FP8 casi duplica el trabajo por julio**: cuBLASLt FP8 hace 2 037 GFLOP/J y cuBLAS fp16 1 051, con la misma potencia de GPU (~90 W).
- En fp16, Triton + TMA (1 064 GFLOP/J) iguala a cuBLAS. El matmul de CuTe consume la mitad (43 W), pero es el menos eficiente (573 GFLOP/J), porque deja la GPU infrautilizada.
- NVIDIA no publica picos de FP16 ni de FP8 para GB10, solo 1 PFLOP FP4 con sparsity, 273 GB/s y 140 W de TDP del SoC (§ [plataforma.md](plataforma.md)). Por eso la referencia de cómputo es el pico medido de `mma.sync`.

## Conclusión

En GB10 con el *stack* actual (torch 2.9 / triton 3.8 / cuSPARSELt), la única técnica que **efectivamente sube el techo** de los tensor cores es la **reducción de precisión a FP8** (+62 % en Triton, hasta +108 % con cuBLASLt), a cambio de un error acotado del ~3.75 %. Las dos técnicas "de catálogo" —TMA y sparsity 2:4— **no rinden aquí**: TMA porque el matmul no es memory-bound, y 2:4 porque falta un kernel sparse nativo de Blackwell. El valor del trabajo está tanto en el resultado positivo (FP8) como en documentar, con evidencia de bajo nivel (PTX, nombres de kernel), *por qué* las abstracciones y técnicas anunciadas no siempre se traducen en aceleración real.

Con el pico real como referencia (§ 5), el diagnóstico se afina: ningún kernel, tampoco cuBLAS, pasa del ~77 % del pico nominal de `mma.sync`. El hueco no está en la instrucción, sino, a partes parecidas, en el reloj que el chip sostiene bajo un GEMM real y en alimentar la MMA.

**Líneas abiertas:** (i) kernel FP8 **persistente** para arañar el ~17 % que separa de cuBLASLt; (ii) revisar 2:4 cuando el stack incorpore kernels sparse para sm_121; (iii) FP8+2:4 vía cuSPARSELt directo; (iv) matmul **MXFP4/NVFP4** (`tl.dot_scaled`) frente al pico de 487 TFLOP/s; (v) contadores de Nsight Compute para atribuir la pérdida por ciclo; (vi) atribuir la bajada de reloj bajo GEMM (*throttle reasons* de NVML, potencia del módulo completo).

## Reproducibilidad

Cada experimento genera automáticamente tabla + gráfica (`docs/TFM/resultados/<entorno>/`; los resultados de esta memoria son del entorno `hennessy-580`) y guarda contexto (GPU, versiones, job de Slurm) en `meta.json`.

```bash
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_tma
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_fp8
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_sparsity
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_pico_mma
```
