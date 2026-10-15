# Métricas de Nsight Compute por kernel

*Generado por `benchmarks/ncu.py` a partir de las ejecuciones con `TFM_NCU=1` (entornos `<nodo>-ncu`).*

*Kernel principal de cada medida (el de mayor duración; +n = otros kernels de la misma llamada) en el mayor tamaño medido. **SM %** y **memoria %**: *Speed of Light* de cómputo y de memoria; **tensor %**: ciclos con el pipe de tensor ocupado; **emisión %**: ciclos en que cada SMSP emite una instrucción; **limita**: recurso que fija la ocupación teórica; **memoria principal**: DRAM en RTX 5090; en GB10 (memoria unificada, sin contadores dram) los bytes que fallan en L2; **espera principal**: motivo con más ciclos de espera por instrucción emitida. ncu con relojes sin fijar (`--clock-control none`).*

*Aún no hay ejecuciones con TFM_NCU=1: `TFM_NCU=1 bash ejecutar.sh encolar todo`.*
