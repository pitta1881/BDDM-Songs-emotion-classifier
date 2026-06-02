# Decisión: Umbral de Varianza PCA

**Umbral elegido**: 95%  
**Dimensiones resultantes**: 232 (de 384 originales)  
**Reducción**: 152 dims eliminadas (40%)  

## Metodología

Se evaluaron umbrales de 50% a 100% (sin PCA) usando LogisticRegression como clasificador
proxy sobre los embeddings disponibles (441,127 train, 110,282 test).
LogReg permite comparar representaciones en segundos con un ranking relativo confiable.

## Resultados

| Umbral | Dims | Accuracy | F1-Macro |
|--------|------|----------|----------|
| 50% | 41 | 0.5335 | 0.2962 |
| 60% | 62 | 0.5389 | 0.3040 |
| 70% | 91 | 0.5435 | 0.3102 |
| 75% | 110 | 0.5460 | 0.3141 |
| 80% | 131 | 0.5494 | 0.3190 |
| 85% | 156 | 0.5502 | 0.3231 |
| 90% | 188 | 0.5524 | 0.3278 |
| 95% | 232 | 0.5540 | 0.3308 | ← elegido
| 99% | 305 | 0.5585 | 0.3479 |
| 100% | 384 | 0.5604 | 0.3504 |

## Criterio de Selección

Se eligió el umbral con **máxima eficiencia marginal**: el último punto donde
la ganancia de accuracy por dimensión añadida (Δacc/Δdims) sigue siendo relevante
(≥10% de la mejor eficiencia observada en la curva).

Este criterio evita el sesgo del método "99% del máximo", que en curvas sin codo
abrupto tiende a elegir umbrales altos pagando muchas dimensiones por ganancias mínimas.

- **Accuracy sin PCA**: 0.5604
- **Accuracy con 95%**: 0.5540 (98.9% del máximo)
- **Pérdida aceptada**: 0.0064 (0.64%)

## Referencias

- Notebook: `notebooks/04_calibracion_pca.ipynb`  
- Datos: `results/pca_calibration.csv`  
- Gráfico: `results/pca_calibration.png`
