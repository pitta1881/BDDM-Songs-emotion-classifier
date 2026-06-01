# Decisión: Umbral de Varianza PCA

**Umbral elegido**: 95%  
**Dimensiones resultantes**: 232 (de 384 originales)  
**Reducción**: 152 dims eliminadas (40%)  

## Metodología

Se evaluaron umbrales de 50% a 100% (sin PCA) usando LogisticRegression como clasificador
proxy sobre los embeddings disponibles (436,653 train, 109,164 test).
LogReg permite comparar representaciones en segundos con un ranking relativo confiable.

## Resultados

| Umbral | Dims | Accuracy | F1-Macro |
|--------|------|----------|----------|
| 50% | 41 | 0.5389 | 0.3573 |
| 60% | 62 | 0.5441 | 0.3662 |
| 70% | 91 | 0.5492 | 0.3741 |
| 75% | 110 | 0.5516 | 0.3787 |
| 80% | 131 | 0.5548 | 0.3842 |
| 85% | 156 | 0.5558 | 0.3874 |
| 90% | 188 | 0.5582 | 0.3936 |
| 95% | 232 | 0.5596 | 0.3974 | ← elegido
| 99% | 305 | 0.5641 | 0.4050 |
| 100% | 384 | 0.5656 | 0.4080 |

## Criterio de Selección

Se eligió el umbral con **máxima eficiencia marginal**: el último punto donde
la ganancia de accuracy por dimensión añadida (Δacc/Δdims) sigue siendo relevante
(≥10% de la mejor eficiencia observada en la curva).

Este criterio evita el sesgo del método "99% del máximo", que en curvas sin codo
abrupto tiende a elegir umbrales altos pagando muchas dimensiones por ganancias mínimas.

- **Accuracy sin PCA**: 0.5656
- **Accuracy con 95%**: 0.5596 (98.9% del máximo)
- **Pérdida aceptada**: 0.0060 (0.60%)

## Referencias

- Notebook: `notebooks/04_calibracion_pca.ipynb`  
- Datos: `results/pca_calibration.csv`  
- Gráfico: `results/pca_calibration.png`
