# Clasificación de Emociones en Música 🎵

Trabajo práctico de BDDM (Base de Datos y Minería de Datos) — UNLu 2024/2025.

La idea es simple: dado un dataset de ~550K canciones de Spotify, predecir la emoción que transmite cada una (alegría, tristeza, enojo, miedo o amor) usando características acústicas y la letra de la canción.

---

## Por qué el género usa Frequency Encoding (y no One-Hot)

Esta es una de las decisiones de preprocesamiento más interesantes del proyecto, así que la explico desde cero.

### El problema con variables categóricas

Los modelos de ML trabajan con números. Cuando tenés una columna como `Genre` con valores como `"hip hop"`, `"reggae"` o `"rock"`, necesitás convertirla a números de alguna manera. La pregunta es *cómo*.

### One-Hot Encoding — cuándo usarlo

One-Hot crea una columna binaria por cada categoría única:

```
Genre       →   genre_rock  genre_pop  genre_jazz
rock        →      1           0          0
pop         →      0           1          0
jazz        →      0           0          1
```

**Ventaja**: no introduce orden artificial ("pop=2 > rock=1" sería mentira). El modelo ve cada categoría como independiente.

**Cuándo funciona bien**: cuando la variable tiene **pocas categorías** — digamos menos de 10 o 15. Con 5 géneros bien definidos, añadís 5 columnas y listo.

**El problema**: si tenés 200 géneros únicos, One-Hot te genera 200 columnas nuevas. La mayoría van a ser casi todas ceros (sparse matrix). Eso es:
- Más memoria
- Más ruido para el modelo
- Más riesgo de overfitting en categorías con pocas muestras
- La maldición de la dimensionalidad siendo tu enemiga

### Frequency Encoding — cuándo usarlo

Frequency Encoding reemplaza cada categoría por la frecuencia con la que aparece en el dataset:

```
Genre            →   Genre_freq
hip hop          →   0.312   (31.2% del dataset)
reggae           →   0.089
rock,pop,comedy  →   0.003   (categoría rara → valor bajo)
```

**Ventaja**: una sola columna, sin explotar la dimensionalidad. Y encima tiene semántica útil: los géneros raros tienen valores bajos, los dominantes tienen valores altos. El modelo puede aprender que un género con freq=0.001 es un nicho poco representado.

**Cuándo funciona bien**: cuando la variable tiene **alta cardinalidad** — muchas categorías únicas.

### Por qué aplicamos Frequency Encoding acá

En este dataset, `Genre` tiene **más de 200 valores únicos** en el dataset completo. Y no son géneros limpios — son combinaciones como `"hip hop,trap,cloud rap"` o `"pop,pop rock,pop punk"` tratadas como una sola categoría. One-Hot hubiera generado 200+ columnas de las cuales la gran mayoría serían casi cero.

Con Frequency Encoding, esa información queda comprimida en una sola columna `Genre_freq` que le dice al modelo "este género es común / raro en el dataset". Suficiente señal, sin el ruido.

---

## El Dataset

- **Fuente**: Kaggle — Spotify Songs Dataset
- **Tamaño raw**: 551,443 canciones × 37 columnas
- **Después del filtro de clases**: 545,825 canciones
- **Después de limpiar nulos**: 545,817 canciones (se eliminaron 8 filas = 0.0015% — trivial)
- **Target**: columna `emotion` con 5 clases: `joy`, `sadness`, `anger`, `fear`, `love`
- **Split**: 80% train (436,653) / 20% test (109,164), estratificado por emoción

El archivo CSV va en `data/spotify_dataset.csv`. No está en el repo porque pesa ~80MB.

---

## Por qué estas 5 clases

El dataset original tiene más clases, pero muchas tienen menos de 5K ejemplos. Con 550K canciones distribuidas de forma desigual, entrenar con clases minoritarias genera modelos que las ignoran. La decisión fue quedarnos con las 5 que tienen representación suficiente para que el modelo pueda aprender algo real de cada una.

---

## Qué hace cada notebook

| # | Notebook | Qué hace |
|---|----------|----------|
| 01 | `01_exploracion_datos.ipynb` | EDA: distribución de emociones, correlaciones, análisis de columnas. Acá se decide qué clases y qué columnas quedan. |
| 02 | `02_preprocesamiento.ipynb` | Limpieza de nulos, conversión de tipos, encoding de variables categóricas, split estratificado 80/20. |
| 03 | `03_embeddings.ipynb` | Genera embeddings de las letras con `all-MiniLM-L6-v2` (384 dimensiones por canción). |
| 04 | `04_calibracion_pca.ipynb` | Busca cuántas dimensiones de PCA capturan el 80% de la varianza, usando Logistic Regression como proxy rápido. |
| 05 | `05_embeddings_pca.ipynb` | Aplica PCA al resultado del notebook 03. Resultado: 131 dimensiones en lugar de 384. |
| 06 | `06_random_forest.ipynb` | Optuna busca los mejores hiperparámetros de RF en un subset. Luego entrena en el dataset completo. |
| 07 | `07_xgboost.ipynb` | Mismo proceso para XGBoost. |
| 08 | `08_comparacion.ipynb` | Compara RF vs XGBoost en métricas, matrices de confusión y rendimiento por clase. |

---

## Por qué PCA y por qué 131 dimensiones

Los embeddings de `all-MiniLM-L6-v2` tienen 384 dimensiones. Muchas de esas dimensiones están correlacionadas o aportan muy poco. PCA encuentra las combinaciones lineales que explican la mayor varianza.

El número 131 no es arbitrario — lo encontramos empíricamente: es el mínimo de dimensiones que preserva el 80% de la varianza del espacio original. Usamos Logistic Regression como proxy para validarlo rápido (notebook 04), y luego aplicamos ese PCA definitivo sobre todos los datos (notebook 05).

---

## Resultados Finales

| Modelo | Accuracy | F1 Weighted | F1 Macro | Veredicto |
|--------|----------|-------------|----------|-----------|
| Random Forest | 58.49% | 57.13% | 46.79% | ✅ Más balanceado |
| XGBoost | 58.85% | 56.78% | 43.66% | Más preciso en clases mayoritarias |

**Ganador recomendado: Random Forest.** Aunque XGBoost tiene marginalmente más accuracy, RF tiene mejor F1-macro — es decir, funciona mejor en las clases minoritarias (fear, love). Para una clasificación de emociones donde todas las clases importan por igual, eso pesa más.

---

## Setup

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Verificar
python -c "import sklearn, xgboost, sentence_transformers, optuna; print('OK')"
```

Los notebooks se ejecutan en orden del 01 al 08. Cada uno guarda su output en `data/processed/` o `data/embeddings/` como `.parquet`, así el siguiente puede levantarlo sin reprocesar.

---

## Estructura

```
.
├── data/
│   ├── spotify_dataset.csv    # Dataset original (descargar de Kaggle)
│   ├── processed/             # train.parquet, test.parquet
│   └── embeddings/            # embeddings y PCA reducidos
├── models/                    # rf_model.joblib, xgb_model.joblib, pca_model.joblib
├── results/                   # Gráficos, métricas, comparaciones
├── notebooks/                 # 01 → 08
├── decisions/                 # Registros de decisiones clave
└── requirements.txt
```

---

## Tecnologías

- **ML**: scikit-learn (Random Forest), XGBoost
- **NLP/Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Optimización de hiperparámetros**: Optuna
- **Reducción de dimensionalidad**: PCA (scikit-learn)
- **Storage**: Parquet (pyarrow) — mucho más rápido que CSV para recargas
- **Visualización**: matplotlib, seaborn

---

*Universidad Nacional de Luján — BDDM 2024/2025*
