**Base de Datos Masivas (11088)**

**Aprendizaje Supervisado para Clasificar Emociones en Canciones**

**Universidad Nacional de Luján – 2025**

Pittavino, Patricio (121476)

# INTRODUCCIÓN

El análisis automático de información musical mediante técnicas de aprendizaje supervisado constituye un área de creciente interés en la industria del entretenimiento. A través de la extracción de atributos acústicos, es posible caracterizar canciones y relacionarlas con percepciones humanas como las emociones que provocan.  
La clasificación automatizada de estas emociones podría ser utilizada en aplicaciones capaces de recomendar contenido personalizado o crear playlist según el estado de ánimo del oyente.  
Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje supervisado capaz de identificar con precisión la emoción principal asociada a una canción, aplicando varios algoritmos y comparando su rendimiento y efectividad.

# MOTIVACIÓN

La motivación de la propuesta reside bajo dos pilares:

1. **Aplicación de conocimientos adquiridos**: Implementar técnicas avanzadas de aprendizaje supervisado, combinando herramientas y algoritmos no vistos en el curso, para resolver un problema complejo, a la vez que se comparan factores como rendimiento, tiempo de ejecución y métricas de evaluación.
2. **Aplicaciones innovadoras**: Crear un modelo de aprendizaje propio que logre integrarse en una aplicación y realice recomendaciones musicales basadas en estados de ánimo y organice automáticamente catálogos musicales.

# OBJETIVOS

- Aplicar el proceso de KDD en un conjunto de datos de características musicales.
- Implementar y entrenar modelos de aprendizaje supervisado que clasifiquen y logren predecir la emoción predominante de una canción a partir de sus atributos acústicos.
- Implementar y comparar dos algoritmos de aprendizaje supervisado: Random Forest o K-Nearest Neighbors.
- Evaluar el desempeño del modelo mediante métricas como la Precisión, F-Score, matriz de confusión, ajustando hiperparámetros para mejorar su efectividad.

# TRABAJOS RELACIONADOS

Existen numerosas investigaciones orientadas a la clasificación de emociones en música, muchas de las cuales utilizan herramientas avanzadas como redes neuronales, lógica difusa o sistemas jerárquicos de decisión. En general, estos trabajos se enfocan en procesar directamente el audio para extraer automáticamente los atributos necesarios y predecir emociones asociadas a cada canción.  
Si bien estos trabajos utilizan metodologías avanzadas, el presente proyecto se focaliza en un enfoque supervisado más accesible. A partir de un conjunto de datos que ya contiene atributos acústicos preprocesados, se busca comparar modelos de clasificación como Random Forest y k-Nearest Neighbors, evaluando su capacidad de predecir la emoción predominante de cada canción.

# METODOLOGÍA

**1\. Selección del Dataset**

Se utilizará el dataset público con canciones etiquetadas según la emoción predominante[^1], junto con sus letras y variables acústicas.

### **2\. Preprocesamiento de Datos**

- Limpieza de datos y manejo de valores nulos o inconsistentes.
- Generación de _embeddings_ numéricos de las letras mediante la librería _sentence-transformers_
- Análisis exploratorio de la distribución de embeddings utilizando reducción de dimensionalidad con PCA.
- División en conjunto de entrenamiento y prueba.

### **3\. Entrenamiento del Modelo**

Se investigarán e implementarán dos algoritmos de aprendizaje supervisado: Random Forest y K-Nearest Neighbors, a la vez que se realizará un ajuste de hiperparametros sobre cada uno.

### **4\. Evaluación y Ajuste**

Se medirán métricas de desempeño como:

- Precisión, Recall y F1-Score.
- Matriz de Confusión para analizar errores de clasificación.
- Comparación de modelos para seleccionar el más efectivo.

# RESULTADOS ESPERADOS

Se espera obtener un modelo que clasifique canciones en categorías emocionales con una precisión aceptable, a la vez que se analizen qué variables son las más determinantes. También se espera lograr una tabla comparativa de la evaluación de los algoritmos utilizados.

# BIBLIOGRAFÍA

[**A survey of music emotion recognition**](https://www.researchgate.net/profile/Donghong-Han/publication/358059278_A_survey_of_music_emotion_recognition/links/62b3cf741010dc02cc54cc2d/A-survey-of-music-emotion-recognition.pdf)  
[**Music Emotion Classification: A Fuzzy Approach**](https://www.researchgate.net/profile/Homer-Chen/publication/221571255_Music_emotion_classification_A_fuzzy_approach/links/53fd426c0cf22f21c2f7e178/Music-emotion-classification-A-fuzzy-approach.pdf)  
[**Are We There Yet? A Brief Survey of Music Emotion Prediction Datasets, Models and Outstanding Challenges**](https://arxiv.org/pdf/2406.08809)  
[**Emotion recognition using a hierarchical binary decision tree approach**](https://sail.usc.edu/publications/files/cclee-speechcomm-2011.pdf)

[^1]: 500K+ Spotify Songs with Lyrics,Emotions & More. ([Kaggle Dataset](https://www.kaggle.com/datasets/devdope/900k-spotify/data?select=spotify_dataset.csv))
