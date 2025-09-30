# Resultados del Clasificador Naive Bayes

## Información del Dataset
- **Columna objetivo:** 8 | Outcome
- **Total positivos:** 268
- **Total negativos:** 500

## División del Dataset
| Conjunto | Tamaño | Proporción Pos | Proporción Neg |
|----------|--------|----------------|----------------|
| Entrenamiento | 614 | 0.348534 | 0.651466 |
| Prueba | 154 | 0.350649 | 0.649351 |

**Conjunto de entrenamiento:**
- Positivos: 214
- Negativos: 400

## Estadísticas por Clase

| Variable | Clase Positiva (Media) | Clase Positiva (SD) | Clase Negativa (Media) | Clase Negativa (SD) |
|----------|------------------------|---------------------|------------------------|---------------------|
| Pregnancies | 5.00 | 3.72 | 3.25 | 3.00 |
| Glucose | 141.60 | 32.50 | 109.41 | 27.24 |
| BloodPressure | 71.54 | 20.89 | 68.63 | 17.53 |
| SkinThickness | 22.66 | 17.93 | 19.97 | 15.05 |
| Insulin | 98.03 | 132.57 | 67.19 | 100.42 |
| BMI | 35.36 | 7.63 | 30.49 | 7.42 |
| DiabetesPedigreeFunction | 0.55 | 0.37 | 0.44 | 0.30 |
| Age | 37.50 | 11.24 | 31.28 | 11.84 |
| Outcome | 1.00 | 0.00 | 0.00 | 0.00 |

## Matriz de Confusión

|  | Predicción: Positivo | Predicción: Negativo |
|---|---------------------|---------------------|
| **Real: Positivo** | TP: 34 | FN: 20 |
| **Real: Negativo** | FP: 24 | TN: 76 |

## Métricas de Evaluación

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.7143 (71.43%) |
| **Precision** | 0.5862 (58.62%) |
| **Recall** | 0.6296 (62.96%) |
| **F1-Score** | 0.6071 (60.71%) |

**Total de datos evaluados:** 154