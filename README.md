# 🧮 RESULTADOS MONTE CARLO

---

*1000 iterations performed in **866ms** (global)*

**Umbral óptimo final:** `-1.3084 ± 0.612853`  
**Score promedio:** `0.695732`

---

## ⚙️ FASE 2: ENTRENAMIENTO FINAL

**Usando umbral final (según F2-SCORE):** `-1.3084`

---

### Entrenamiento

- Muestras totales: **614**
  - Clase Positiva: **214**
  - Clase Negativa: **400**

**Estado:** Entrenamiento completado ✅

---

### 📊 Estadísticas del modelo final

| Variable                  | Clase Positiva (Media ± SD) | Clase Negativa (Media ± SD) |
|----------------------------|------------------------------|------------------------------|
| **Pregnancies**            | 5 ± 3.71774                  | 3.2525 ± 3.00104             |
| **Glucose**                | 141.598 ± 32.4999            | 109.41 ± 27.2393             |
| **BloodPressure**          | 71.5421 ± 20.8909            | 68.6325 ± 17.5336            |
| **SkinThickness**          | 22.6636 ± 17.9255            | 19.97 ± 15.0503              |
| **Insulin**                | 98.0327 ± 132.573            | 67.1875 ± 100.42             |
| **BMI**                    | 35.3556 ± 7.6336             | 30.49 ± 7.4165               |
| **DiabetesPedigreeFunction** | 0.54885 ± 0.368846          | 0.436983 ± 0.302458          |
| **Age**                    | 37.5047 ± 11.2395            | 31.28 ± 11.8443              |
| **Outcome**                | 1 ± 0                        | 0 ± 0                        |

---

## 🧩 FASE 3: EVALUACIÓN EN TEST

**Priors:**  
P(pos) = 0.5  
P(neg) = 0.5  

---

### Resultados con Umbral Optimizado

**Umbral usado:** `-1.3084`

#### Matriz de Confusión

|              | Pred. Positivo | Pred. Negativo |
|---------------|----------------|----------------|
| **Real Positivo** | TP = 45         | FN = 9          |
| **Real Negativo** | FP = 42         | TN = 58         |

---

### 📈 Métricas de Desempeño

| Métrica     | Valor     |
|--------------|-----------|
| **Accuracy** | 0.668831  |
| **Precision**| 0.517241  |
| **Recall**   | 0.833333  |
| **F1-Score** | 0.638298  |

---
