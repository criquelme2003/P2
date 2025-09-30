# Clasificador Bayesiano para Diabetes de los Indios Pima
# Universidad Católica de Temuco - INFO-1165

# Cargar librerías necesarias

library(mlbench)
library(ggplot2)
library(reshape2)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

# Cargar el dataset
data(PimaIndiansDiabetes)
datos <- PimaIndiansDiabetes

# Verificar estructura de los datos
cat("Dimensiones del dataset:", dim(datos), "\n")
cat("Estructura de los datos:\n")
str(datos)

# Verificar distribución de clases
table(datos$diabetes)
prop.table(table(datos$diabetes))

# ============================================================================
# 2. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA (80%-20%)
# ============================================================================

set.seed(123)  # Para reproducibilidad

# Crear índices estratificados para mantener proporción de clases
pos_indices <- which(datos$diabetes == "pos")
neg_indices <- which(datos$diabetes == "neg")

# Seleccionar 80% de cada clase para entrenamiento
train_pos <- sample(pos_indices, size = round(0.8 * length(pos_indices)))
train_neg <- sample(neg_indices, size = round(0.8 * length(neg_indices)))

train_indices <- c(train_pos, train_neg)
test_indices <- setdiff(1:nrow(datos), train_indices)

# Crear conjuntos de entrenamiento y prueba
datos_train <- datos[train_indices, ]
datos_test <- datos[test_indices, ]

# Verificar proporciones en ambos conjuntos
cat("Proporción en entrenamiento:\n")
prop.table(table(datos_train$diabetes))
cat("Proporción en prueba:\n")



prop.table(table(datos_test$diabetes))

# ============================================================================
# 3. CÁLCULO DE ESTADÍSTICAS PARA EL MODELO BAYESIANO
# ============================================================================

# Separar datos por clase
train_pos <- datos_train[datos_train$diabetes == "pos", ]
train_neg <- datos_train[datos_train$diabetes == "neg", ]

# Variables predictoras (todas excepto la variable objetivo)
variables <- names(datos)[1:8]

# Calcular medias y desviaciones estándar para cada clase
stats_pos <- list()
stats_neg <- list()

for (var in variables) {
  stats_pos[[var]] <- list(
    media = mean(train_pos[[var]], na.rm = TRUE),
    sd = sd(train_pos[[var]], na.rm = TRUE)
  )
  
  stats_neg[[var]] <- list(
    media = mean(train_neg[[var]], na.rm = TRUE),
    sd = sd(train_neg[[var]], na.rm = TRUE)
  )
}


stats_neg
prior_pos <- nrow(train_pos) / nrow(datos_train)

prior_neg <- nrow(train_neg) / nrow(datos_train)

cat("Probabilidad a priori P(Positivo):", prior_pos, "\n")
cat("Probabilidad a priori P(Negativo):", prior_neg, "\n")

# ============================================================================
# 4. FUNCIÓN DE CLASIFICACIÓN BAYESIANA
# ============================================================================

clasificar_bayesiano <- function(datos_test, stats_pos, stats_neg, prior_pos, prior_neg, epsilon = 0) {
  n_test <- nrow(datos_test)
  predicciones <- character(n_test)
  log_likelihood_pos <- numeric(n_test)
  log_likelihood_neg <- numeric(n_test)
  
  for (i in 1:n_test) {
    # Calcular log-verosimilitud para clase positiva
    log_prob_pos <- log(prior_pos)
    for (var in variables) {
      valor <- datos_test[i, var]
      if (!is.na(valor)) {
        prob_var <- dnorm(valor, 
                         mean = stats_pos[[var]]$media, 
                         sd = stats_pos[[var]]$sd)
        # Evitar log(0) añadiendo un pequeño valor
        log_prob_pos <- log_prob_pos + log(max(prob_var, 1e-300))
      }
    }
    
    # Calcular log-verosimilitud para clase negativa
    log_prob_neg <- log(prior_neg)
    for (var in variables) {
      valor <- datos_test[i, var]
      if (!is.na(valor)) {
        prob_var <- dnorm(valor, 
                         mean = stats_neg[[var]]$media, 
                         sd = stats_neg[[var]]$sd)
        # Evitar log(0) añadiendo un pequeño valor
        log_prob_neg <- log_prob_neg + log(max(prob_var, 1e-300))
      }
    }
    
    # Guardar log-verosimilitudes
    log_likelihood_pos[i] <- log_prob_pos
    log_likelihood_neg[i] <- log_prob_neg
    
    # Clasificar según la diferencia y el umbral epsilon
    diferencia <- log_prob_pos - log_prob_neg
    if (diferencia > epsilon) {
      predicciones[i] <- "pos"
    } else {
      predicciones[i] <- "neg"
    }
  }
  
  return(list(
    predicciones = predicciones,
    log_likelihood_pos = log_likelihood_pos,
    log_likelihood_neg = log_likelihood_neg
  ))
}

# ============================================================================
# 5. FUNCIÓN PARA CALCULAR MÉTRICAS DE EVALUACIÓN
# ============================================================================

calcular_metricas <- function(y_true, y_pred) {
  # Crear matriz de confusión
  TP <- sum(y_true == "pos" & y_pred == "pos")
  TN <- sum(y_true == "neg" & y_pred == "neg")
  FP <- sum(y_true == "neg" & y_pred == "pos")
  FN <- sum(y_true == "pos" & y_pred == "neg")
  
  # Calcular métricas
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1_score <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  
  return(list(
    TP = TP, TN = TN, FP = FP, FN = FN,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    accuracy = accuracy
  ))
}

# ============================================================================
# 6. EVALUACIÓN CON DIFERENTES UMBRALES EPSILON
# ============================================================================

# Crear vector de umbrales epsilon
epsilons <- seq(-3, 3, length.out = 100)
resultados <- data.frame(
  epsilon = epsilons,
  precision = numeric(length(epsilons)),
  recall = numeric(length(epsilons)),
  f1_score = numeric(length(epsilons)),
  accuracy = numeric(length(epsilons))
)

cat("Evaluando diferentes umbrales...\n")
pb <- txtProgressBar(min = 0, max = length(epsilons), style = 3)

for (i in 1:length(epsilons)) {
  eps <- epsilons[i]
  
  # Clasificar con el umbral actual
  resultado_clasificacion <- clasificar_bayesiano(datos_test, stats_pos, stats_neg, 
                                                 prior_pos, prior_neg, epsilon = eps)
  
  # Calcular métricas
  metricas <- calcular_metricas(datos_test$diabetes, resultado_clasificacion$predicciones)
  
  # Guardar resultados
  resultados[i, "precision"] <- metricas$precision
  resultados[i, "recall"] <- metricas$recall
  resultados[i, "f1_score"] <- metricas$f1_score
  resultados[i, "accuracy"] <- metricas$accuracy
  
  setTxtProgressBar(pb, i)
}
close(pb)

# ============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ============================================================================

# Transformar datos para ggplot2
resultados_long <- melt(resultados, id.vars = "epsilon", 
                       variable.name = "metrica", value.name = "valor")

# Crear gráfica con ggplot2
p1 <- ggplot(resultados_long, aes(x = epsilon, y = valor, color = metrica)) +
  geom_line(size = 1.2) +
  geom_point(size = 0.8, alpha = 0.7) +
  labs(
    title = "Métricas de Rendimiento vs Umbral de Decisión (ε)",
    subtitle = "Clasificador Bayesiano - Dataset Diabetes Pima",
    x = "Umbral de Decisión (ε)",
    y = "Valor de la Métrica",
    color = "Métrica"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  scale_color_manual(
    values = c("precision" = "#E31A1C", "recall" = "#1F78B4", 
              "f1_score" = "#33A02C", "accuracy" = "#FF7F00"),
    labels = c("Precisión", "Recall", "F1-Score", "Exactitud")
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1))

print(p1)

# ============================================================================
# 8. ANÁLISIS DE UMBRAL ÓPTIMO
# ============================================================================

# Encontrar umbral óptimo basado en F1-Score
indice_optimo <- which.max(resultados$f1_score)
epsilon_optimo <- resultados$epsilon[indice_optimo]
f1_optimo <- resultados$f1_score[indice_optimo]

cat("\n========== ANÁLISIS DE UMBRAL ÓPTIMO ==========\n")
cat("Umbral óptimo (ε):", round(epsilon_optimo, 4), "\n")
cat("F1-Score máximo:", round(f1_optimo, 4), "\n")

# Evaluar con el umbral óptimo
resultado_optimo <- clasificar_bayesiano(datos_test, stats_pos, stats_neg, 
                                        prior_pos, prior_neg, epsilon = epsilon_optimo)
metricas_optimo <- calcular_metricas(datos_test$diabetes, resultado_optimo$predicciones)

# Mostrar matriz de confusión y métricas del umbral óptimo
cat("\n========== MATRIZ DE CONFUSIÓN (Umbral Óptimo) ==========\n")
cat("Verdaderos Positivos (TP):", metricas_optimo$TP, "\n")
cat("Verdaderos Negativos (TN):", metricas_optimo$TN, "\n")
cat("Falsos Positivos (FP):", metricas_optimo$FP, "\n")
cat("Falsos Negativos (FN):", metricas_optimo$FN, "\n")

cat("\n========== MÉTRICAS DE RENDIMIENTO (Umbral Óptimo) ==========\n")
cat("Exactitud (Accuracy):", round(metricas_optimo$accuracy, 4), "\n")
cat("Precisión:", round(metricas_optimo$precision, 4), "\n")
cat("Recall (Sensibilidad):", round(metricas_optimo$recall, 4), "\n")
cat("F1-Score:", round(metricas_optimo$f1_score, 4), "\n")

# Crear gráfica adicional mostrando el punto óptimo
p2 <- ggplot(resultados, aes(x = epsilon, y = f1_score)) +
  geom_line(color = "#33A02C", size = 1.2) +
  geom_point(size = 0.8, alpha = 0.7, color = "#33A02C") +
  geom_vline(xintercept = epsilon_optimo, linetype = "dashed", color = "red", size = 1) +
  geom_point(aes(x = epsilon_optimo, y = f1_optimo), color = "red", size = 3) +
  annotate("text", x = epsilon_optimo + 0.3, y = f1_optimo - 0.02, 
           label = paste("ε óptimo =", round(epsilon_optimo, 3)), 
           color = "red", size = 4) +
  labs(
    title = "F1-Score vs Umbral de Decisión",
    subtitle = "Identificación del Umbral Óptimo",
    x = "Umbral de Decisión (ε)",
    y = "F1-Score"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12)
  ) +
  scale_y_continuous(limits = c(0, 1))

print(p2)

# ============================================================================
# 9. ANÁLISIS DEL DESBALANCE DE CLASES
# ============================================================================

cat("\n========== ANÁLISIS DEL DESBALANCE DE CLASES ==========\n")
cat("Distribución en conjunto de entrenamiento:\n")
print(table(datos_train$diabetes))
cat("Proporción en entrenamiento:\n")
print(prop.table(table(datos_train$diabetes)))

cat("\nDistribución en conjunto de prueba:\n")
print(table(datos_test$diabetes))
cat("Proporción en prueba:\n")
print(prop.table(table(datos_test$diabetes)))

# Evaluar con epsilon = 0 (sin ajuste por desbalance)
resultado_sin_ajuste <- clasificar_bayesiano(datos_test, stats_pos, stats_neg, 
                                            prior_pos, prior_neg, epsilon = 0)
metricas_sin_ajuste <- calcular_metricas(datos_test$diabetes, resultado_sin_ajuste$predicciones)

cat("\n========== COMPARACIÓN: SIN AJUSTE vs CON UMBRAL ÓPTIMO ==========\n")
cat("Sin ajuste (ε = 0):\n")
cat("  Exactitud:", round(metricas_sin_ajuste$accuracy, 4), "\n")
cat("  Precisión:", round(metricas_sin_ajuste$precision, 4), "\n")
cat("  Recall:", round(metricas_sin_ajuste$recall, 4), "\n")
cat("  F1-Score:", round(metricas_sin_ajuste$f1_score, 4), "\n")

cat("\nCon umbral óptimo (ε =", round(epsilon_optimo, 4), "):\n")
cat("  Exactitud:", round(metricas_optimo$accuracy, 4), "\n")
cat("  Precisión:", round(metricas_optimo$precision, 4), "\n")
cat("  Recall:", round(metricas_optimo$recall, 4), "\n")
cat("  F1-Score:", round(metricas_optimo$f1_score, 4), "\n")

cat("\n========== PROGRAMA COMPLETADO ==========\n")




