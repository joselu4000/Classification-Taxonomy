# R 4.3.2.
# FILE: grafico_filogenetico.R
# AUTHOR: José Luis López Carmona
# CREATE DATE: 29/05/2024

# Libraries
library(readr)  
library(dplyr)  

# Read data
input_micro <- "C:/Users/JoseLuisLopezCarmona/Documents/MCD/TFM/Codigo/datos/taxonomy.csv"
datos <- read_csv(input_micro)

# Reindex
datos <- datos[, -1]
datos <- datos %>% distinct()
etiquetas <- datos$GENUS
datos <- datos %>% mutate(across(where(is.character), as.factor) %>%
                            mutate(across(where(is.factor), as.integer)))
row.names(datos) <- etiquetas

# Distance 
distancia <- dist(datos, method = "euclidean")
cluster <- hclust(distancia)

# Plot
library(ggplot2)  
dend_plot <- fviz_dend(cluster, k = 100, cex = 0.55, type = "circular", lwd = 1, 
                       xlab = "", ylab = "", repel = FALSE)
dend_plot <- dend_plot + theme(
  axis.title.y = element_blank(),  
  axis.text.y = element_blank(),   
  axis.ticks.y = element_blank()   
)

# Explivative circles:
# Phylum 1
dend_plot <- dend_plot +
  geom_point(aes(x = -1.8, y = 103), color = "black", size = 4) +
  annotate("text", x = -1.7, y = 95, label = "Filo", color = "black", size = 5, angle = 0, fontface = "bold", hjust = -0.1)
# Clase 3
dend_plot <- dend_plot +
  annotate("text", x = -1.8, y = 55, label = "Clase", color = "darkblue", size = 5, angle = 0, fontface = "bold") +
  annotate("segment", x = 3, xend = 95, y = 55, yend = 55, colour = "darkblue", size = 3, alpha = 1)
# Order 20
dend_plot <- dend_plot +
  annotate("text", x = -1.8, y = 20, label = "Orden", color = "darkblue", size = 5, angle = 0, fontface = "bold") +
  annotate("segment", x = 1, xend = 97, y = 20, yend = 20, colour = "darkblue", size = 3, alpha = 0.7)
# Family 39
dend_plot <- dend_plot +
  annotate("text", x = -1.8, y = 10, label = "Familia", color = "darkblue", size = 5, angle = 0, fontface = "bold") +
  annotate("segment", x = 1, xend = 97, y = 10, yend = 10, colour = "darkblue", size = 3, alpha = 0.5)
# Genus 100
dend_plot <- dend_plot +
  annotate("text", x = -1.8, y = 5, label = "Género", color = "black", size = 5, angle = 0, fontface = "bold") +
  annotate("segment", x = 1, xend = 97, y = 5, yend = 5, colour = "blue", size = 3, alpha = 0.2)


# Print plot
print(dend_plot)
