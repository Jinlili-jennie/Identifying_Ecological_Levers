data <- read.csv("GenusSample.csv", header = TRUE, row.names = 1)

N <- nrow(data)
M <- ncol(data)

species_id <- c() 
sample_id <- c() 
absent_composition <- c()
absent_collection <- c()

for (j1 in 1:N) {
  print(paste("Genus", j1))
  
  for (j2 in 1:M) {
    
    if (data[j1, j2] > 0) {
      
      y_0 <- data[, j2]
      y_0_binary <- y_0
      y_0_binary[y_0_binary > 0] <- 1
      
      if (sum(y_0_binary) > 1) {
        
        y_0[j1] <- 0
        y_0[y_0 > 0] <- y_0[y_0 > 0] / sum(y_0[y_0 > 0])
        absent_composition <- cbind(absent_composition, y_0)
        
        species_id <- c(species_id, j1)
        sample_id <- c(sample_id, j2)
        
        absent_collection <- cbind(absent_collection, y_0_binary)
      }
    }
  }
}

write.table(species_id, file = 'Species_id.csv', row.names = FALSE, col.names = FALSE, sep = ",")
write.table(sample_id, file = 'Sample_id.csv', row.names = FALSE, col.names = FALSE, sep = ",")
write.table(absent_composition, file = 'Ptest.csv', row.names = FALSE, col.names = FALSE, sep = ",")
write.table(absent_collection, file = 'Ztest.csv', row.names = FALSE, col.names = FALSE, sep = ",")
