library(ggplot2)
library(scales)
library(ggpubr)
library(gridExtra)
library(ggExtra)

species_id = read.table(file = 'Species_id.csv', header = F, sep=",")
sample_id = read.table(file = 'Sample_id.csv', header = F, sep=",")
Ptrain = read.table(file = 'GenusSample.csv', header = F, sep=",")
qtst = read.table(file = 'qtst.csv', header = F, sep=",")
qtrn = read.table(file = 'qtrn.csv', header = F, sep=",")

num_species = nrow(Ptrain)
num_samples = ncol(Ptrain)

keystoneness_matrix = matrix(0, nrow = num_samples, ncol = num_species)
rownames(keystoneness_matrix) = 1:num_samples
colnames(keystoneness_matrix) = 1:num_species

for (i in 1:nrow(sample_id)){ 
  sample_idx = sample_id$V1[i]  
  species_idx = species_id$V1[i]  
  
  if (sample_idx > 0 & sample_idx <= num_samples & species_idx > 0 & species_idx <= num_species) {
    
    q_i = qtrn[sample_idx,]
    q_i_null = q_i
    q_i_null[species_idx] = 0
    q_i_null = q_i_null / sum(q_i_null)
    
    p_i = Ptrain[, sample_idx]
    p_i_null = p_i
    p_i_null[species_idx] = 0
    p_i_null = p_i_null / sum(p_i_null)
    
    BC_pred = sum(abs(q_i_null - qtst[i,])) / sum(abs(q_i_null + qtst[i,]))
    
    keystone_predicted_value = BC_pred * as.numeric(1 - Ptrain[species_idx, sample_idx])
    
    keystoneness_matrix[sample_idx, species_idx] = keystone_predicted_value
  }
}

keystoneness_df = as.data.frame(keystoneness_matrix)

write.csv(keystoneness_df, file = "Ks_keystoneness.csv", row.names = TRUE)