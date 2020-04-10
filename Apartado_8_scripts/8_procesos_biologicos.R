BiocManager::install("ChIPseeker")
BiocManager::install("TxDb.Hsapiens.UCSC.hg19.knownGene")
BiocManager::install("EnsDb.Hsapiens.v75")
BiocManager::install("AnnotationDbi")
BiocManager::install("org.Hs.eg.db")
BiocManager::install("clusterProfiler")

library("ChIPseeker")
library("TxDb.Hsapiens.UCSC.hg19.knownGene")
library("EnsDb.Hsapiens.v75")
library("AnnotationDbi")
library("org.Hs.eg.db")
library("clusterProfiler")

symbols <- read.table("./gene_list.txt", header = T)
head(symbols)

symbols <- scan("./gene_list.txt", what="", sep="\n")
symbols[1]
entrez <- mapIds(org.Hs.eg.db, symbols, 'ENTREZID', 'SYMBOL')

ego <- enrichGO(gene = entrez, 
                keyType = "ENTREZID", 
                OrgDb = org.Hs.eg.db, 
                ont = "BP", 
                pAdjustMethod = "BH", 
                qvalueCutoff = 0.05, 
                readable = TRUE)

cluster_summary <- data.frame(ego)
write.csv(cluster_summary, "clusterProfiler.csv")

dotplot(ego, showCategory=30)

      ekegg <- enrichKEGG(gene = entrez,
                    organism = 'hsa',
                    pvalueCutoff = 0.05)

dotplot(ekegg)
