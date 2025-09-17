# load gene x count matrix
library(dplyr)
library(data.table)
data_path <- '/home/rstudio/result/res20230710/data20230710_shiying_bin/cpm/count_mat.csv.gz'

dt = fread(data_path,header=TRUE) %>% as.matrix(rownames=1)
smpls <- colnames(dt)
grp <- unlist(lapply(smpls, function(x){strsplit(x , "_rep[12]")[[1]][1]}))
genes <- rownames(dt)

time <- factor(rep(c("0min","60min","360min"),4))
type <- factor(c(rep("RPF",3),rep("mRNA",3),rep("RPF",3),rep("mRNA",3)))

########################################

library(edgeR)
d <- DGEList(
  counts=dt,
  genes=genes,
  group=grp)

# filering
d_full <- d
idx_keep <- rowSums(cpm(d)>100) >= 1
d <- d[idx_keep,]
dim(d)
d$samples$lib.size <- colSums(d$counts)

# normalize data by TMM
d <- calcNormFactors(d,method = 'TMM')
nc <- cpm(d, normalized.lib.sizes=TRUE) %>% as.data.table
genes_now <- d$genes %>% as.vector
row.names(nc) <- genes_now$genes
fwrite(
  nc,
  '/home/rstudio/result/res20230710/data20230710_shiying_bin/cpm/edgeR/count_mat_TMM.csv',
  row.names = TRUE)

# fix the dispersion value
d <- estimateDisp(d)

############

time <- factor(rep(c("0min","60min","360min"),4))
type <- factor(c(rep("RPF",3),rep("mRNA",3),rep("RPF",3),rep("mRNA",3)))
batch <- factor(cbind(rep(1,6),rep(2,6)))

df <- data.frame(
  time = time,
  type = type,
  batch = batch
)
rownames(df) <- smpls

design <- model.matrix(~ batch + time + type + time:type)

fit <- glmQLFit(d, design)

qlf <- glmQLFTest(fit, coef=5)
fwrite(
  qlf$table,
  paste0(
      '/home/rstudio/result/res20230710/data20230710_shiying_bin/cpm/edgeR/qlf_',
      'TE360min',
      '.csv'),
    row.names = TRUE)

qlf <- glmQLFTest(fit, coef=6)
fwrite(
  qlf$table,
  paste0(
    '/home/rstudio/result/res20230710/data20230710_shiying_bin/cpm/edgeR/qlf_',
    'TE60min',
    '.csv'),
  row.names = TRUE)

# 
# pairs <- list(
#   c("RPF_0min","RPF_60min"),
#   c("RPF_0min","RPF_360min"),
#   c("mRNA_0min","mRNA_60min"),
#   c("mRNA_0min","mRNA_360min")
# )
# 
# for (p in pairs){
#   et <- exactTest(d,pair=p)
#   fwrite(
#     et$table,
#     paste0(
#       '/home/rstudio/result/res20230710/data20230710_shiying/cpm/exact_test_',
#       p[1],'_',p[2],
#       '.csv'),
#     row.names = TRUE)
# }