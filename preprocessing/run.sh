# !/bin/bash

exp_metadata=/home/data/exp_metadata.csv

while read line
do
    arr=(${line//,/ })
    tmp=${arr[1]}
    datafile=${arr[3]}

    if [ "$tmp" = "exp_name" ]; then
        continue
    fi
    echo $tmp
    echo $datafile
    # cutadapt -j 12 \
    # 	-g "^GGG" -a "A{10}" -n 2\
    # 	-m 15 \
    # 	--max-n=0.1 \
    # 	--discard-casava \
    # 	-o /home/result/trimmed_${tmp}.fastq.gz \
    # 	/home/data/${datafile} > log_cutadapt_${tmp}.txt
    
    # bowtie --quiet \
	# 	-q \
	# 	-v 0 \
	# 	--norc \
	# 	-p 12 \
	# 	-S \
	# 	--sam-nohead \
	# 	--un /home/result/rRNA_left_${tmp}.fastq \
	# 	-q /home/ref/Homo_sapiens_yuanhui/ref_rRNA/rRNA_NCBI_ENS_merged \
	# 	<(zcat /home/result/trimmed_${tmp}.fastq.gz) \
	# 	| awk 'BEGIN{FS="\t"}{if($2==0){print}}' \
	# 	>> /home/result/rRNA_align_${tmp}.fastq
    # rm -rf /home/result/rRNA_align_${tmp}.fastq
    # gzip /home/result/rRNA_left_${tmp}.fastq
    # rm -rf /home/result/trimmed_${tmp}.fastq.gz

    # STAR --runThreadN 12 \
    #     --genomeDir /home/ref/Homo_sapiens_109_saori/STAR_ref \
    #     --readFilesIn /home/result/rRNA_left_${tmp}.fastq.gz \
    #     --outSAMtype BAM SortedByCoordinate  \
    #     --readFilesCommand zcat \
    #     --runDirPerm All_RWX \
    #     --outFileNamePrefix /home/result/STAR_align_$tmp\_ \
    #     --outSAMattributes All \
    #     --outFilterScoreMinOverLread 0 \
    #     --outFilterMatchNminOverLread 0 \
    #     --outBAMsortingBinsN 200
    
    # file=/home/result/STAR_align_${tmp}_Aligned.sortedByCoord.out.bam
    # samtools view -H $file > /home/result/header
    # samtools view $file \
    #     | grep -P "^\S+\s0\s" \
    #     | grep -P "NH:i:1\b" \
    #     | grep -E -w 'NM:i:0|NM:i:1|NM:i:2' \
    #     | cat /home/result/header -| samtools view -bS ->/home/result/uniq_STAR_align_${tmp}.bam
    # samtools index /home/result/uniq_STAR_align_${tmp}.bam

    file=/home/result/STAR_align_${tmp}_Aligned.sortedByCoord.out.bam
    samtools view -H $file > /home/result/header
    samtools view $file \
        | grep -P "^\S+\s0\s" \
        | grep -E -w 'NM:i:0|NM:i:1|NM:i:2' \
        | cat /home/result/header -| samtools view -bS ->/home/result/primary_STAR_align_${tmp}.bam
    samtools index /home/result/primary_STAR_align_${tmp}.bam

    # rm -rf /home/result/rRNA_left_${tmp}.fastq.gz

done < $exp_metadata