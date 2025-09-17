from pathlib import Path
import numpy as np
import pandas as pd

cur_dir = Path(__file__)
save_dir = cur_dir.parent / "result/res20230710/data20230710_shiying_bin"
if not save_dir.exists():
    save_dir.mkdir(parents=True)

from myRiboSeq import myprep_bin

ref = myprep_bin.prep_data(
    save_dir=None,
    ref_dir = 'ref/Homo_sapiens_109_saori',
    data_dir="data/data20230710_shiying/uniq",
    sp = 'hsa')

smpls = ref.exp_metadata.df_metadata['sample_name'].values
# smpls = smpls[ np.array([0,1,2,6,7,8]) ]
col_list = dict(zip(smpls,["#808080"]*len(smpls)))

myprep_bin.read_stat(
    load_dir=save_dir,
    ref=ref,
    excel_file=save_dir / 'reads_info.xlsx'
)

# myprep_bin.calc_norm_coverage(
#     save_dir=save_dir,
#     load_dir=save_dir / 'prep_data',
#     smpls=smpls,
#     threshold_tr_count=8,
#     ref=ref
# )

from myRiboSeq import myqc_bin

# myqc_bin.region_mapped(
#     save_dir=save_dir,
#     load_dir=save_dir / 'prep_data',
#     threshold_tr_count=0,
#     smpls=smpls,
#     ref=ref,
#     read_len_list=[25,35]
# )

# myqc_bin.inframe_ratio(
#     save_dir=save_dir,
#     load_dir=save_dir / 'prep_data',
#     threshold_tr_count=8,
#     smpls=smpls,
#     readlen_list = [25,35]
# )

# myqc_bin.plot_region_len(
#     save_dir=save_dir,
#     load_dir=save_dir / 'prep_data',
#     col_list=col_list,
#     attrs=['read_length','length'],
#     ref=ref
# )

# myprep_bin.calc_norm_ribo_density(
#     save_dir=save_dir,
#     load_dir=save_dir / 'prep_data',
#     is_length=False,
#     smpls=smpls,
#     is_norm=True,
#     ref=ref
#     # read_len_list=[29,35]
# )

# myqc_bin.fft_ribo(
#     save_dir=save_dir,
#     load_dir=save_dir / 'norm_ribo_density',
#     dict_range={
#         '5_start':[90,240]
#     },
#     fname='all',
#     smpls=smpls,
#     readlen_list=[]
# )


# myqc_bin.aggregation_ribo(
#     save_dir=save_dir,
#     load_dir=save_dir / 'norm_ribo_density',
#     plot_range5=(-200,500),
#     plot_range3=(-560,200),
#     threshold_tr_count=0,
#     threshold_pos_count=1000,
#     col_list=col_list,
#     is_norm=True,
#     ref=ref
#     # read_len_list=[29,35]
#     )

# myqc.aggregation_plot_length(
#     save_dir=save_dir,
#     load_dir=save_dir / 'norm_ribo_density',
#     plot_range5=(-50,100),
#     plot_range3=(-100,50),
#     threshold_tr_count=0,
#     threshold_pos_count=1000,
#     col_list=col_list,
#     mode='heatmap'
# )

##############################

# pairs = [
#     ('RPF_0min_rep1','RPF_0min_rep2'),
#     ('RPF_60min_rep1','RPF_60min_rep2'),
#     ('RPF_360min_rep1','RPF_360min_rep2'),
#     ('mRNA_0min_rep1','mRNA_0min_rep2'),
#     ('mRNA_60min_rep1','mRNA_60min_rep2'),
#     ('mRNA_360min_rep1','mRNA_360min_rep2'),
# ]

pairs2 = [
    ('60min_rep1','0min_rep1'),
    ('360min_rep1','0min_rep1'),
    ('60min_rep2','0min_rep2'),
    ('360min_rep2','0min_rep2'),
]

pairs = [
    ('RPF_0min_rep1','mRNA_0min_rep1'),
    ('RPF_0min_rep2','mRNA_0min_rep2'),
    ('RPF_60min_rep1','mRNA_60min_rep1'),
    ('RPF_60min_rep2','mRNA_60min_rep2'),
    ('RPF_360min_rep1','mRNA_360min_rep1'),
    ('RPF_360min_rep2','mRNA_360min_rep2')
]

pairs = [
    ('RPF_0min','mRNA_0min'),
    ('RPF_60min','mRNA_60min'),
    ('RPF_360min','mRNA_360min'),
]

##############################

from myRiboSeq import Shiying

# Shiying.te(
#     save_dir=save_dir,
#     load_dir=save_dir / 'rpkm',
#     pairs=pairs,
#     pairs2=[
#         ('60min','0min'),
#         ('360min','0min')
#     ],
#     threshold=1.5,
#     threshold_tr=5,
#     nrep=2
# )

# Shiying.scatter_rpkm(
#     save_dir=save_dir,
#     load_dir=save_dir / 'rpkm',
#     ref=ref,
#     pairs=pairs,
#     threshold=0,
#     fname='_RPF-mRNA'
# )

# Shiying.cpm(
#     save_dir=save_dir,
#     load_dir=save_dir / 'prep_data',
#     ref=ref
# )

Shiying.volcano_plot(
    save_dir=save_dir,
    load_data=save_dir / 'cpm' / 'edgeR' / 'qlf_TE360min.csv.gz',
    threshold_fc=1.5,
    threshold_pval=0.1,
    tr_highlight=[],
    ref=ref,
    fname='edgeR_360min'
)

Shiying.volcano_plot(
    save_dir=save_dir,
    load_data=save_dir / 'cpm' / 'edgeR' / 'qlf_TE60min.csv.gz',
    threshold_fc=1.5,
    threshold_pval=0.1,
    tr_highlight=[],
    ref=ref,
    fname='edgeR_60min'
)

Shiying.volcano_plot(
    save_dir=save_dir,
    load_data=save_dir / 'cpm' / 'DESeq2' / 'time360min' / 'deltaTE.txt.gz',
    threshold_fc=1.5,
    threshold_pval=0.1,
    tr_highlight=[],
    ref=ref,
    fname='DESeq2_360min'
)

Shiying.volcano_plot(
    save_dir=save_dir,
    load_data=save_dir / 'cpm' / 'DESeq2' / 'time60min' / 'deltaTE.txt.gz',
    threshold_fc=1.5,
    threshold_pval=0.1,
    tr_highlight=[],
    ref=ref,
    fname='DESeq2_60min'
)

# Shiying.merge_DESeq2_edgeR(
#     save_dir=save_dir,
#     files_deseq2={
#         '60min':save_dir / 'cpm' / 'DESeq2' / 'time60min' / 'deltaTE.txt.gz',
#         '360min':save_dir / 'cpm' / 'DESeq2' / 'time360min' / 'deltaTE.txt.gz'
#     },
#     files_edger={
#         '60min':save_dir / 'cpm' / 'edgeR' / 'qlf_TE60min.csv.gz',
#         '360min':save_dir / 'cpm' / 'edgeR' / 'qlf_TE360min.csv.gz'
#     },
#     ref=ref
# )

# Shiying.heatmap_timecourse(
#     save_dir=save_dir,
#     load_dir=save_dir / 'translation_efficiency',
#     smpls=['0min','60min','360min'],
#     ref=ref,
#     n_cluster=5
# )

#################

# threshold_fc = 1.5
# threshold_pval = 0.1
# df_cluster = pd.read_csv(save_dir / 'heatmap_timecourse' / 'clustering_TE_6.csv.gz',index_col=0,header=0)

# for t in [60,360]:
#     df = pd.read_csv(save_dir / 'cpm' / 'DESeq2' / f'time{t}min' / 'deltaTE.txt.gz',index_col=0,sep='\t')
#     df_cluster[f'DESeq2_{t}min'] = [''] * len(df_cluster)
#     for tt,tt_str in zip(['>','<'],['increase','decrease']):
#         tr_list = list(df.query(f'(log2FoldChange {tt}= {np.log2(threshold_fc)}) and (pvalue <= {threshold_pval})').index)
#         for tr in tr_list:
#             if tr in df_cluster.index:
#                 df_cluster.loc[ tr, f'DESeq2_{t}min' ] = tt_str

# for t in [60,360]:
#     df = pd.read_csv(save_dir / 'cpm' / 'edgeR' / f'qlf_TE{t}min.csv.gz',index_col=0)
#     df_cluster[f'edgeR_{t}min'] = [''] * len(df_cluster)
#     for tt,tt_str in zip(['>','<'],['increase','decrease']):
#         tr_list = list(df.query(f'(logFC {tt}= {np.log2(threshold_fc)}) and (PValue <= {threshold_pval})').index)
#         for tr in tr_list:
#             if tr in df_cluster.index:
#                 df_cluster.loc[ tr, f'edgeR_{t}min' ] = tt_str

# df_cluster.to_csv(save_dir / 'heatmap_timecourse' / 'clustering_TE_6_add.csv.gz')
print("hoge")

#################

df = pd.read_csv(save_dir / 'heatmap_timecourse' / 'clustering_TE_8_add.csv.gz',index_col=0,header=0)
smpls = smpls[np.array([0,6,1,7,2,8,3,9,4,10,5,11])]
for n in [0,2,4,6]:
    tr_list = list(df.query(f'(cluster == {n}) and ((DESeq2_60min != "") or (DESeq2_360min != "") or (edgeR_60min != "") or (edgeR_360min != ""))').index)
    tr_list_names = [
            ref.id.dict_name[tr]['symbol']
            for tr in tr_list
        ]
    print(f'{len(tr_list)} transcripts...')
    n_for = int(round(len(tr_list)/50,0))
    for i in range(n_for):
        Shiying.indiv_plot(
            save_dir=save_dir,
            load_dir=save_dir / 'norm_coverage',
            tr_list=tr_list[i*50:(i+1)*50],
            tr_list_names=tr_list_names[i*50:(i+1)*50],
            fname=f'_cluster{n}_{i}',
            mode='coverage',
            ref=ref,
            smpls=smpls
        )
    if i*50 < len(tr_list):
        i += 1
        Shiying.indiv_plot(
            save_dir=save_dir,
            load_dir=save_dir / 'norm_coverage',
            tr_list=tr_list[i*50:],
            tr_list_names=tr_list_names[i*50:],
            fname=f'_cluster{n}_{i}',
            mode='coverage',
            ref=ref,
            smpls=smpls
        )

#################


threshold_fc = 1.5
threshold_pval = 0.1
smpls = smpls[np.array([0,6,1,7,2,8,3,9,4,10,5,11])]

for t in [60,360]:
    df = pd.read_csv(save_dir / 'cpm' / 'DESeq2' / f'time{t}min' / 'deltaTE.txt.gz',index_col=0,sep='\t')
    for tt,tt_str in zip(['>','<'],['increase','decrease']):
        tr_list = list(df.query(f'(log2FoldChange {tt}= {np.log2(threshold_fc)}) and (pvalue <= {threshold_pval})').index)
        tr_list_names = [
            ref.id.dict_name[tr]['symbol']
            for tr in tr_list
        ]
        tr_list = np.array(tr_list)
        tr_list_names = np.array(tr_list_names)
        print(f'{len(tr_list)} transcripts...')
        n_for = int(round(len(tr_list)/50,0))
        for i in range(n_for):
            Shiying.indiv_plot(
                save_dir=save_dir,
                load_dir=save_dir / 'norm_ribo_density',
                tr_list=tr_list[i*50:(i+1)*50],
                tr_list_names=tr_list_names[i*50:(i+1)*50],
                fname=f'_DESeq2_{t}min_{tt_str}_{i}',
                mode='offset_15',
                ref=ref,
                smpls=smpls
            )
        if i*50 < len(tr_list):
            i += 1
            Shiying.indiv_plot(
                save_dir=save_dir,
                load_dir=save_dir / 'norm_ribo_density',
                tr_list=tr_list[i*50:],
                tr_list_names=tr_list_names[i*50:],
                fname=f'_DESeq2_{t}min_{tt_str}_{i}',
                mode='offset_15',
                ref=ref,
                smpls=smpls
            )
