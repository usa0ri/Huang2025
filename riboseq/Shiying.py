from pathlib import Path
import pandas as pd
from pyparsing import col
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pysam

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import itertools
from matplotlib.font_manager import FontProperties
from scipy.stats import gaussian_kde, pearsonr

import myRiboSeq.old.mylib as my
import myRiboSeq.myRiboBin as mybin

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

'''for edgeR inputs (raw count of each gene)'''
def cpm(
    save_dir,
    load_dir,
    ref,
    thres_pos
):
    save_dir = save_dir / 'cpm'
    if not save_dir.exists():
        save_dir.mkdir()
    
    df_out = [];df_out2 = []
    for s in ref.exp_metadata.df_metadata['sample_name']:
        print(f'generating count data matrix in {s}...')
        obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=-1)
        counts_ = []
        if 'RPF' in s:
            len_tr = len(dict_tr)
            for i,(tr,v) in enumerate(dict_tr.items()):
                print(f'\r{i}/{len_tr} transcripts...',end='')
                cnt_pos_ = v.groupby(['dist5_start']).apply(len)
                cnt_pos = cnt_pos_[ cnt_pos_ < thres_pos ]
                if len(cnt_pos_)-len(cnt_pos) > 0:
                    print(f'{len(cnt_pos_)-len(cnt_pos)} positions filtered')
                cnt_pos = cnt_pos.iloc[
                    (cnt_pos.index >= -12) *\
                    (cnt_pos.index <= ref.annot.annot_dict[tr]['cds_len']+15) ]
                counts_.append(cnt_pos.values.sum())
            counts_ = np.array(counts_)
            # counts_ = np.array([
            #     len(v.query('cds_label == 1'))
            #     for v in dict_tr.values()
            # ])
        else:
            len_tr = len(dict_tr)
            for i,(tr,v) in enumerate(dict_tr.items()):
                print(f'\r{i}/{len_tr} transcripts...',end='')
                cnt_pos_ = v.groupby(['dist5_start']).apply(len)
                cnt_pos = cnt_pos_[ cnt_pos_ < thres_pos ]
                if len(cnt_pos_)-len(cnt_pos) > 0:
                    print(f'{len(cnt_pos_)-len(cnt_pos)} positions filtered')
                counts_.append(cnt_pos.values.sum())
            counts_ = np.array(counts_)
            # counts_ = np.array([
            #     len(v)
            #     for v in dict_tr.values()
            # ])

        tr_lists = list(dict_tr.keys())
        df_now = pd.DataFrame(counts_,index=tr_lists,columns=[s])
        df_now = df_now.iloc[ (df_now[s] > 8).values,: ]
        df_now.to_csv(
            save_dir / f'count_mat_{s}.csv.gz'
        )
        counts = np.array(counts_)[np.array(counts_) > 8]
        tr_lists = np.array(tr_lists)[ np.array(counts_) > 8 ]

        if len(df_out) == 0:
            df_out = pd.DataFrame(counts,index=tr_lists,columns=[s])
            df_out2 = pd.DataFrame(counts,index=tr_lists,columns=[s])
        else:
            df_out = pd.merge(
                df_out,
                pd.DataFrame(counts,index=tr_lists,columns=[s]),
                how="inner",left_index=True,right_index=True,
                copy=False
            )
            df_out2 = pd.merge(
                df_out2,
                pd.DataFrame(counts,index=tr_lists,columns=[s]),
                how="outer",left_index=True,right_index=True,
                copy=False
            )
    df_out.to_csv(save_dir / f'count_mat.csv.gz')
    df_out2.to_csv(save_dir / f'count_mat_outer.csv.gz')

def _common_substring(s1,s2):
    l1 = len(s1)
    l2 = len(s2)
    dp = [[0]*(l2+1) for i in range(l1+1)]
    for i in range(l1-1,-1,-1):
        for j in range(l2-1,-1,-1):
            r = max(dp[i+1][j], dp[i][j+1])
            if s1[i] == s2[j]:
                r = max(r,dp[i+1][j+1]+1)
            dp[i][j] = r
    out = []
    idx1 = []
    i=0;j=0
    while i<l1 and j<l2:
        if s1[i] == s2[j]:
            out.append(s1[i])
            idx1.append(i)
            i += 1; j += 1
        elif dp[i][j] == dp[i+1][j]:
            i += 1
        elif dp[i][j] == dp[i][j+1]:
            j += 1
    idx_ = [list(g) for _, g in itertools.groupby(idx1, key=lambda n, c=itertools.count(): n - next(c))]
    idx = idx_[np.argmax([len(x) for x in idx_])]
    return ''.join(np.array(list(s1))[np.array(idx)])

def _scatter_genes(
    rpkms,
    pair,
    threshold,
    c,
    pdf,
    texts = ''
):
    rpkms.loc[:,f'{c}_ratio'] = rpkms.loc[:,f'{c}_{pair[0]}'] / rpkms.loc[:,f'{c}_{pair[1]}']
    if threshold>0:
        rpkms.loc[:,f'{c}_label'] = rpkms.\
            apply(lambda x: 'high' if (x[f'{c}_ratio'] > threshold) \
                  else ('low' if x[f'{c}_ratio'] < 1/threshold else 'medium'), axis=1)
    else:
        rpkms.loc[:,f'{c}_label'] = 'medium'
    # rpkms.loc[:,f'{c}_label'] = rpkms.\
    #     apply(lambda x: x[f'{c}_label'] if (x[f'{c}_{pair[0]}'] > threshold_tr and x[f'{c}_{pair[1]}'] > threshold_tr) else 'medium', axis=1)
    
    dict_n_label = { l[0][0]:len(l[1]) for l in list(rpkms.groupby([f'{c}_label']))}
    colors = {'high':"#BF0D0D",'low':"#0D1BBF","medium":"#3E3E3E"}

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(
        np.log2(rpkms.query(f'{c}_label == "medium"')[f'{c}_{pair[1]}']),
        np.log2(rpkms.query(f'{c}_label == "medium"')[f'{c}_{pair[0]}']),
        color=colors['medium'],
        s=1)
    if threshold>0:
        ax.scatter(
            np.log2(rpkms.query(f'{c}_label == "high"')[f'{c}_{pair[1]}']),
            np.log2(rpkms.query(f'{c}_label == "high"')[f'{c}_{pair[0]}']),
            color=colors['high'],
            s=1)
        ax.scatter(
            np.log2(rpkms.query(f'{c}_label == "low"')[f'{c}_{pair[1]}']),
            np.log2(rpkms.query(f'{c}_label == "low"')[f'{c}_{pair[0]}']),
            color=colors['low'],
            s=1)
    ax.set_xlabel(f'{pair[1]} (log2 {c})',fontsize=20)
    ax.set_ylabel(f'{pair[0]} (log2 {c})',fontsize=20)
    max_lim = np.amax([ax.get_xlim()[1],ax.get_ylim()[1]])
    min_lim = np.amin([ax.get_xlim()[0],ax.get_ylim()[0]])
    ax.set_xticks(list(range(int(np.floor(min_lim))//2*2-2,int(np.ceil(max_lim))//2*2+2,4)))
    ax.set_yticks(list(range(int(np.floor(min_lim))//2*2-2,int(np.ceil(max_lim))//2*2+2,4)))
    ax.set_xlim([min_lim,max_lim])
    ax.set_ylim([min_lim,max_lim])

    i = 0
    for k,v in dict_n_label.items():
        ax.text(
            int(np.ceil(min_lim))+1,
            int(np.floor(max_lim)) - i*1.5,
            f'{v} ({round(v / len(rpkms)*100, 1)}%)',
            color=colors[k],
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=15)
        i += 1

    ax.tick_params(axis='both', labelsize=20)
    # ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)
    if texts != '':
        ax.text(1,max_lim-1,texts,fontsize=20)
    fig.tight_layout()
    plt.show()
    fig.savefig(pdf,format='pdf')
    plt.close()

    return rpkms
        

'''calculate RPKM (read per kilobase per million mapped reads)'''
def rpkm(
    save_dir,
    load_dir,
    ref
):
    save_dir = save_dir / 'rpkm'
    if not save_dir.exists():
        save_dir.mkdir()
    for s in ref.exp_metadata.df_metadata['sample_name']:
        print(f'calculating RPKM in {s}...')
        outfile_name = save_dir / f'rpkm_{s}.csv.gz'
        obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir
            )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=-1)
        count_sum = np.array([
            len(v)
            for v in dict_tr.values()
        ]).sum(axis=None)
        print(count_sum)
        rpkm = np.array([
            len(v) / (count_sum * ref.annot.annot_dict[v['tr_id'].iloc[0]]["cdna_len"]) * 1e+9
            for v in dict_tr.values()
        ])
        tpm_ = np.array([
            (len(v)/ref.annot.annot_dict[v['tr_id'].iloc[0]]["cdna_len"])
            for v in dict_tr.values()
        ])
        tpm = tpm_ / np.sum(tpm_,axis=None) * 1e+6
        pd.DataFrame({'rpkm':rpkm,'tpm':tpm},index=dict_tr.keys()).\
            to_csv(outfile_name,compression="gzip")

def scatter_rpkm(
    save_dir,
    load_dir,
    pairs,
    ref,
    threshold=0,
    fname = ''
):
    save_dir = save_dir / 'scatter_rpkm'
    if not save_dir.exists():
        save_dir.mkdir()
    # scatter plots
    if threshold>0:
        outfile_name = save_dir / f'scatter_rpkm{fname}_fc{threshold}.pdf'
    else:
        outfile_name = save_dir / f'scatter_rpkm{fname}.pdf'
    counts = ['rpkm','tpm']
    pdf = PdfPages(outfile_name)
    for pair in pairs:
        print(f'\ncalculating rpkm ratio {pair[0]} / {pair[1]}...')
        outfile_name = save_dir / f'rpkm_ratio_{pair[0]}_{pair[1]}'
        rpkm1 = pd.read_csv(load_dir / f'rpkm_{pair[0]}.csv.gz',header=0,index_col=0)
        rpkm2 = pd.read_csv(load_dir / f'rpkm_{pair[1]}.csv.gz',header=0,index_col=0)
        rpkms = pd.merge(rpkm1,rpkm2,
            left_index=True, right_index=True,
            how='inner',suffixes=[f'_{p}' for p in pair])
        r = pearsonr(
            rpkms[f'{counts[0]}_{pair[0]}'],
            rpkms[f'{counts[0]}_{pair[1]}'])
        rpkms = _scatter_genes(
            rpkms=rpkms,
            pair=pair,
            pdf=pdf,
            c=counts[0],
            threshold=threshold,
            texts=f'Pearson\'s r = {round(r.correlation,2)}'
        )
        r = pearsonr(
            rpkms[f'{counts[1]}_{pair[0]}'],
            rpkms[f'{counts[1]}_{pair[1]}'])
        rpkms = _scatter_genes(
            rpkms=rpkms,
            pair=pair,
            pdf=pdf,
            c=counts[1],
            threshold=threshold,
            texts=f'Pearson\'s r = {round(r.correlation,2)}'
        )
        rpkms.to_csv(outfile_name.with_suffix('.csv.gz'),compression="gzip")
        # stat
        for c in counts:
            outfile_name = save_dir / f'{c}_ratio_stat_{pair[0]}_{pair[1]}'
            rpkms_label = rpkms.query(f"{c}_ratio>0").pivot_table(values=f"{c}_ratio",index=f"{c}_label",aggfunc=len)
            rpkms_label[f"{c}_percent"] = rpkms_label[f"{c}_ratio"] / rpkms_label[f"{c}_ratio"].sum() *100
            rpkms_label.to_csv(outfile_name.with_suffix('.csv.gz'),compression="gzip")
    pdf.close()
  
def summarize_rpkm(
    save_dir,
    load_dir,
    pairs,
    ref,
    mode
):
    assert mode in ('rpkm','tpm')
    df = []
    for pair in pairs:
        df_now = pd.read_csv( load_dir / f'rpkm_ratio_{pair[0]}_{pair[1]}.csv.gz', header=0, index_col=0 )
        if len(df)>0:
            if f'{mode}_{pair[0]}' not in df.columns:
                df = pd.merge(
                    df,
                    df_now[[f'{mode}_{pair[0]}']],
                    left_index=True,right_index=True
                )
            if f'{mode}_{pair[1]}' not in df.columns:
                df = pd.merge(
                    df,
                    df_now[[f'{mode}_{pair[1]}']],
                    left_index=True,right_index=True
                )
        else:
            df = df_now[[f'{mode}_{pair[1]}',f'{mode}_{pair[0]}']]

    df['name'] = [
        ref.id.dict_name.get(tr_id,{'symbol':''})['symbol']
        for tr_id in df.index
    ]
    for pair in pairs:
        df[f'ratio {pair[0]}/{pair[1]}'] = df[f'{mode}_{pair[0]}'] / df[f'{mode}_{pair[1]}']

    df.to_csv(save_dir / f'{mode}.csv.gz')

def volcano_plot(
    save_dir,
    load_data,
    threshold_fc,
    threshold_pval,
    tr_highlight,
    ref,
    fname
):
    save_dir = save_dir / 'volcano_plot'
    if not save_dir.exists():
        save_dir.mkdir()

    if 'edgeR' in fname:
        df = pd.read_csv( load_data, header=0, index_col=0 )
        var_logfc = 'logFC'
        var_pval = 'PValue'
        xlim_now=7
        ylim_now=7
    
    elif 'DESeq2' in fname:
        df = pd.read_csv( load_data, header=0, index_col=0 ,sep='\t')
        var_logfc = 'log2FoldChange'
        var_pval = 'pvalue'
        xlim_now=4.5
        ylim_now=4

    idx_high = df.query(f'({var_logfc}>= {np.log2(threshold_fc)}) and ({var_pval} <= {threshold_pval})').index
    idx_low = df.query(f'({var_logfc} <= {np.log2(1/threshold_fc)}) and ({var_pval} <= {threshold_pval})').index
    idx_medium = np.array([x for x in df.index if (x not in idx_high) and (x not in idx_low)])

    dict_n_label = {
        'high':len(idx_high),
        'low':len(idx_low),
        'medium':len(idx_medium)
    }
    colors = {'high':"#BF0D0D",'low':"#0D1BBF","medium":"#3E3E3E"}

    outfile_name = save_dir / f'volcano_plot_{fname}.pdf'
    pdf = PdfPages(outfile_name)

    fig, ax = plt.subplots(1,1,figsize=(4,4))
    for c,g in zip(['medium','high','low'],[idx_medium,idx_high,idx_low]):
        ax.scatter(
            df.loc[g,var_logfc],
            df.loc[g,var_pval].apply(lambda x: -np.log10(x)),
            color=colors[c],
            s=1
        )
    i = 0
    for k,v in dict_n_label.items():
        if k=='high':
            x = xlim_now-1
            y = ylim_now-0.1
            ha = 'right'
        elif k=='low':
            x = -xlim_now+1
            y = ylim_now-0.1
            ha = 'left'
        elif k == 'medium':
            x = 0
            y = ylim_now-0.1
            ha = 'center'

        ax.text(
            x,y,
            f'{v} ({round(v / len(df)*100, 1)}%)',
            color=colors[k],
            horizontalalignment=ha,
            verticalalignment='top',
            fontsize=10)
        i += 1
    
    for tr in tr_highlight:
        if tr in df.index:
            ax.plot(
                [df.loc[tr,var_logfc],df.loc[tr,var_logfc]*1.05],
                [-np.log10(df.loc[tr,var_pval]),-np.log10(df.loc[tr,var_pval])*1.05],
                color="black"
            )
            ax.text(
                x=df.loc[tr,'logFC']*1.05,
                    y=-np.log10(df.loc[tr,var_pval])*1.05,
                    s=ref.id.dict_name[tr]['symbol'],
                    color='black'
                )
    # xlim_now = np.sqrt(np.max(np.array(ax.get_xlim())**2))
    ax.set_xlim(-xlim_now,xlim_now)
    ax.set_ylim(-0.1,ylim_now)
    ax.set_xlabel('Log2(FC)',fontsize=10)
    ax.set_ylabel('-log10(p value)',fontsize=10)
    fig.savefig(pdf,format='pdf')
    pdf.close()

def merge_DESeq2_edgeR(
    save_dir,
    files_deseq2,
    files_edger,
    ref
):
    save_dir = save_dir / 'volcano_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    threshold_fc = 1.5
    threshold_pval = 0.1

    df_deseq2 = {}
    for s,f in files_deseq2.items():
        df = pd.read_csv( f, header=0, index_col=0 , sep='\t')
        df_deseq2[s] = df.query(
            f'(log2FoldChange>= {np.log2(threshold_fc)} or log2FoldChange<= {np.log2(1/threshold_fc)}) and (pvalue <= {threshold_pval}) '
            )
    
    df_edger = {}
    for s,f in files_edger.items():
        df = pd.read_csv( f, header=0, index_col=0)
        df_edger[s] = df.query(
            f'(logFC>= {np.log2(threshold_fc)} or logFC<= {np.log2(1/threshold_fc)}) and (PValue <= {threshold_pval}) '
            )
    
    tr_list_common = []
    for s in files_deseq2.keys():
        n_deseq2 = len(df_deseq2[s])
        n_edger = len(df_edger[s])
        tr_list_common.append(list(set(list(df_deseq2[s].index)) & set(list(df_edger[s].index))))
        df_common = pd.merge(
            df_deseq2[s],df_edger[s],
            how='inner',left_index=True,right_index=True
        )
        df_common.to_csv(save_dir / f'common_DESeq2_edgeR_{s}.csv.gz')
    
    tr_ = list(set(tr_list_common[0]) & set(tr_list_common[1]))
    pd.DataFrame(tr_).to_csv(save_dir / 'tr_common.csv.gz')
    print("hoge")
  

def _calc_dendrogam(X,n_cluster):
    model = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward',compute_distances=True)
    model.fit_predict(X)
    # create counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i,merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 #leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return model, linkage_matrix 

def _heatmap_fc(
    dfs,
    n_cluster,
    tr_disp,
    ref,
    outfile_name
):
    tr_names = [
        ref.id.dict_name[tr]['symbol']
        if tr in ref.id.dict_name.keys() else ''
        for tr in dfs.index
    ]
    tr_lists = my._name2id(ref,tr_disp)
    df_out = dfs.copy()
    df_out['name'] = tr_names

    tr_plot = [ref.id.dict_name[tr]['symbol'] for tr in tr_lists if tr in df_out.index] 

    grps = dfs.columns

    # model_tr,linkage_tr = _calc_dendrogam(dfs.values,n_cluster)
    kmeans = KMeans(
        init='k-means++',
        n_clusters=n_cluster,
        n_init=4,
        random_state=0
    )
    estimator = make_pipeline(
        StandardScaler(),
        kmeans
    ).fit(dfs.values)
    df_out["cluster"] = estimator[-1].labels_
    df_out.sort_values("cluster").to_csv(outfile_name.with_suffix('.csv.gz'),compression="gzip")

    # plot heatmap
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    fig, axs = plt.subplots(1, 3, figsize=(6,10),gridspec_kw={'width_ratios': [10,1,1]})
    X = df_out.sort_values("cluster").drop(['cluster','name'],axis=1).apply(lambda x: np.log2(x)).values
    sns.heatmap(
        X,
        cmap="coolwarm",
        vmin=-4,vmax=4,
        cbar_kws={'label':'log2(FC)','fraction':0.4},
        cbar_ax=axs[2],
        yticklabels=False,
        xticklabels=grps,
        ax=axs[0])
    axs[0].set_xticklabels(grps,rotation=30,ha="right")
    idx_tr = [np.where(df_out.sort_values("cluster").index == tr)[0][0] for tr in tr_lists if tr in df_out.index]
    
    cmap = mpl.cm.get_cmap('Set3')
    start = 0
    for i in range(n_cluster):
        d = df_out.sort_values("cluster").query(f'cluster == {i}').index
        stop = start + len(d)
        axs[1].fill_between(
            y1=start,y2=stop,
            x=[0,1],
            color=cmap( i/n_cluster )
        )
        start = stop
    axs[1].sharey(axs[0])
    axs[1].set_xticks([])
    axs[1].set_title("Clusters")
    # axs[1].set_yticks([])
    for tmp in ['top','right','bottom','left']:
        axs[1].spines[tmp].set_visible(False)

    axs[0].set_yticks(idx_tr)
    axs[0].set_yticklabels(tr_plot)
    axs[0].yaxis.set_label_position('right')

    fig.tight_layout()
    fig.savefig(pdf,format='pdf')

    # each cluster information
    fig, axs = plt.subplots(1, n_cluster, figsize=(n_cluster*2,3))
    for i,ax in enumerate(axs):
        df_now = df_out.query(f'cluster == {i}').drop(['cluster','name'],axis=1).apply(lambda x: np.log2(x))
        mean_grps = df_now[grps].mean().values
        std_grps = df_now[grps].std().values
        if len(df_now)==1:
            std_grps = np.zeros(len(grps))
        axs[i].bar(
            x=grps,
            height=mean_grps,
            width=0.8,
            yerr=std_grps,
            capsize=n_cluster/2,
            color=cmap( i/n_cluster )
        )
        axs[i].set_xticklabels(
            axs[i].get_xticklabels(), 
            rotation=30, ha='right',fontsize=7.5)
        axs[i].set_title(f'cluster {i}\n(n={len(df_now)})',fontsize=7.5)
    fig.tight_layout()
    fig.savefig(pdf,format='pdf')

    pdf.close()
    plt.close('all')

def heatmap_timecourse(
    save_dir,
    load_dir,
    smpls,
    ref,
    n_cluster,
    tr_disp=[]
):
    save_dir = save_dir / 'heatmap_timecourse'
    if not save_dir.exists():
        save_dir.mkdir()

    dfs = []
    for s in smpls:
        df = pd.read_csv(load_dir / f'te_{s}.csv.gz', header=0, index_col=0 )
        if len(dfs)>0:
            dfs = pd.merge(
                dfs,
                pd.DataFrame(df[f'TE_tpm_{s}'].values,columns=[s],index=df.index),
                how="inner",right_index=True,left_index=True)
        else:
            dfs = pd.DataFrame(df[f'TE_tpm_{s}'].values,columns=[s],index=df.index)
    
    # for s in smpls:
    #     dfs[s] = dfs[s] / dfs[smpls[0]]
    
    outfile_name = save_dir / f'clustering_TE_{n_cluster}'
    _heatmap_fc(dfs,n_cluster,tr_disp,ref,outfile_name)

def heatmap_fc_reps(
    save_dir,
    load_dirs,
    pairs,
    ref,
    n_cluster,
    rep_names,
    tr_disp=[]
):
    save_dir = save_dir / 'heatmap_fc_reps'
    if not save_dir.exists():
        save_dir.mkdir()

    dfs_all = []
    grps_all = []
    for pair in pairs:
        dfs = []
        grps = []
        for load_dir,rep_name in zip(load_dirs,rep_names):
            df = pd.read_csv(load_dir / f'rpkm_ratio_{pair[0]}_{pair[1]}.csv.gz', header=0, index_col=0 )
            if len(dfs)>0:
                dfs = pd.merge(
                    dfs,df['rpkm_ratio'],
                    how="inner",right_index=True,left_index=True)
            else:
                dfs = df['rpkm_ratio']
            grps.append(pair[0] + '/' + pair[1] + '(' + rep_name + ')')
            grps_all.append(grps[-1])

        dfs = dfs.set_axis(labels=grps,axis=1,copy=False)
        if len(dfs_all)>0:
            dfs_all = pd.merge(
                dfs_all,dfs,
                how="inner",right_index=True,left_index=True)           
        else:
            dfs_all = dfs
        
        outfile_name = save_dir / f'clustering_RPKM_{pair[0]}_{n_cluster}'
        _heatmap_fc(save_dir,dfs,n_cluster,tr_disp,ref,outfile_name)
    
    order = np.argsort(grps_all)[::-1]
    dfs_all = dfs_all.iloc[:,order]
    outfile_name = save_dir / f'clustering_RPKM_all_{n_cluster}'
    _heatmap_fc(dfs_all,n_cluster,tr_disp,ref,outfile_name)

def te(
    save_dir,
    load_dir,
    pairs,
    pairs2,
    threshold,
    threshold_tr,
    nrep
):
    save_dir = save_dir / 'translation_efficiency'
    if not save_dir.exists():
        save_dir.mkdir()

    for pair in pairs:
        for i in range(nrep):
            if i == 0:
                df_ribo = pd.read_csv(load_dir / f'rpkm_{pair[0]}_rep{i+1}.csv.gz', header=0, index_col=0 )
            else:
                df_ribo_ = pd.read_csv(load_dir / f'rpkm_{pair[0]}_rep{i+1}.csv.gz', header=0, index_col=0 )
                df_ribo = df_ribo.add(df_ribo_,fill_value=0)
            if i == 0:
                df_rna = pd.read_csv(load_dir / f'rpkm_{pair[1]}_rep{i+1}.csv.gz', header=0, index_col=0 )
            else:
                df_rna_ = pd.read_csv(load_dir / f'rpkm_{pair[1]}_rep{i+1}.csv.gz', header=0, index_col=0 )
                df_rna = df_rna.add(df_rna_,fill_value=0)
        df_ribo /= nrep
        df_rna /= nrep
        df_ribo = df_ribo.query(f'rpkm > {threshold_tr}')
        df_rna = df_rna.query(f'rpkm > {threshold_tr}')
        pair_te = _common_substring(pair[0],pair[1])
        if '_R' in pair_te:
            pair_te = pair_te[:-2]
        if pair_te.startswith('_'):
            pair_te = pair_te[1:]
        df = pd.merge(
            df_ribo,df_rna,
            how='inner',left_index=True,right_index=True,
            suffixes=['_'+pair[0],'_'+pair[1]])
        df[f'TE_rpkm_{pair_te}'] = df[f'tpm_{pair[0]}'] / df[f'tpm_{pair[1]}']
        df[f'TE_tpm_{pair_te}'] = df[f'tpm_{pair[0]}'] / df[f'tpm_{pair[1]}']
        df.to_csv(save_dir / f'te_{pair_te}.csv.gz')

    outfile_name = save_dir / f'scatter_TE_fc{threshold}.pdf'
    pdf = PdfPages(outfile_name)
    for pair in pairs2:
        tes1 = pd.read_csv(save_dir / f'te_{pair[0]}.csv.gz',header=0,index_col=0)
        tes2 = pd.read_csv(save_dir / f'te_{pair[1]}.csv.gz',header=0,index_col=0)
        tes = pd.merge(
            tes1,tes2,
            how='inner',left_index=True,right_index=True,
            suffixes=['_'+pair[0],'_'+pair[1]]
        )
        tes = _scatter_genes(
            rpkms=tes,
            pair=pair,
            pdf=pdf,
            c='TE_tpm',
            threshold=threshold
        )
        outfile_name = save_dir / f'te_{pair[0]}_{pair[1]}.csv.gz'
        tes.to_csv(outfile_name,compression="gzip")
    pdf.close()

           
def calc_norm_coverage(
    load_dir,
    save_dir,
    threshold_tr_count,
    smpls,
    ref):

    save_dir = save_dir / f'norm_coverage'
    if not save_dir.exists():
        save_dir.mkdir()
    
    for s in smpls:
        print(f'\ncalculating normalized ribosome coverage for {s}...')

        # BAM file
        a = ref.exp_metadata.df_metadata.query(f'sample_name == "{s}"').align_files.iloc[-1]
        infile = pysam.AlignmentFile(ref.data_dir / a)

        # dict_tr
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count)
        total_counts = len(df_data)

        len_tr = len(dict_tr)
        cvgs_cds = {}
        cvgs_3utr = {}
        cvgs_5utr = {}
        for i,(tr,val) in enumerate(dict_tr.items()):
            print(f'\r{i}/{len_tr} transcripts',end='')
            if tr not in ref.annot.annot_dict.keys():
                continue
            # ACGT
            cvge_ = infile.count_coverage(contig=tr)
            cvge = np.array(cvge_[0]) + np.array(cvge_[1]) + np.array(cvge_[2]) + np.array(cvge_[3])
            cvge[ cvge > 500 ] = np.amax( cvge[cvge <= 500] )
            # cds_cnt = infile.count(
            #     contig=tr,
            #     start=ref.annot.annot_dict[tr]['start'],
            #     stop=ref.annot.annot_dict[tr]['stop']+3)
            cvge = cvge / total_counts * 1e+6
            cvge_5utr = cvge[: ref.annot.annot_dict[tr]['start'] ]
            cvge_3utr = cvge[ref.annot.annot_dict[tr]['stop']+3: ]
            cvge_cds = cvge[ ref.annot.annot_dict[tr]['start']:\
                ref.annot.annot_dict[tr]['stop']+3 ]
            cvgs_cds[tr] = cvge_cds
            cvgs_3utr[tr] = cvge_3utr
            cvgs_5utr[tr] = cvge_5utr
        
        my._mysave(save_dir / f'norm_cvgs_cds_{s}.joblib',cvgs_cds)
        my._mysave(save_dir / f'norm_cvgs_3utr_{s}.joblib',cvgs_3utr)
        my._mysave(save_dir / f'norm_cvgs_5utr_{s}.joblib',cvgs_5utr)


def indiv_plot(
    save_dir,
    load_dir,
    tr_list,
    fname,
    mode,
    ref,
    smpls
):
    save_dir = save_dir / f'indiv_plot'
    if not save_dir.exists():
        save_dir.mkdir()
    
    num_smpl = len(smpls)
    dts_cds = {};dts_5utr={};dts_3utr={}
    for p in smpls:
        if mode == 'coverage':
            dts_cds[p] = my._myload(load_dir / f'norm_cvgs_cds_{p}.joblib')
            dts_5utr[p] = my._myload(load_dir / f'norm_cvgs_5utr_{p}.joblib')
            dts_3utr[p] = my._myload(load_dir / f'norm_cvgs_3utr_{p}.joblib')
        elif 'offset' in mode:
            offset = int(mode.split('_')[-1])
            offset = int(mode.split('_')[-1])
            obj = mybin.myBinRiboNorm(
                smpl=p,
                sp=ref.sp,
                save_dir=load_dir,
                mode='start',
                read_len_list=[],
                is_length=False,
                is_norm=True,
                dirname=''
            )
            obj.decode()
            tmp = pd.DataFrame(obj.count['count5'].T.todense(),index=obj.tr['tr5'],columns=obj.pos['pos5'])
            dt = tmp.iloc[ tmp.index.map(lambda x: x in tr_list), : ]
            dt = dt.set_axis( dt.columns+offset, axis='columns', copy=False )
            dt = dt[ dt < 1000 ]
            dts_cds[p] = {};dts_5utr[p]={};dts_3utr[p]={}
            for tr in tr_list:
                if tr not in dt.index:
                    continue
                len_5utr = ref.annot.annot_dict[tr]['start']
                len_cds = ref.annot.annot_dict[tr]['cds_len']
                len_3utr = ref.annot.annot_dict[tr]['cdna_len'] - ref.annot.annot_dict[tr]['stop'] - 3
                len_tr = ref.annot.annot_dict[tr]['cdna_len']
                if len_5utr > 0:
                    cols = np.array(range(-len_5utr,0))
                    dts_5utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                        dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                        fill_value=0
                    )
                else:
                    dts_5utr[p][tr] = []

                cols = np.array(range(0,len_cds))
                dts_cds[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                    dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                    fill_value=0
                )
                # dts_5utr[tr] = dt.loc[tr, np.array(range(-len_5utr,0)) ]
                # dts_cds[tr] = dt.loc[tr, np.array(range(0,len_cds)) ]
                if len_3utr <= 0:
                    dts_3utr[p][tr] = []
                else:
                    cols = np.array(range(len_cds,len_tr-len_5utr))
                    dts_3utr[p][tr] = pd.Series(np.zeros(len(cols)),index=cols).add(
                        dt.loc[tr,:].iloc[ (dt.columns >= cols[0])*(dt.columns < cols[-1]) ],
                        fill_value=0
                    )

    outfile_name = save_dir / f'indiv_plot_{fname}'
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for ii,tr in enumerate(tr_list):
        print(f'\r{ii}/{len(tr_list)} transcripts...',end='')
        fig, ax = plt.subplots(int(num_smpl/2),2,figsize=(6,(num_smpl/2)),sharex=True,sharey=True)
        ax[-1,0].set_xlabel('Position along the transcript (nt)') 
        ax[-1,0].set_ylabel('Normalized density')
        tr_name = ref.id.dict_name[tr]['symbol']
        fig.suptitle(f'{tr_name}_{tr}')
        # seq = ref.ref_cdna[tr][
        #     ref.annot.annot_dict[tr]["start"] : ref.annot.annot_dict[tr]["stop"]+3
        # ]
        if mode == 'coverage':
            lw_tmp = 1000/ref.annot.annot_dict[tr]['cdna_len']
        elif mode == 'asite':
            lw_tmp = 1
        else:
            lw_tmp = 1
        color = "#818589"
        xs = {};ys = {};ylim_now = 0
        for iii,p in enumerate(smpls):
            x_ = iii//2
            y_ = iii%2
            if tr not in dts_5utr[p]:
                continue
            ys[p] = np.hstack((dts_5utr[p][tr],dts_cds[p][tr],dts_3utr[p][tr]))
            xs[p] = np.arange( -len(dts_5utr[p][tr]),len(dts_3utr[p][tr])+len(dts_cds[p][tr]), 1, dtype=int )
            ax[x_,y_].axvspan(0,len(dts_cds[p][tr]),color="#FFF8DC",alpha=0.5)
            if tr == 'ENST00000674920':#ATF4:
                # https://www.ncbi.nlm.nih.gov/nuccore/NM_182810.3
                # uORF1 
                ax[x_,y_].axvspan(87-len(dts_5utr[p][tr]),98-len(dts_5utr[p][tr]),color="#7393B3",alpha=0.5)
                ax[x_,y_].axvspan(186-len(dts_5utr[p][tr]),365-len(dts_5utr[p][tr]),color="#FF69B4",alpha=0.5)
            
            ax[x_,y_].vlines(xs[p],0,ys[p],colors=color,lw=lw_tmp)
            ax[x_,y_].set_title(p)
            ylim_now = np.amax([ylim_now,ax[x_,y_].get_ylim()[1]])
        
        ax[-1,0].set_ylim(ax[-1,0].get_ylim()[0],ylim_now)
                    
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close()
    pdf.close()

def filter_gene(
    save_dir,
    cpm_cnt_path,
    ref
):
    save_dir = save_dir / 'filter_genes'
    if not save_dir.exists():
        save_dir.mkdir()
    
    df_cnt_ = pd.read_csv(cpm_cnt_path,index_col=0,header=0)
    tot_cnt = df_cnt_.sum(axis=0)

    df_cnt = df_cnt_.div(tot_cnt) * 1e+6

    for t in ['RPF','mRNA']:
        df_cnt[f'inc_{t}_0-60min'] = \
            ((df_cnt[f'{t}_60min_rep1'] - df_cnt[f'{t}_0min_rep1'] > 0) *\
            (df_cnt[f'{t}_60min_rep2'] - df_cnt[f'{t}_0min_rep2'] > 0))
        df_cnt[f'inc_{t}_60-360min'] = \
            ((df_cnt[f'{t}_360min_rep1'] - df_cnt[f'{t}_60min_rep1'] > 0) *\
            (df_cnt[f'{t}_360min_rep2'] - df_cnt[f'{t}_60min_rep2'] > 0))
        df_cnt[f'dec_{t}_0-60min'] = \
            ((df_cnt[f'{t}_60min_rep1'] - df_cnt[f'{t}_0min_rep1'] < 0) *\
            (df_cnt[f'{t}_60min_rep2'] - df_cnt[f'{t}_0min_rep2'] < 0))
        df_cnt[f'dec_{t}_60-360min'] = \
            ((df_cnt[f'{t}_360min_rep1'] - df_cnt[f'{t}_60min_rep1'] < 0) *\
            (df_cnt[f'{t}_360min_rep2'] - df_cnt[f'{t}_60min_rep2'] < 0))
        
        df_cnt[f'incdec_{t}'] = ''
        df_cnt[f'incdec_{t}'].iloc[
            (df_cnt[f'inc_{t}_0-60min'] * df_cnt[f'inc_{t}_60-360min'])
        ] = 'inc_inc'
        df_cnt[f'incdec_{t}'].iloc[
            (df_cnt[f'inc_{t}_0-60min'] * df_cnt[f'dec_{t}_60-360min'])
        ] = 'inc_dec'
        df_cnt[f'incdec_{t}'].iloc[
            (df_cnt[f'dec_{t}_0-60min'] * df_cnt[f'inc_{t}_60-360min'])
        ] = 'dec_inc'
        df_cnt[f'incdec_{t}'].iloc[
            (df_cnt[f'dec_{t}_0-60min'] * df_cnt[f'dec_{t}_60-360min'])
        ] = 'dec_dec'

        df_cnt_filter = df_cnt.query( f'incdec_{t} != ""' )
    
        df_cnt_.loc[ list(df_cnt_filter.index),: ].to_csv(save_dir / f'tmp_{t}.csv.gz')
        df_cnt_filter.to_csv(save_dir / f'tmp_norm_{t}.csv.gz')


def output(
    save_dir,
    ref
):
    save_dir = save_dir / 'output'
    if not save_dir.exists():
        save_dir.mkdir()

    # normalized CDS counts of each gene by total counts per sample
    df_cnt_filter_RPF = pd.read_csv(
        save_dir.parent / 'filter_genes' / 'tmp_norm_RPF.csv.gz',
        index_col=0,header=0)
    # normalized counts of each gene by total counts per sample
    df_cnt = pd.read_csv(
        save_dir.parent / 'cpm' / 'count_mat.csv.gz',
        index_col=0,header=0)
    tot_cnt = df_cnt.sum(axis=0)

    df_cnt = df_cnt.div(tot_cnt) * 1e+6
    
    df_cnt_filter_RPF['TE_0min_rep1'] = df_cnt_filter_RPF['RPF_0min_rep1'].div(df_cnt['mRNA_0min_rep1'])
    df_cnt_filter_RPF['TE_0min_rep2'] = df_cnt_filter_RPF['RPF_0min_rep2'].div(df_cnt['mRNA_0min_rep2'])
    df_cnt_filter_RPF['TE_60min_rep1'] = df_cnt_filter_RPF['RPF_60min_rep1'].div(df_cnt['mRNA_60min_rep1'])
    df_cnt_filter_RPF['TE_60min_rep2'] = df_cnt_filter_RPF['RPF_60min_rep2'].div(df_cnt['mRNA_60min_rep2'])
    df_cnt_filter_RPF['TE_360min_rep1'] = df_cnt_filter_RPF['RPF_360min_rep1'].div(df_cnt['mRNA_360min_rep1'])
    df_cnt_filter_RPF['TE_360min_rep2'] = df_cnt_filter_RPF['RPF_360min_rep2'].div(df_cnt['mRNA_360min_rep2'])
    
    df_cnt_filter_RPF['TE_ratio_60/0min_rep1'] = df_cnt_filter_RPF['TE_60min_rep1'] /df_cnt_filter_RPF['TE_0min_rep1'] 
    df_cnt_filter_RPF['TE_ratio_60/0min_rep2'] = df_cnt_filter_RPF['TE_60min_rep2'] /df_cnt_filter_RPF['TE_0min_rep2'] 
    df_cnt_filter_RPF['TE_ratio_360/60min_rep1'] = df_cnt_filter_RPF['TE_360min_rep1'] /df_cnt_filter_RPF['TE_60min_rep1']
    df_cnt_filter_RPF['TE_ratio_360/60min_rep2'] = df_cnt_filter_RPF['TE_360min_rep2'] /df_cnt_filter_RPF['TE_60min_rep2']
    df_cnt_filter_RPF['TE_ratio_360/0min_rep1'] = df_cnt_filter_RPF['TE_360min_rep1'] /df_cnt_filter_RPF['TE_0min_rep1']
    df_cnt_filter_RPF['TE_ratio_360/0min_rep2'] = df_cnt_filter_RPF['TE_360min_rep2'] /df_cnt_filter_RPF['TE_0min_rep2']
    

    df_cnt_filter_RPF[[c for c in df_cnt_filter_RPF.columns if ('TE' in c) or ('RPF' in c) or ('mRNA' in c) ]]\
        .to_csv(save_dir / 'ratio_RPF_mRNA.csv.gz')

    print("hoge")