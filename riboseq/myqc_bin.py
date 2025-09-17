
from pathlib import Path
import pandas as pd
from pyparsing import col
import pysam
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import re
import collections
from scipy import fftpack
from matplotlib.font_manager import FontProperties
import time
from Bio import SeqIO
import gzip
from tqdm import tqdm
from scipy.spatial import distance

import myRiboSeq.mylib_bin as my
import myRiboSeq.myRiboBin as mybin
import myRiboSeq.myprep_bin as myprep

# thresholds for total counts of transcripts
thresholds = [np.inf,64,32,16,8,0]
color_frame = my.color_frame
color_region = my.color_region
codon_table = my.codon2aa_table
color_atcg = my.color_atcg

font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

'''read length distribution in each region (5'UTR, CDS, 3'UTR), cumulative for all transcripts'''
def plot_region_len(
    load_dir,
    save_dir,
    smpls,
    ref,
    attrs,
    full_align=False,
    suffix=''):

    save_dir = save_dir / 'region_len'
    if not save_dir.exists():
        save_dir.mkdir()
    
    pdfs = {}
    for attr in attrs:
        if attr == 'read_length':
            outfile_name = save_dir / f'read_length{suffix}'
        elif attr == 'length':
            outfile_name = save_dir / f'aligned_length{suffix}'
        pdfs[attr] = PdfPages(outfile_name.with_suffix(".pdf"))
    # for s in ref.exp_metadata.df_metadata['sample_name']:
    for s in smpls:
        print(f'plotting region-specific read length distribution for {s}...')
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=-1,is_seq=False,is_cds=True)
        if full_align:
            idx_full = df_data['length'] == df_data['read_length']
            df_data = df_data.iloc[ idx_full.values, : ]
            
        for attr in attrs:

            if (attr == 'read_aligned_length'):
                df_data['read_aligned_length'] = df_data['cut3']-df_data['cut5']
        
            UTR_CDS_pivot = df_data[[attr, "cds_label"]]\
                .pivot_table(index=attr,columns="cds_label",aggfunc=len,fill_value=0)
            outfile_name_data = save_dir / f'{attr}_{s}'
            UTR_CDS_pivot.to_csv(outfile_name_data.with_suffix('.csv.gz'),compression="gzip")

            if np.sum(UTR_CDS_pivot.index < 85)>0:
                UTR_CDS_pivot = UTR_CDS_pivot.iloc[ UTR_CDS_pivot.index < 85 ,: ]
            fig, axes = plt.subplots(nrows=2, ncols=1)
            len_tmp = list(range(UTR_CDS_pivot.index[0],np.max(UTR_CDS_pivot.index)+1))
            df_tmp1 = pd.Series(np.zeros(len(len_tmp)),index=len_tmp)\
                .add(UTR_CDS_pivot.sum(axis=1),fill_value=0)
            df_tmp1.plot(use_index=True,ax=axes[0],rot=90)
            df_tmp = pd.merge(
                UTR_CDS_pivot.div(UTR_CDS_pivot.sum(axis=1), axis=0),
                pd.DataFrame(np.zeros(len(len_tmp)),index=len_tmp),
                how="outer",left_index=True,right_index=True
            )
            # axes[1].plot(df_tmp1.index,df_tmp1.values)
            df_tmp.plot.bar(
                y=['5UTR','CDS','3UTR'],
                stacked=True,
                use_index=True,
                cmap=ListedColormap([color_region[key] for key in ['5UTR', 'CDS', '3UTR']]),
                ax=axes[1])
            xtick_now = (len_tmp-np.min(len_tmp))
            axes[0].set_xticks(np.array(len_tmp)[xtick_now[::2]])
            axes[0].set_xticklabels(np.array(len_tmp)[xtick_now[::2]])
            axes[1].set_xticks(xtick_now[::2])
            axes[1].set_xticklabels(np.array(len_tmp)[xtick_now[::2]])
            axes[0].set_xlim(len_tmp[0]-1,len_tmp[-1]+1)
            axes[1].set_xlim(xtick_now[0]-1,xtick_now[-1]+1)
            axes[0].set_ylabel("Count",fontsize=10)
            axes[0].set_xlabel("Length",fontsize=10)
            axes[1].set_ylabel("Fraction",fontsize=10)
            axes[1].set_xlabel("Length",fontsize=10)

            fig.suptitle(f'{attr} ({s})')
            fig.tight_layout()
            # plt.show()
            fig.savefig(pdfs[attr], format='pdf')
            plt.close('all')

    for attr in attrs:
        pdfs[attr].close()


def plot_read_len(
    load_dir,
    save_dir,
    ref,
    attrs,
    full_align=False,
    suffix=''):

    save_dir = save_dir / 'read_len'
    if not save_dir.exists():
        save_dir.mkdir()
    
    pdfs = {}
    for attr in attrs:
        if attr == 'read_length':
            outfile_name = save_dir / f'read_length{suffix}'
        elif attr == 'length':
            outfile_name = save_dir / f'aligned_length{suffix}'
        pdfs[attr] = PdfPages(outfile_name.with_suffix(".pdf"))
    
    for s in ref.exp_metadata.df_metadata['sample_name']:
        print(f'plotting region-specific read length distribution for {s}...')
        obj = mybin.myBinRiboNC(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir,
            biotype=ref.biotype
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(is_seq=False)
        if full_align:
            idx_full = df_data['length'] == df_data['read_length']
            df_data = df_data.iloc[ idx_full.values, : ]
            
        for attr in attrs:

            if (attr == 'read_aligned_length'):
                df_data['read_aligned_length'] = df_data['cut3']-df_data['cut5']
            
            df_plot = df_data[attr].value_counts().sort_index()
        
            fig, ax = plt.subplots(1,1,figsize=(4,2))
            df_plot.plot(use_index=True,ax=ax)
            
            ax.set_ylabel("Count")
            ax.set_xlabel("Length")

            fig.suptitle(f'{attr} ({s})')
            fig.tight_layout()
            fig.savefig(pdfs[attr], format='pdf')
            plt.close('all')

    for attr in attrs:
        pdfs[attr].close()


def _calc_codon(seq,pos_start,pos_stop):
    count_codons = collections.Counter(re.split('(...)',seq[pos_start:pos_stop+3])[1::2])
    return dict(count_codons)

def _agg_ribo(
    load_dir,
    s,
    xlim_now,
    threshold_tr_count,
    threshold_pos_count,
    read_len_list,
    ref,
    is_norm=True
    ):
    xs = {}
    ys = {}
    for mode in ('start','stop'):
        obj = mybin.myBinRiboNorm(
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir,
            mode=mode,
            read_len_list=read_len_list,
            is_length=False,
            is_norm=is_norm,
            dirname=''
        )
        obj.decode()
        # filter transcripts > threshold_tr_count
        if type(threshold_tr_count) is int:
            idx_columns = np.array(obj.count['count5'].sum(axis=0)[0] > threshold_tr_count).flatten()
        elif type(threshold_tr_count) is dict:
            tr_list = threshold_tr_count['tr_list']
            idx_columns = [np.where(np.array(obj.tr['tr5']) == tr)[0][0] for tr in tr_list]
        count = obj.count['count5'].tocsc()[:,np.array(idx_columns)].tolil()
        num_nonzero = count.nnz
        # filter positions < threshold_pos_count
        count[ count > threshold_pos_count ] = 0
        print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')

        # FIXME
        idx_ = np.where(
            (np.array(obj.pos['pos5'])>=xlim_now[mode][0]) * (np.array(obj.pos['pos5'])<=xlim_now[mode][1]))[0]
        x = np.array(obj.pos['pos5'])[idx_]
        if is_norm:
            y = np.array(count.mean(axis=1)).flatten()[idx_]
        else:
            y = np.array(count.sum(axis=1)).flatten()[idx_]

        # idx_pos = [np.where(np.array(dt['pos5']) == x)[0][0] for x in xlim_now[base]]
        # x = np.array(list(range(xlim_now[base][0],xlim_now[base][1])))
        # y = np.array(count.mean(axis=1)).flatten()[idx_pos[0]:idx_pos[1]]
        xs[mode] = x
        ys[mode] = y
    return xs,ys

'''aggregation plot (metagene plot)
    relative distance from start/stop site (x axis) v.s. mean read density (y axis)
'''
def aggregation_ribo(
    save_dir,
    load_dir,
    plot_range5,
    plot_range3,
    threshold_tr_count,
    threshold_pos_count,
    col_list,
    ref,
    fname=[],
    read_len_list = [],
    is_norm=True
    ):

    if type(fname) is str:
        save_dir = save_dir / f'aggregation_ribo_{fname}'
    else:
        save_dir = save_dir / 'aggregation_ribo'

    if not save_dir.exists():
        save_dir.mkdir()
    save_dir = Path(save_dir)
    if len(read_len_list)>0:
        outfile_name = save_dir / (
            'aggregation_ribo_'
            f'5pos{plot_range5[0]}-{plot_range5[1]}_'
            f'3pos{plot_range3[0]}-{plot_range3[1]}_'
            f'tr{threshold_tr_count}_pos{threshold_pos_count}_'
            f'readlen{read_len_list[0]}-{read_len_list[1]}'
        )
    else:
        outfile_name = save_dir / (
            'aggregation_ribo_'
            f'5pos{plot_range5[0]}-{plot_range5[1]}_'
            f'3pos{plot_range3[0]}-{plot_range3[1]}_'
            f'tr{threshold_tr_count}_pos{threshold_pos_count}'
        )
    xlim_now = {'start':plot_range5,'stop':plot_range3}

    df_agg = {}
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in col_list.keys():
        print(f'\naggregation plots for {s}...')
        
        xs,ys = _agg_ribo(
            load_dir,
            s,
            xlim_now,
            threshold_tr_count,
            threshold_pos_count,
            read_len_list,
            ref,
            is_norm)

        fig,axs = plt.subplots(2,2,figsize=(9,6))
        for i,base in enumerate(["start","stop"]):
            axs[0,i].plot(xs[base],ys[base],color="#71797E")
            axs[0,i].set_xlabel(f"Distance from {base} codon (nt)")
            axs[0,i].set_ylabel("Mean read density")
            axs[0,i].set_xlim(xlim_now[base][0],xlim_now[base][1])
        # frame
        frames = [''] * 3
        for i in range(3):
            for j,base in enumerate(["start","stop"]):
                idx = (xs[base] % 3 == i)
                frames[i], = axs[1,j].plot(
                    xs[base][idx],
                    ys[base][idx],
                    color=color_frame[f'frame{i}'],
                    label=f'frame{i}')
                axs[1,j].set_xlabel(f"Distance from {base} codon (nt)")
                if is_norm:
                    axs[1,j].set_ylabel("Mean read density")
                else:
                    axs[1,j].set_ylabel("Read counts")
            axs[1,j].set_xlim(xlim_now[base][0],xlim_now[base][1])
        axs[1,1].legend(handles=[frames[0],frames[1],frames[2]],loc='upper right')
        # match y axis
        ylim_max = np.max([axs[0,0].get_ylim()[1],axs[0,1].get_ylim()[1]])
        axs[0,0].set_ylim(-ylim_max*0.05,ylim_max)
        axs[0,1].set_ylim(-ylim_max*0.05,ylim_max)
        axs[1,0].set_ylim(-ylim_max*0.05,ylim_max)
        axs[1,1].set_ylim(-ylim_max*0.05,ylim_max)

        fig.suptitle(f'{s}')
        fig.tight_layout()
        # plt.show()
        # fig.savefig(outfile_name.with_suffix('.svg'))
        fig.savefig(pdf, format='pdf')
        plt.close('all')

        df_agg[s] = [xs,ys]
    
    # subplots for all samples
    fig,axs = plt.subplots(2,len(df_agg),figsize=(len(df_agg)*3,6),sharex=True,sharey=True)
    for i,base in enumerate(["start","stop"]):
        for j,s in enumerate(col_list.keys()):
            for k in range(3):
                idx = (df_agg[s][0][base] % 3 == k)
                if len(df_agg)==1:
                    axs[i].plot(
                        df_agg[s][0][base][idx],
                        df_agg[s][1][base][idx],
                        color=color_frame[f'frame{k}'],
                        label=f'frame{i}')
                else:
                    axs[i,j].plot(
                        df_agg[s][0][base][idx],
                        df_agg[s][1][base][idx],
                        color=color_frame[f'frame{k}'],
                        label=f'frame{i}')
            if len(df_agg)==1:
                axs[i].set_xlabel(f"Distance from {base} codon (nt)")
                axs[i].set_ylabel('Normalized read density')
                axs[i].set_title(s)
            else:
                axs[i,j].set_xlabel(f"Distance from {base} codon (nt)")
                axs[i,j].set_ylabel('Normalized read density')
                axs[i,j].set_title(s)
    fig.tight_layout(rect=[0, 0, .9, 1])
    if len(df_agg)==1:
        axs[i].legend(labels=[f'frame{x}' for x in range(3)],loc='upper left',bbox_to_anchor=(1,1))
    else:
        axs[i,j].legend(labels=[f'frame{x}' for x in range(3)],loc='upper left',bbox_to_anchor=(1,1))
    # plt.show()
    fig.savefig(pdf, format='pdf')
    plt.close('all')

    
    pdf.close()

def aggregation_ribo_light(
    save_dir,
    load_dir,
    plot_range_start,
    plot_range_stop,
    threshold_tr_count,
    threshold_pos_count,
    smpls,
    ref,
    fname=[],
    read_len_list = [],
    is_norm=True,
    col_list = {}
    ):

    if len(fname)>0:
        save_dir = save_dir / f'aggregation_ribo_{fname}'
    else:
        save_dir = save_dir / 'aggregation_ribo'

    if not save_dir.exists():
        save_dir.mkdir()
    save_dir = Path(save_dir)
    if len(read_len_list)>0:
        outfile_name = save_dir / (
            'aggregation_ribo_light_'
            f'start{plot_range_start[0]}-{plot_range_start[1]}_'
            f'stop{plot_range_stop[0]}-{plot_range_stop[1]}_'
            f'tr{threshold_tr_count}_pos{threshold_pos_count}_'
            f'readlen{read_len_list[0]}-{read_len_list[1]}'
        )
    else:
        outfile_name = save_dir / (
            'aggregation_ribo_light_'
            f'start{plot_range_start[0]}-{plot_range_start[1]}_'
            f'stop{plot_range_stop[0]}-{plot_range_stop[1]}_'
            f'tr{threshold_tr_count}_pos{threshold_pos_count}'
        )
    
    xlim_now = {'start':plot_range_start,'stop':plot_range_stop}
    dict_xs = {};dict_ys = {};dict_cnt = {}
    for s in smpls:
        dict_xs[s] = {};dict_ys[s] = {}
        dict_cnt[s] = 0
        for base in ['start','stop']:
            dict_xs[s][base] = np.arange(xlim_now[base][0],xlim_now[base][1],dtype=int)
            dict_ys[s][base] = np.zeros(xlim_now[base][1]-xlim_now[base][0],dtype=float)
    offset = 12

    for s in smpls:
        print(f'\naggregation plots for {s}...')
        
        obj = mybin.myBinRibo(
            data_dir=load_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=-1,is_seq=False,is_cds=False,is_frame=False)

        # normalized ribosome density
        for tr,reads in dict_tr.items():
            reads_orf = reads[ 
                (reads['dist5_start'] >= -offset) * (reads['dist5_stop'] < -offset)
            ][['dist5_start','dist5_stop']]
            n_reads_orf = len(reads_orf)
            if n_reads_orf < threshold_tr_count:
                continue
            start_orf = (reads['cut5'] - reads['dist5_start']).iloc[0]
            end_orf = (reads['cut5'] - reads['dist5_stop']).iloc[0]
            len_orf = end_orf - start_orf
            if len(read_len_list)>0:
                reads = reads[ 
                    (reads['read_length'] >= read_len_list[0]) * (reads['read_length'] <= read_len_list[1])
                ]
            reads_start = (reads['dist5_start'].value_counts() / (n_reads_orf/len_orf)).to_dict()
            for k,v in reads_start.items():
                if v > threshold_pos_count:
                    continue
                if (k >= plot_range_start[0]) * (k < plot_range_start[1]):
                    dict_ys[s]['start'][ k - plot_range_start[0] ] += v
            reads_stop = ((reads['cut5'] - end_orf).value_counts() / (n_reads_orf/len_orf)).to_dict()
            for k,v in reads_stop.items():
                if v > threshold_pos_count:
                    continue
                if (k >= plot_range_stop[0]) * (k < plot_range_stop[1]):
                    dict_ys[s]['stop'][ k - plot_range_stop[0] ] += v
            dict_cnt[s] += 1
        dict_ys[s]['start'] /= dict_cnt[s]
        dict_ys[s]['stop'] /= dict_cnt[s]

    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in smpls:
        xs = dict_xs[s]
        ys = dict_ys[s]
        fig,axs = plt.subplots(2,2,figsize=(9,6))
        for i,base in enumerate(["start","stop"]):
            axs[0,i].plot(xs[base],ys[base],color="#71797E")
            axs[0,i].set_xlabel(f"Distance from {base} codon (nt)")
            axs[0,i].set_ylabel("Mean read density")
            axs[0,i].set_xlim(xlim_now[base][0],xlim_now[base][1])
        # frame
        frames = [''] * 3
        for i in range(3):
            for j,base in enumerate(["start","stop"]):
                idx = (xs[base] % 3 == i)
                frames[i], = axs[1,j].plot(
                    xs[base][idx],
                    ys[base][idx],
                    color=color_frame[f'frame{i}'],
                    label=f'frame{i}')
                axs[1,j].set_xlabel(f"Distance from {base} codon (nt)")
                if is_norm:
                    axs[1,j].set_ylabel("Mean read density")
                else:
                    axs[1,j].set_ylabel("Read counts")
            axs[1,j].set_xlim(xlim_now[base][0],xlim_now[base][1])
        axs[1,1].legend(handles=[frames[0],frames[1],frames[2]],loc='upper right')
        # match y axis
        ylim_max = np.max([axs[0,0].get_ylim()[1],axs[0,1].get_ylim()[1]])
        axs[0,0].set_ylim(-ylim_max*0.05,ylim_max)
        axs[0,1].set_ylim(-ylim_max*0.05,ylim_max)
        axs[1,0].set_ylim(-ylim_max*0.05,ylim_max)
        axs[1,1].set_ylim(-ylim_max*0.05,ylim_max)

        fig.suptitle(f'{s}')
        fig.tight_layout()
        # plt.show()
        # fig.savefig(outfile_name.with_suffix('.svg'))
        fig.savefig(pdf, format='pdf')
        plt.close('all')
    
    # subplots for all samples
    fig,axs = plt.subplots(2,len(smpls),figsize=(len(smpls)*3,6),sharex=True,sharey=True)
    for i,base in enumerate(["start","stop"]):
        for j,s in enumerate(smpls):
            for k in range(3):
                idx = (dict_xs[s][base] % 3 == k)
                if len(smpls)==1:
                    axs[i].plot(
                        dict_xs[s][base][idx],
                        dict_ys[s][base][idx],
                        color=color_frame[f'frame{k}'],
                        label=f'frame{i}')
                else:
                    axs[i,j].plot(
                        dict_xs[s][base][idx],
                        dict_ys[s][base][idx],
                        color=color_frame[f'frame{k}'],
                        label=f'frame{i}')
            if len(smpls)==1:
                axs[i].set_xlabel(f"Distance from {base} codon (nt)")
                axs[i].set_ylabel('Normalized read density')
                axs[i].set_title(s)
            else:
                axs[i,j].set_xlabel(f"Distance from {base} codon (nt)")
                axs[i,j].set_ylabel('Normalized read density')
                axs[i,j].set_title(s)
    fig.tight_layout(rect=[0, 0, .9, 1])
    if len(smpls)==1:
        axs[i].legend(labels=[f'frame{x}' for x in range(3)],loc='upper left',bbox_to_anchor=(1,1))
    else:
        axs[i,j].legend(labels=[f'frame{x}' for x in range(3)],loc='upper left',bbox_to_anchor=(1,1))
    # plt.show()
    fig.savefig(pdf, format='pdf')
    plt.close('all')

    # single plot for all samples
    if len(col_list) == 0:
        col_list = dict(zip(smpls,['#808080']*len(smpls)))
    fig,axs = plt.subplots(1,2,figsize=(6,4),sharey=True)
    for i,base in enumerate(["start","stop"]):
        for s in smpls:
            axs[i].plot(
                dict_xs[s][base],
                dict_ys[s][base],
                color=col_list[s],
                label=s)
        axs[i].set_ylabel('Normalized read density')  
    fig.tight_layout(rect=[0, 0, .9, 1])
    axs[i].legend(labels=smpls,loc='upper left',bbox_to_anchor=(1,1))

    fig.savefig(pdf, format='pdf')
    plt.close('all')

    pdf.close()
    print("hoge")

def aggregation_plot_length(
    save_dir,
    load_dir,
    plot_range5,
    plot_range3,
    threshold_pos_count,
    smpls,
    ref,
    mode:str,
    read_len_list=[],
    fname=[]
):
    if type(fname) is str:
        save_dir = save_dir / f'aggregation_ribo_length_{fname}'
    else:
        save_dir = save_dir / 'aggregation_ribo_length'
    if not save_dir.exists():
        save_dir.mkdir()
    
    xlim_now = {'start':plot_range5,'stop':plot_range3}

    for end in [5,3]:
        if len(read_len_list)==2:
            outfile_name = save_dir / (
                f'aggregation{end}_ribo_length_'
                f'{read_len_list[0]}to{read_len_list[1]}_'
                f'5pos{plot_range5[0]}-{plot_range5[1]}_'
                f'3pos{plot_range3[0]}-{plot_range3[1]}_'
                f'_pos{threshold_pos_count}_{mode}'
            )
        else:
            outfile_name = save_dir / (
                f'aggregation{end}_ribo_length_'
                f'5pos{plot_range5[0]}-{plot_range5[1]}_'
                f'3pos{plot_range3[0]}-{plot_range3[1]}_'
                f'_pos{threshold_pos_count}_{mode}'
            )
        
        pdf = PdfPages(outfile_name.with_suffix(".pdf"))
        for s in smpls:
            print(f'\naggregation plots for {s}...')
            for base in ('start','stop'):
                obj = mybin.myBinRiboNorm(
                    smpl=s,
                    sp=ref.sp,
                    save_dir=load_dir,
                    mode=base,
                    read_len_list=[],
                    is_length=True,
                    is_norm=True,
                    dirname=''
                )
                dts = obj.decode_readlen()
                
                if len(read_len_list)==2:
                    readlen_list = list(range(read_len_list[0],read_len_list[1]))
                else:
                    readlen_list = list(dts.keys())
                X = np.zeros((len(readlen_list),(xlim_now[base][1]-xlim_now[base][0])))
                for i,readlen in enumerate(readlen_list):
                    dt = dts.get(readlen)
                    if len(dt[f'tr{end}'] ) == 0:
                        continue                
                    
                    tr_list = dt[f'tr{end}']
                    idx_columns = [np.where(np.array(dt[f'tr{end}']) == tr)[0][0] for tr in tr_list if tr in dt[f'tr{end}']]
                
                    count = dt[f'count{end}'].tocsc()[:,np.array(idx_columns)].tolil()
                    num_nonzero = count.nnz
                    # filter positions < threshold_pos_count
                    count[ count > threshold_pos_count ] = 0
                    print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')

                    # FIXME
                    idx_pos = (np.array(dt[f'pos{end}']) >= xlim_now[base][0]) * (np.array(dt[f'pos{end}']) < xlim_now[base][1])
                    y = np.array(count.mean(axis=1)).flatten()[idx_pos]
                    idx_pos_ = np.array([x in dt[f'pos{end}'] for x in range(xlim_now[base][0],xlim_now[base][1])])
                    X[ i, idx_pos_ ] = y
            
                xlabel_now = np.arange((xlim_now[base][0]//3*3),(xlim_now[base][1]//3*3)-1,3)
                x = list(range(xlim_now[base][0],xlim_now[base][1]))
                idx_x = xlabel_now - xlim_now[base][0]
                idx_y = np.linspace(0,len(readlen_list)-1,10,dtype=int)
                if mode == 'heatmap':
                    size_factor = len(idx_pos_)/100
                    size_factor2 = len(readlen_list)/30
                    fig,axs = plt.subplots(1,1,figsize=(12*size_factor,8*size_factor2))
                    X += 1
                    for i in range(np.shape(X)[0]):
                        for j in range(np.shape(X)[1]):
                            if X[i,j]>0:
                                X[i,j] = np.log2(X[i,j])
                    axs = sns.heatmap(
                        X,
                        ax=axs,
                        linewidth=0,
                        cmap=sns.color_palette("light:b", as_cmap=True),
                        cbar=False,
                        xticklabels=True)
                    axs.set_yticks(idx_y+0.5)
                    axs.set_yticklabels(np.array(readlen_list)[idx_y])
                    axs.set_xticks(idx_x+0.5)
                    axs.set_xticklabels(xlabel_now,rotation=30)
                    axs.set_xlabel(f'Position of {end} end from {base} codon (nt)')
                    axs.set_ylabel('Read length')
                elif mode == 'line':
                    size_factor = len(idx_pos_)/100
                    size_factor2 = len(readlen_list)/30
                    fig,axs = plt.subplots(len(readlen_list),1,figsize=(9*size_factor,9*size_factor2),sharex=True)
                    X += 1
                    for i in range(np.shape(X)[0]):
                        for j in range(np.shape(X)[1]):
                            if X[i,j]>0:
                                X[i,j] = np.log2(X[i,j])
                    for i,readlen in enumerate(readlen_list):
                        axs[i].plot(
                            x,X[i,:],
                            color="#808080"
                        )
                        axs[i].set_xticks(xlabel_now)
                        axs[i].set_xticklabels(xlabel_now)
                        axs[i].set_ylabel(readlen,rotation=90)
                        axs[i].set_yticks([])
                    axs[i].set_xticklabels(xlabel_now,rotation=30)
                    axs[i].set_xlabel(f'Position of {end} end from {base} codon (nt)')
                    for tmp in ['right','left','top','bottom']:
                        for ax in axs:
                            ax.spines[tmp].set_visible(False)

                fig.suptitle(f'{s}')
                fig.tight_layout()
                # plt.show()
                # fig.savefig(outfile_name.with_suffix('.svg'))
                fig.savefig(pdf, format='pdf')
                plt.close('all')
        pdf.close()

def aggregation_tr(
    save_dir,
    load_dir,
    plot_range5,
    plot_range3,
    tr_list,
    threshold_pos_count,
    col_list
):
    save_dir = save_dir / 'aggregation_tr'
    if not save_dir.exists():
        save_dir.mkdir()

    tr_list_name = tr_list['name']
    outfile_name = save_dir / (
        'aggregation_ribo_'
        f'5pos{plot_range5[0]}-{plot_range5[1]}_'
        f'3pos{plot_range3[0]}-{plot_range3[1]}_'
        f'{tr_list_name}_pos{threshold_pos_count}'
    )
    xlim_now = {'start':plot_range5,'stop':plot_range3}

    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in col_list.keys():
        print(f'\naggregation plots for {s}...')

        xs,ys = _agg_ribo(
            load_dir,
            s,
            xlim_now,
            tr_list,
            threshold_pos_count)
        
        fig,axs = plt.subplots(2,2,figsize=(9,6))
        for i,base in enumerate(["start","stop"]):
            axs[0,i].plot(xs[base],ys[base],color="#71797E")
            axs[0,i].set_xlabel(f"Distance from {base} codon (nt)")
            axs[0,i].set_ylabel("Mean read density")
            axs[0,i].set_xlim(xlim_now[base][0],xlim_now[base][1])
        # frame
        frames = [''] * 3
        for i in range(3):
            for j,base in enumerate(["start","stop"]):
                idx = (xs[base] % 3 == i)
                frames[i], = axs[1,j].plot(
                    xs[base][idx],
                    ys[base][idx],
                    color=color_frame[f'frame{i}'],
                    label=f'frame{i}')
                axs[1,j].set_xlabel(f"Distance from {base} codon (nt)")
                axs[1,j].set_ylabel("Mean read density")
            axs[1,j].set_xlim(xlim_now[base][0],xlim_now[base][1])
        axs[1,1].legend(handles=[frames[0],frames[1],frames[2]],loc='upper right')
        # match y axis
        ylim_max = np.max([axs[0,0].get_ylim()[1],axs[0,1].get_ylim()[1]])
        axs[0,0].set_ylim(0,ylim_max)
        axs[0,1].set_ylim(0,ylim_max)
        axs[1,0].set_ylim(0,ylim_max)
        axs[1,1].set_ylim(0,ylim_max)

        fig.suptitle(f'{s}')
        fig.tight_layout()
        # plt.show()
        # fig.savefig(outfile_name.with_suffix('.svg'))
        fig.savefig(pdf, format='pdf')
        plt.close('all')
    pdf.close()
    
def aggregation_tr_length(
    save_dir,
    load_dir,
    plot_range5,
    plot_range3,
    tr_list,
    threshold_pos_count,
    col_list,
    ref,
    read_lengths=[]
):
    save_dir = save_dir / 'aggregation_tr_length'
    if not save_dir.exists():
        save_dir.mkdir()
    xlim_now = {'start':plot_range5,'stop':plot_range3}
    grp_name = tr_list['name']

    for end in [3,5]:
        if len(read_lengths)==2:
            outfile_name = save_dir / (
                f'aggregation{end}_ribo_length_'
                f'{read_lengths[0]}to{read_lengths[1]}_'
                f'5pos{plot_range5[0]}-{plot_range5[1]}_'
                f'3pos{plot_range3[0]}-{plot_range3[1]}_'
                f'{grp_name}_pos{threshold_pos_count}'
            )
        else:
            outfile_name = save_dir / (
                f'aggregation{end}_ribo_length_'
                f'5pos{plot_range5[0]}-{plot_range5[1]}_'
                f'3pos{plot_range3[0]}-{plot_range3[1]}_'
                f'{grp_name}_pos{threshold_pos_count}'
            )
        pdf = PdfPages(outfile_name.with_suffix(".pdf"))
        for s in col_list.keys():
            print(f'\naggregation plots for {s}...')
            for base in ('start','stop'):
                obj = mybin.myBinRiboNorm(
                    smpl=s,
                    sp=ref.sp,
                    save_dir=load_dir,
                    mode=end,
                    read_len_list=[],
                    is_length=True,
                    is_norm=True,
                    dirname=''
                )
                dts = obj.decode()
                # dts = my._myload(load_dir / f'df_norm_density_{base}_readlen_{s}.joblib')
                X = np.zeros((len(dts.keys()),(xlim_now[base][1]-xlim_now[base][0])))
                if len(read_lengths)==2:
                    readlen_list = list(range(read_lengths[0],read_lengths[1]))
                else:
                    readlen_list = list(dts.keys())
                for i,readlen in enumerate(readlen_list):
                    dt = dts.get(readlen)
                    if len(dt[f'tr{end}'] ) == 0:
                        continue                
                    # filter transcripts > threshold_tr_count
                    idx_columns = [np.where(np.array(dt[f'tr{end}']) == tr)[0][0] for tr in tr_list['tr_list'] if tr in dt[f'tr{end}']]

                    if len(idx_columns) == 0:
                        continue
                    count = dt[f'count{end}'].tocsc()[:,np.array(idx_columns)].tolil()
                    num_nonzero = count.nnz
                    # filter positions < threshold_pos_count
                    count[ count > threshold_pos_count ] = 0
                    print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')

                    # FIXME
                    idx_pos = np.array(dt[f'pos{end}'] >= xlim_now[base][0]) * np.array(dt[f'pos{end}'] < xlim_now[base][1])
                    y = np.array(count.mean(axis=1)).flatten()[idx_pos]
                    idx_pos_ = np.array([x in dt[f'pos{end}'] for x in range(xlim_now[base][0],xlim_now[base][1])])
                    X[ i, idx_pos_ ] = y
            
                size_factor = len(idx_pos_)/100
                fig,axs = plt.subplots(1,1,figsize=(9*size_factor,6))
                axs = sns.heatmap(
                    X,
                    ax=axs,
                    linewidth=0,
                    cmap=sns.color_palette("light:b", as_cmap=True),
                    cbar=False,
                    xticklabels=True)
                idx_y = np.linspace(0,len(dts)-1,10,dtype=int)
                axs.set_yticks(idx_y+0.5)
                axs.set_yticklabels(np.array(list(dts.keys()))[idx_y])
                xlabel_now = np.arange((xlim_now[base][0]//3*3),(xlim_now[base][1]//3*3)-1,3)
                idx_x = xlabel_now - xlim_now[base][0]
                axs.set_xticks(idx_x+0.5)
                axs.set_xticklabels(xlabel_now,rotation=30)
                axs.set_xlabel(f'Position of {end} end from {base} codon (nt)')
                axs.set_ylabel('Read length')

                fig.suptitle(f'{s}')
                fig.tight_layout()
                # plt.show()
                # fig.savefig(outfile_name.with_suffix('.svg'))
                fig.savefig(pdf, format='pdf')
                plt.close('all')
        pdf.close()

def region_mapped(
    save_dir,
    load_dir,
    threshold_tr_count,
    smpls,
    ref,
    read_len_list=[]
):
    save_dir = save_dir / f'region_mapped'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if len(read_len_list)==0:
        outfile_name = save_dir / f'region_mapped'
    else:
        outfile_name = save_dir / f'region_mapped_{read_len_list[0]}-{read_len_list[1]}' 
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in smpls:
        print(f'\nevaluating transcript regions mapped for {s}...')
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count,is_cds=True)
        t0 = time.time()
        if len(read_len_list)>0:
            df_data = df_data.query(
                f'(read_length >= {read_len_list[0]}) and (read_length <= {read_len_list[1]})'
            )
        region = df_data['cds_label'].value_counts() / len(df_data)
        t1 = time.time()
        print(f'\n {np.round(t1-t0,3)} sec')
        order = ['CDS','5UTR','3UTR']
        
        fig,ax = plt.subplots(1,1,figsize=(2,3))
        if np.all([o in region.index for o in order]):
            region.loc[order].plot.bar(
                ax=ax,
                color=[
                    color_region['CDS'],
                    color_region['5UTR'],
                    color_region['3UTR']
                ],
                width=0.8)
        else:
            region.plot.bar(
                ax=ax,
                color=[
                    color_region['CDS'],
                    color_region['5UTR'],
                    color_region['3UTR']
                ],
                width=0.8)

        ax.set_xlabel('Region')
        ax.set_ylabel('Frequency')
        ax.set_ylim(0,1)

        fig.suptitle(s)
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf,format="pdf")
        plt.close('all')
        print("hoge")
    pdf.close()

def inframe_ratio_old(
    save_dir,
    load_dir,
    threshold_tr_count,
    smpls,
    ref,
    readlen_list = []
):
    save_dir = save_dir / f'inframe_ratio'
    if not save_dir.exists():
        save_dir.mkdir()
    order = [0,1,2]

    outfile_name = save_dir / f'inframe_ratio_old'
    if len(readlen_list)>0:
        outfile_name = save_dir / f'inframe_ratio_old_readlen{readlen_list[0]}-{readlen_list[1]}'
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in smpls:
        print(f'\ncalculating in-frame ratio for {s}...')
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count,is_frame=True,is_cds=True)

        if len(readlen_list)>0:
            df_data = df_data.query(f'(read_length >= {readlen_list[0]}) and (read_length <= {readlen_list[1]})').copy()
        ifr = df_data.query('cds_label == "CDS"')['frame5'].value_counts() \
            / len(df_data.query('cds_label == "CDS"'))
        df_tr_frame = df_data.query('cds_label == "CDS"')[['tr_id','frame5']].value_counts().sort_index()
        ifr_tr = [
            df_tr_frame[tr][0] / df_tr_frame[tr].sum()
            if 0 in df_tr_frame[tr].index else 0
            for tr in df_tr_frame.index.get_level_values("tr_id")
        ]
        hist, bin = np.histogram(ifr_tr, bins=100, density=False)
        hist = hist / np.sum(hist)
        x = np.convolve(bin, np.ones(2)/2, mode="valid")

        fig,axs = plt.subplots(2,1,figsize=(3,5),gridspec_kw={'height_ratios': [1, 2]})
        axs[0].bar(
            x,
            hist,
            facecolor="#808080",
            edgecolor="#000000",
            label=s,
            width=0.01)
        axs[0].set_xlabel('IFR distribution')
        axs[0].set_ylabel('Frequency')

        ifr.loc[order].plot.bar(
            ax=axs[1],
            color="#808080",
            width=0.8)
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('Frequency')
        axs[1].set_ylim(0,1)

        fig.suptitle(s)
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf,format="pdf")
        plt.close('all')
        print("hoge")
    
    pdf.close()


def inframe_ratio(
    save_dir,
    load_dir,
    threshold_tr_count,
    smpls,
    ref,
    readlen_list = []
):
    save_dir = save_dir / f'inframe_ratio'
    if not save_dir.exists():
        save_dir.mkdir()
    order = [0,1,2]

    outfile_name = save_dir / f'inframe_ratio'
    if len(readlen_list)>0:
        outfile_name = save_dir / f'inframe_ratio_readlen{readlen_list[0]}-{readlen_list[1]}'
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in smpls:
        print(f'\ncalculating in-frame ratio for {s}...')
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count,is_frame=False,is_cds=False)

        fractions = []
        for tr,df in tqdm(dict_tr.items()):
            if len(df) < threshold_tr_count:
                continue
            if len(readlen_list)>0:
                df = df.query(f'(read_length >= {readlen_list[0]}) and (read_length <= {readlen_list[1]})').copy()
            reads = df['dist5_start'].value_counts().to_dict()
            # ifr_now = []
            for i in range(0,ref.annot.annot_dict[tr]['cds_len'],3):
                fractions_now = [reads.get(i*3+ii,0) for ii in range(3)]
                if np.sum(fractions_now)>0:
                    fractions_now /= np.sum(fractions_now)
                    # ifr_now.append(fractions_now[0])
                    fractions.append(fractions_now)
        
        hist, bin = np.histogram(np.array(fractions)[:,0], bins=100, density=False)
        hist = hist / np.sum(hist)
        x = np.convolve(bin, np.ones(2)/2, mode="valid")

        fig,axs = plt.subplots(2,1,figsize=(3,5),gridspec_kw={'height_ratios': [1, 2]})
        axs[0].bar(
            x,
            hist,
            facecolor="#808080",
            edgecolor="#000000",
            label=s,
            width=0.01)
        axs[0].set_xlabel('IFR distribution')
        axs[0].set_ylabel('Frequency')
    
        df_fractions = pd.DataFrame(np.mean(np.array(fractions),axis=0),columns=['frequency']).set_axis(list(range(3)),axis=0)
        df_fractions.loc[order].plot.bar(
            ax=axs[1],
            color="#808080",
            width=0.8)
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('Frequency')
        axs[1].set_ylim(0,1)

        fig.suptitle(s)
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf,format="pdf")
        plt.close('all')
        print("hoge")
    
    pdf.close()


def _plot_norm_density_ORF(
    fig,
    axs,
    dict_tr,
    threshold_pos_count,
    x,
    title_now,
    col_now,
    ref,
    read_len_list=[]
):
    dict_out = dict(zip(x,np.zeros(len(x))))
    count_out = dict(zip(x,np.zeros(len(x))))
    num_tr = len(dict_tr)
    for i,(tr,v) in enumerate(dict_tr.items()):
        print(f'\r{i}/{num_tr} transcripts...',end='')
        if len(read_len_list)>0:
            v = v[ (v['read_length'] >= read_len_list[0]) * (v['read_length'] <= read_len_list[1]) ]
        rel_pos,cnts = np.unique(
            (v['dist5_start']/ref.annot.annot_dict[tr]['cds_len']).apply(lambda x: round(x,2)).values,
            return_counts=True)
        norm_cnts_ = cnts / (np.sum(v['cds_label']==1) / ref.annot.annot_dict[tr]['cds_len'])
        idx_tmp = norm_cnts_ < threshold_pos_count
        norm_cnts = norm_cnts_[ idx_tmp ]
        rel_pos = rel_pos[ idx_tmp ]
        for rel_pos_,cnt_ in zip(rel_pos,norm_cnts):
            if (rel_pos_ >= x[0]) and (rel_pos_ <= x[-1]):
                dict_out[rel_pos_] += cnt_
                count_out[rel_pos_] += 1
            # elif rel_pos_ < x[0]:
            #     dict_out[ x[0] ] += cnt_
            #     count_out[ x[0] ] += 1
            # elif rel_pos_ > x[-1]:
            #     dict_out[ x[-1] ] += cnt_
            #     count_out[ x[-1] ] += 1

    for k,v in dict_out.items():
        dict_out[k] /= num_tr
    
    axs.plot(x,dict_out.values(),color=col_now)
    axs.axvspan(0,1,color="#FFF8DC",alpha=0.5)
    axs.set_xlabel('Relative position of 5 ends along transcript\n(normalized by CDS length)')
    axs.set_ylabel('Averaged normalized\n density')
    fig.suptitle(title_now)
    fig.tight_layout()
    # plt.show()
    # fig.savefig(outfile_name.with_suffix('.svg'))
    return fig,axs

def _plot_norm_density_gene(
    fig,
    axs,
    dict_tr,
    threshold_pos_count,
    x,
    title_now,
    col_now,
    ref,
    read_len_list=[]
):
    dict_out = dict(zip(x,np.zeros(len(x))))
    count_out = dict(zip(x,np.zeros(len(x))))
    num_tr = len(dict_tr)
    for i,(tr,v) in enumerate(dict_tr.items()):
        print(f'\r{i}/{num_tr} transcripts...',end='')
        if len(read_len_list)>0:
            v = v[ (v['read_length'] >= read_len_list[0]) * (v['read_length'] <= read_len_list[1]) ]
        # if v['biotype'] == 'protein_coding':
        #     rel_pos,cnts = np.unique(
        #         list(map(lambda x:round(x,2),((v['df']['dist5_start'].values + v['df']['start'])/v['tr_info']['cdna_len']))),
        #         return_counts=True)
        # else:
        rel_pos,cnts = np.unique(
            (v['cut5']/ref.annot.annot_dict[tr]['cdna_len']).apply(lambda x: round(x,2)).values,
            return_counts=True)

        norm_cnts_ = cnts / (np.sum(v['cds_label']==1) / (ref.annot.annot_dict[tr]['cds_len']))
        idx_tmp = norm_cnts_ < threshold_pos_count
        norm_cnts = norm_cnts_[ idx_tmp ]
        rel_pos = rel_pos[ idx_tmp ]
        for rel_pos_,cnt_ in zip(rel_pos,norm_cnts):
            dict_out[rel_pos_] += cnt_
            count_out[rel_pos_] += 1

    for k,v in dict_out.items():
        dict_out[k] /= num_tr
    
    axs.plot(x,dict_out.values(),color=col_now)
    axs.set_xlabel('Relative position of 5 ends along transcript')
    axs.set_ylabel('Averaged normalized\n density')
    fig.suptitle(title_now)
    fig.tight_layout()
    # plt.show()
    # fig.savefig(outfile_name.with_suffix('.svg'))
    return fig,axs

def plot_norm_density_ORF(
    save_dir,
    load_dir,
    threshold_tr_count,
    threshold_pos_count,
    smpls,
    ref,
    biotype='',
    read_len_list = []
):
    save_dir = save_dir / 'norm_density_ORF'
    if not save_dir.exists():
        save_dir.mkdir()
    
    outfile_name = save_dir / f'norm_density_ORF_{biotype}_tr{threshold_tr_count}_pos{threshold_pos_count}'
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    x = np.arange(-50,150,1)/100
    x2 = np.arange(0,101,1)/100
    for s in smpls:
        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count)
        print(f'\nPlotting normalized ribosome density over ORFs for {s}...')
        if biotype in ['','protein_coding']:
            fig,axs = plt.subplots(1,1,figsize=(5,3))
            fig,axs = _plot_norm_density_ORF(fig,axs,dict_tr,threshold_pos_count,x,s,"#808080",ref,read_len_list)
            fig.savefig(pdf, format='pdf')
        fig,axs = plt.subplots(1,1,figsize=(5,3))
        fig,axs = _plot_norm_density_gene(fig,axs,dict_tr,threshold_pos_count,x2,s,"#808080",ref,read_len_list)
        fig.savefig(pdf, format='pdf')
    pdf.close()
                 
def fft_ribo(
    save_dir,
    load_dir,
    dict_range:dict,
    fname,
    smpls,
    ref,
    readlen_list
):
    save_dir = save_dir / f'fft_ribo_{fname}'
    if not save_dir.exists():
        save_dir.mkdir()
    
    for start_end,region_range in dict_range.items():

        end,mode = start_end.split('_')
        outfile_name = save_dir / f'fft_aggregation{start_end}_{region_range[0]}-{region_range[1]}'
        pdf = PdfPages(outfile_name.with_suffix(".pdf"))
        for s in smpls:
            print(f'\nPeriodicity analysis by Fourier transformation for {s}...')
            obj = mybin.myBinRiboNorm(
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir,
                mode='start',
                read_len_list=readlen_list,
                is_length=False,
                is_norm=True,
                dirname=''
            )
            obj.decode()
            
            data = np.array(list(obj.count[f'count{end}'].tocsc()[
                (np.array(obj.pos[f'pos{end}']) >= region_range[0])*\
                    (np.array(obj.pos[f'pos{end}']) < region_range[1])
                ].mean(axis=1))).flatten()

            x = np.arange(region_range[0],region_range[1])
            spectrum = fftpack.fft(data)
            amp = np.sqrt((spectrum.real **2) + (spectrum.imag **2))
            amp = amp / (len(data)/2)
            phase = np.arctan2(spectrum.imag,spectrum.real)
            phase = np.degrees(phase)
            freq = np.linspace(0,1,len(data))

            fig,axs = plt.subplots(4,1,figsize=(7,7))
            axs[0].plot(x,data,lw=1,color="#808080")
            axs[0].set_xlabel(f'Nucleotide (nt) from {mode} codon')
            axs[0].set_ylabel('Amplitude')
            # axs[0].set_xticks(np.arange(0,len(data)))
            # axs[0].set_xlim(0,len(data))

            for i in range(2):
                axs[1].axvline(x=(i+1)/3,ymin=0,ymax=np.max(amp),lw=1,color="#FFBF00",linestyle='--')
            axs[1].plot(freq,amp,lw=1,color="#808080")
            axs[1].set_xlabel("Frequency (nt-1)")
            axs[1].set_ylabel("Amplitude")
            
            for i in range(2):
                axs[2].axvline(x=(i+1)/3,ymin=0,ymax=np.max(amp),lw=1,color="#FFBF00",linestyle='--')
            axs[2].axhline(y=0,lw=1,color="#000000",linestyle='--')
            axs[2].axhline(y=90,lw=1,color="#D3D3D3",linestyle='--')
            axs[2].axhline(y=-90,lw=1,color="#D3D3D3",linestyle='--')
            axs[2].plot(freq,phase,lw=1,color="#808080")
            axs[2].set_xlabel("Frequency (nt-1)")
            axs[2].set_ylabel("Phase (degree)")

            period = 1/freq[1:]
            x_freq = period[ (period >= 2) * (period <= 10) ]
            y = amp[1:][ (period >= 2) * (period <= 10) ]
            axs[3].plot(x_freq, y,lw=1,color="#808080")
            axs[3].set_xlabel("Period (nt)")
            axs[3].set_ylabel("Amplitude")

            fig.suptitle(f'Periodicity analysis of {end} end ({s})')
            fig.tight_layout()

            # plt.show()
            fig.savefig(pdf, format='pdf')
            plt.close('all')
        pdf.close()

def _myfft(
    data
):
    spectrum = fftpack.fft(data)
    amp = np.sqrt((spectrum.real **2) + (spectrum.imag **2))
    amp = amp / (len(data)/2)
    phase = np.arctan2(spectrum.imag,spectrum.real)
    phase = np.degrees(phase)
    freq = np.linspace(0,1,len(data))
    return spectrum,amp,phase,freq

def _calc_RPF_end(
    dict_tr,
    range_pos,
    bg,
    ref
):
    
    dict_codon_pos = {};dict_codon_pos_bg = {};dict_seq = {};dict_seq_bg = {}
    for end in ['5','3']:
        dict_codon_pos[end] = {'cnt':{},'rpm':{}};dict_codon_pos_bg[end] = {'cnt':{},'rpm':{}}
        dict_seq[end] = {'seq':[],'cnt':[]};dict_seq_bg[end] = {'seq':[],'cnt':[]}
        for k in ['cnt','rpm']:
            for codon in codon_table.keys():
                dict_codon_pos[end][k][codon] = {};dict_codon_pos_bg[end][k][codon] = {}
                for pos in np.arange(range_pos[0],range_pos[1]):
                    dict_codon_pos[end][k][codon][pos] = 0
                    dict_codon_pos_bg[end][k][codon][pos] = 0
    for i,(tr,df_) in enumerate(dict_tr.items()):
        print(f'\r{i}/{len(dict_tr)} transcripts...',end='')
        start_now = -range_pos[0]*3
        stop_now = ref.annot.annot_dict[tr]["cds_len"] - range_pos[1]*3
        seq = ref.ref_cdna[tr][ref.annot.annot_dict[tr]["start"] :ref.annot.annot_dict[tr]["stop"]]
        aa_seq_ = np.array(re.split('(...)',seq)[1::2])
        for end in ['5','3']:
            df = df_.query(f'(dist{end}_start >= {start_now}) and (dist{end}_start <= {stop_now})')
            tot_cnt = len(df)
            if tot_cnt==0:
                continue
            for pos_ in df[f'dist{end}_start'].unique():
                cnt_now = np.sum(df[f'dist{end}_start'].values == pos_)
                if cnt_now>1e+4:
                    continue
                rpm_now = cnt_now / tot_cnt
                # if cnt_now >= threshold_pos:
                #     continue
                pos = pos_//3*3
                dict_seq[end]['seq'].append(seq[ int(pos_) + range_pos[0]*3 : int(pos_) + range_pos[1]*3 ])
                dict_seq[end]['cnt'].append(cnt_now)
                aa_seq = aa_seq_[ int(pos/3) + range_pos[0] : int(pos/3) + range_pos[1] ] 
                for j,aa in enumerate(aa_seq):
                    dict_codon_pos[end]['cnt'][aa][j+range_pos[0]] += cnt_now
                    dict_codon_pos[end]['rpm'][aa][j+range_pos[0]] += rpm_now
            # backgrounds such as randomly distributed reads
            if len(bg) == 0:
                continue
            elif bg == 'random':
                stop_tr = ref.annot.annot_dict[tr]['stop']
                start_tr = ref.annot.annot_dict[tr]['start']
                np.random.seed()
                pos_random = np.random.randint(-range_pos[0]*3,(stop_tr-start_tr)-range_pos[1]*3,size=tot_cnt)
                for pos_ in np.unique(pos_random):
                    cnt_now = np.sum(pos_random == pos_)
                    # if cnt_now >= threshold_pos:
                    #     continue
                    rpm_now = cnt_now / tot_cnt
                    pos = pos_//3*3
                    dict_seq_bg[end]['seq'].append(seq[ int(pos_) + range_pos[0]*3 : int(pos_) + range_pos[1]*3 ])
                    dict_seq_bg[end]['cnt'].append(cnt_now)
                    aa_seq = aa_seq_[ int(pos/3) + range_pos[0] : int(pos/3) + range_pos[1] ]
                    for j,aa in enumerate(aa_seq):
                        dict_codon_pos_bg[end]['cnt'][aa][j+range_pos[0]] += cnt_now
                        dict_codon_pos_bg[end]['rpm'][aa][j+range_pos[0]] += rpm_now
            elif type(bg) == dict:
                df_bg = bg[tr]['df'].query(f'(dist{end}_start >= {start_now}) and (dist{end}_start <= {stop_now})')
                tot_cnt = len(df_bg)
                if tot_cnt==0:
                    continue
                for pos_ in df_bg[f'dist{end}_start'].unique():
                    cnt_now = np.sum(df_bg[f'dist{end}_start'].values == pos_)
                    if cnt_now>1e+4:
                        continue
                    rpm_now = cnt_now / tot_cnt
                    # if cnt_now >= threshold_pos:
                    #     continue
                    pos = pos_//3*3
                    dict_seq_bg[end]['seq'].append(seq[ int(pos_) + range_pos[0]*3 : int(pos_) + range_pos[1]*3 ])
                    dict_seq_bg[end]['cnt'].append(cnt_now)
                    aa_seq = aa_seq_[ int(pos/3) + range_pos[0] : int(pos/3) + range_pos[1] ]
                    for j,aa in enumerate(aa_seq):
                        dict_codon_pos_bg[end]['cnt'][aa][j+range_pos[0]] += cnt_now
                        dict_codon_pos_bg[end]['rpm'][aa][j+range_pos[0]] += rpm_now

        dict_codon_pos2 = {};dict_codon_pos_bg2 = {}
        for end in ['3','5']:
            dict_codon_pos2[end] = {};dict_codon_pos_bg2[end] = {}
            for cnt in ['cnt','rpm']:
                dict_codon_pos2[end][cnt] = {};dict_codon_pos_bg2[end][cnt] = {}
                for k,v in dict_codon_pos[end][cnt].items():
                    dict_codon_pos2[end][cnt][k] = np.array(list(v.values()))
                for k,v in dict_codon_pos_bg[end][cnt].items():
                    dict_codon_pos_bg2[end][cnt][k] = np.array(list(v.values()))
            
    return dict_codon_pos2,dict_codon_pos_bg2,dict_seq,dict_seq_bg
 
def codon_RPF_end(
    save_dir,
    load_dir,
    pairs,
    ref,   
    is_save_kplogo=False
):
    save_dir = save_dir / 'codon_RPF_end'
    if not save_dir.exists():
        save_dir.mkdir()
    save_dir = Path(save_dir)

    threshold_tr_count = 0
    range_pos = [-10,11]
    x = np.arange(range_pos[0],range_pos[1])

    for pair in pairs:
        outfile_name = save_dir / f'codon_RPF_end_{pair[0]}_{pair[1]}'
        pdf = PdfPages(outfile_name.with_suffix(".pdf"))
        dict_pos_list = {'5':{},'3':{}}
        for p in pair:
            print(f'Analyzing codon distribution around RPF reads for {p} reads...')

            obj = mybin.myBinRibo(
                data_dir=ref.data_dir,
                smpl=p,
                sp=ref.sp,
                save_dir=load_dir
            )
            obj.decode()
            df_data,dict_tr = obj.make_df(tr_cnt_thres=-1,is_seq=True)
            dict_codon_pos2,dict_codon_pos_bg2,dict_seq,dict_seq_bg = _calc_RPF_end(
                dict_tr=dict_tr,
                range_pos=range_pos,
                ref=ref,
                bg='random')
            
            for end in ['5','3']:
                if is_save_kplogo:
                    pd.DataFrame().from_dict(dict_seq[end],orient='columns').to_csv( save_dir / f'codon_RPF_{end}end_{p}.tsv', sep='\t', header=None ,index=False)
                    pd.DataFrame().from_dict(dict_seq_bg[end],orient='columns').to_csv( save_dir / f'codon_RPF_{end}end_{p}_bg.tsv', sep='\t',  header=None, index=False )
                dict_pos_list[end][p] = {}
                for cnt,cnt_name in zip(['cnt','rpm'],['count','rpm']):
                    df_codon_pos = pd.DataFrame().from_dict(dict_codon_pos2[end][cnt],orient='index')
                    df_codon_pos = df_codon_pos.set_axis(x,axis='columns',copy=False)
                    df_codon_pos.to_csv( save_dir / f'codon_eRPF_{end}end_{p}_{cnt}.csv.gz' )
                    df_codon_pos = df_codon_pos / df_codon_pos.sum(axis=0) *100
                    df_codon_pos_bg = pd.DataFrame().from_dict(dict_codon_pos_bg2[end][cnt],orient='index')
                    df_codon_pos_bg = df_codon_pos_bg.set_axis(x,axis='columns',copy=False)
                    df_codon_pos_bg = df_codon_pos_bg / df_codon_pos_bg.sum(axis=0) *100
                    valmax = np.amax([df_codon_pos.values.max(axis=None),df_codon_pos_bg.values.max(axis=None)])

                    dict_pos_list[end][p][cnt] = df_codon_pos
                
                    fig,axs = plt.subplots(8,8,figsize=(16,16))
                    for i, (codon,dat) in enumerate(df_codon_pos.iterrows()):
                        nrow = i//8
                        ncol = i%8
                        axs[nrow,ncol].vlines(x[ x%5==0 ],0,valmax,color="#808080",linestyles='dashed')
                        if len(df_codon_pos_bg)>0:
                            dat_bg = df_codon_pos_bg.loc[codon,:].values
                            axs[nrow,ncol].plot(x,dat_bg,color="#808080")
                        axs[nrow,ncol].plot(x,dat,color="#FF0000")
                        axs[nrow,ncol].set_title(codon)
                        axs[nrow,ncol].set_xlabel(f'position from {end} end')
                        axs[nrow,ncol].set_ylabel(f'{cnt_name}(%)')
                        axs[nrow,ncol].set_ylim([0,valmax])
                    fig.suptitle(f'{p} (red), random (gray) {end} end')
                    fig.tight_layout()
                    # plt.show()
                    fig.savefig(pdf, format='pdf')
                    plt.close('all')

        # both
        for end in ['5','3']:
            for cnt,cnt_name in zip(['cnt','rpm'],['count','rpm']):
                valmax = np.amax([dict_pos_list[end][pair[0]][cnt].values.max(axis=None),dict_pos_list[end][pair[1]][cnt].values.max(axis=None)])
                
                fig,axs = plt.subplots(8,8,figsize=(16,16))
                for i, (codon,dat) in enumerate(dict_pos_list[end][pair[0]][cnt].iterrows()):
                    dat_bg = dict_pos_list[end][pair[1]][cnt].loc[codon,:].values
                    nrow = i//8
                    ncol = i%8
                    axs[nrow,ncol].vlines(x[ x%5==0 ],0,valmax,color="#808080",linestyles='dashed')
                    axs[nrow,ncol].plot(x,dat_bg,color="#F4BB44")
                    axs[nrow,ncol].plot(x,dat,color="#800080")
                    axs[nrow,ncol].set_title(codon)
                    axs[nrow,ncol].set_xlabel(f'position from {end} end')
                    axs[nrow,ncol].set_ylabel(f'{cnt_name}(%)')
                    axs[nrow,ncol].set_ylim([0,valmax])
                fig.suptitle(f'{pair[0]} (purple),{pair[1]} (yellow), {end} end')
                fig.tight_layout()
                # plt.show()
                fig.savefig(pdf, format='pdf')
                plt.close('all')
        pdf.close()
   
def aggregation_ribo_3end(
    save_dir,
    load_dir,
    plot_range,
    threshold_tr_count,
    threshold_pos_count,
    col_list,
    ref
):
    save_dir = save_dir / 'aggregation_ribo_3end'
    if not save_dir.exists():
        save_dir.mkdir()
    
    save_dir = Path(save_dir)
    outfile_name = save_dir / (
        'aggregation_ribo_3end_'
        f'pos{plot_range[0]}-{plot_range[1]}_'
        f'tr{threshold_tr_count}_pos{threshold_pos_count}'
    )
    
    df_agg = {}
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in col_list.keys():
        print(f'\naggregation plots (aligned to 3 end of transcript) for {s}...')

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=save_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count,is_seq=False)
        # df_data = df_data[[x in dict_tr.keys() for x in df_data['tr_id']]]
        df_data_readlen = df_data.groupby('read_length')
        read_len_list = np.sort(list(df_data_readlen.groups.keys()))

        dt = myprep._calc_norm_ribo_density(
            df_data=df_data,
            dict_tr=dict_tr,
            read_len=[read_len_list[0],read_len_list[-1]],
            mode='3end',
            ref=ref,
            is_norm=True)
        # filter transcripts > threshold_tr_count
        if type(threshold_tr_count) is int:
            idx_columns = np.array(dt['count5'].sum(axis=0)[0] > threshold_tr_count).flatten()
        elif type(threshold_tr_count) is dict:
            tr_list = threshold_tr_count['tr_list']
            idx_columns = [np.where(np.array(dt['tr5']) == tr)[0][0] for tr in tr_list]
        count = dt['count5'].tocsc()[:,np.array(idx_columns)].tolil()
        num_nonzero = count.nnz
        # filter positions < threshold_pos_count
        count[ count > threshold_pos_count ] = 0
        print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')

        # FIXME
        y = []
        for pos in range(plot_range[0],plot_range[1]):
            if pos in dt['pos5']:
                y.append( count[ dt['pos5'] == pos, : ].mean(axis=1).tolist()[0][0] )
            else:
                y.append(0)
        y = np.array(y)
        x = np.array(list(range(plot_range[0],plot_range[1])))
        df_agg[s] = [x,y]
    
    # subplots for all samples
    for j,s in enumerate(col_list.keys()):
        fig,axs = plt.subplots(1,1,figsize=(3,3),sharex=True,sharey=True)
        axs.plot(
            df_agg[s][0],
            df_agg[s][1],
            color="#808080")
        axs.set_xlabel(f"Distance from end of transcript (nt)")
        axs.set_ylabel('Normalized read density')
        axs.set_title(s)
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close('all')
    pdf.close()

  
def aggregation_ribo_5end(
    save_dir,
    load_dir,
    plot_range,
    threshold_tr_count,
    threshold_pos_count,
    col_list,
    ref
):
    save_dir = save_dir / 'aggregation_ribo_5end'
    if not save_dir.exists():
        save_dir.mkdir()
    
    save_dir = Path(save_dir)
    outfile_name = save_dir / (
        'aggregation_ribo_5end_'
        f'pos{plot_range[0]}-{plot_range[1]}_'
        f'tr{threshold_tr_count}_pos{threshold_pos_count}'
    )
    
    df_agg = {}
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    for s in col_list.keys():
        print(f'\naggregation plots (aligned to 5 end of transcript) for {s}...')

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=threshold_tr_count,is_seq=False)
        # df_data = df_data[[x in dict_tr.keys() for x in df_data['tr_id']]]
        df_data_readlen = df_data.groupby('read_length')
        read_len_list = np.sort(list(df_data_readlen.groups.keys()))

        dt = myprep._calc_norm_ribo_density(
            df_data=df_data,
            dict_tr=dict_tr,
            read_len=[read_len_list[0],read_len_list[-1]],
            mode='5end',
            ref=ref,
            is_norm=True)
        # filter transcripts > threshold_tr_count
        if type(threshold_tr_count) is int:
            idx_columns = np.array(dt['count5'].sum(axis=0)[0] > threshold_tr_count).flatten()
        elif type(threshold_tr_count) is dict:
            tr_list = threshold_tr_count['tr_list']
            idx_columns = [np.where(np.array(dt['tr5']) == tr)[0][0] for tr in tr_list]
        count = dt['count5'].tocsc()[:,np.array(idx_columns)].tolil()
        num_nonzero = count.nnz
        # filter positions < threshold_pos_count
        count[ count > threshold_pos_count ] = 0
        print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')

        # FIXME
        y = []
        for pos in range(plot_range[0],plot_range[1]):
            if pos in dt['pos5']:
                y.append( count[ dt['pos5'] == pos, : ].mean(axis=1).tolist()[0][0] )
            else:
                y.append(0)
        y = np.array(y)
        x = np.array(list(range(plot_range[0],plot_range[1])))
        df_agg[s] = [x,y]
    
    # subplots for all samples
    for j,s in enumerate(col_list.keys()):
        fig,axs = plt.subplots(1,1,figsize=(3,3),sharex=True,sharey=True)
        axs.plot(
            df_agg[s][0],
            df_agg[s][1],
            color="#808080")
        axs.set_xlabel(f"Distance from start of transcript (nt)")
        axs.set_ylabel('Normalized read density')
        axs.set_title(s)
        fig.tight_layout()
        # plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close('all')
    pdf.close()

def aggregation_ribo_commontr(
    save_dir,
    load_dir,
    plot_range5,
    plot_range3,
    threshold_tr_count,
    threshold_pos_count,
    pair,
    ref,
    read_len_list = [],
    is_norm=True
    ):

    save_dir = save_dir / 'aggregation_ribo'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if len(read_len_list)>0:
        outfile_name = save_dir / (
            'aggregation_ribo_'
            f'5pos{plot_range5[0]}-{plot_range5[1]}_'
            f'3pos{plot_range3[0]}-{plot_range3[1]}_'
            f'tr{threshold_tr_count}_pos{threshold_pos_count}_'
            f'readlen{read_len_list[0]}-{read_len_list[1]}_'
            f'{pair[0]}_{pair[1]}'
        )
    else:
        outfile_name = save_dir / (
            'aggregation_ribo_'
            f'5pos{plot_range5[0]}-{plot_range5[1]}_'
            f'3pos{plot_range3[0]}-{plot_range3[1]}_'
            f'tr{threshold_tr_count}_pos{threshold_pos_count}'
            f'{pair[0]}_{pair[1]}'
        )
    xlim_now = {'start':plot_range5,'stop':plot_range3}

    df_agg = {}
    pdf = PdfPages(outfile_name.with_suffix(".pdf"))
    dts = {};dict_idx = {}
    for s in pair:
        dts[s] = {};dict_idx[s] = {}
        for base in ('start','stop'):
            obj = mybin.myBinRiboNorm(
                smpl=s,
                sp=ref.sp,
                save_dir=load_dir,
                mode=base,
                read_len_list=read_len_list,
                is_length=False,
                is_norm=is_norm,
                dirname=''
            )
            dt = obj.decode()
            # filter transcripts > threshold_tr_count
            idx_columns = np.array(dt['count5'].sum(axis=0)[0] > threshold_tr_count).flatten()

            dts[s][base] = dt
            dict_idx[s][base] = np.array(dt['tr5'])[idx_columns]
    
    # common transcripts
    tr_list_common = list(
        set(dict_idx[pair[0]]['start']) & set(dict_idx[pair[1]]['start'])
    )
    print(f'{len(tr_list_common)} transcripts are shared...')
    # only in pait[0]
    tr_list0 = list(
        set(dict_idx[pair[0]]['start']) - set(dict_idx[pair[1]]['start'])
    )
    print(f'{len(tr_list0)} transcripts are only in {pair[0]}...')
    tr_list1 = list(
        set(dict_idx[pair[1]]['start']) - set(dict_idx[pair[0]]['start'])
    )
    print(f'{len(tr_list1)} transcripts are only in {pair[1]}...')

    for tr_list,title_now in zip(
        [tr_list_common,tr_list0,tr_list1],
        ['common',f'only in {pair[0]}',f'only in {pair[1]}']
        
    ):
        if title_now == 'common':
            pair_ = pair
        elif pair[0] in title_now:
            pair_ = [pair[0]]
        elif pair[1] in title_now:
            pair_ = [pair[1]]

        for s in pair_:
            xs = {}
            ys = {}
            for base in ('start','stop'):
                dt = dts[s][base]
                len_tr = len(dt['tr5'])
                idx_columns = np.array([tr in tr_list for tr in dt['tr5']])
                count = dt['count5'].tocsc()[:,idx_columns].tolil()
                num_nonzero = count.nnz
                # filter positions < threshold_pos_count
                count[ count > threshold_pos_count ] = 0
                print(f'{num_nonzero-count.nnz} transcript-position pairs are filtered...')
                # FIXME
                idx_ = np.where(
                    (np.array(dt['pos5'])>=xlim_now[base][0]) * (np.array(dt['pos5'])<=xlim_now[base][1]))[0]
                x = np.array(dt['pos5'])[idx_]
                if is_norm:
                    y = np.array(count.mean(axis=1)).flatten()[idx_]
                else:
                    y = np.array(count.sum(axis=1)).flatten()[idx_]

                xs[base] = x
                ys[base] = y
            if 'start' not in xs.keys():
                continue
            fig,axs = plt.subplots(2,2,figsize=(9,6))
            for i,base in enumerate(["start","stop"]):
                axs[0,i].plot(xs[base],ys[base],color="#71797E")
                axs[0,i].set_xlabel(f"Distance from {base} codon (nt)")
                axs[0,i].set_ylabel("Mean read density")
                axs[0,i].set_xlim(xlim_now[base][0],xlim_now[base][1])
            # frame
            frames = [''] * 3
            for i in range(3):
                for j,base in enumerate(["start","stop"]):
                    idx = (xs[base] % 3 == i)
                    frames[i], = axs[1,j].plot(
                        xs[base][idx],
                        ys[base][idx],
                        color=color_frame[f'frame{i}'],
                        label=f'frame{i}')
                    axs[1,j].set_xlabel(f"Distance from {base} codon (nt)")
                    if is_norm:
                        axs[1,j].set_ylabel("Mean read density")
                    else:
                        axs[1,j].set_ylabel("Read counts")
                axs[1,j].set_xlim(xlim_now[base][0],xlim_now[base][1])
            axs[1,1].legend(handles=[frames[0],frames[1],frames[2]],loc='upper right')
            # match y axis
            ylim_max = np.max([axs[0,0].get_ylim()[1],axs[0,1].get_ylim()[1]])
            axs[0,0].set_ylim(-ylim_max*0.05,ylim_max)
            axs[0,1].set_ylim(-ylim_max*0.05,ylim_max)
            axs[1,0].set_ylim(-ylim_max*0.05,ylim_max)
            axs[1,1].set_ylim(-ylim_max*0.05,ylim_max)

            fig.suptitle(f'{s} ({title_now} {len(tr_list)}/{len_tr} transcripts)')
            fig.tight_layout()
            # plt.show()
            # fig.savefig(outfile_name.with_suffix('.svg'))
            fig.savefig(pdf, format='pdf')
            plt.close('all')

            df_agg[s] = [xs,ys]

        if np.any([p in title_now for p in pair]):
            continue
        # subplots for all samples
        fig,axs = plt.subplots(2,len(df_agg),figsize=(len(df_agg)*3,6),sharex=True,sharey=True)
        for i,base in enumerate(["start","stop"]):
            for j,s in enumerate(pair_):
                for k in range(3):
                    idx = (df_agg[s][0][base] % 3 == k)
                    if len(df_agg)==1:
                        axs[i].plot(
                            df_agg[s][0][base][idx],
                            df_agg[s][1][base][idx],
                            color=color_frame[f'frame{k}'],
                            label=f'frame{i}')
                    else:
                        axs[i,j].plot(
                            df_agg[s][0][base][idx],
                            df_agg[s][1][base][idx],
                            color=color_frame[f'frame{k}'],
                            label=f'frame{i}')
                if len(df_agg)==1:
                    axs[i].set_xlabel(f"Distance from {base} codon (nt)")
                    axs[i].set_ylabel('Normalized read density')
                    axs[i].set_title(s)
                else:
                    axs[i,j].set_xlabel(f"Distance from {base} codon (nt)")
                    axs[i,j].set_ylabel('Normalized read density')
                    axs[i,j].set_title(s)
        fig.tight_layout(rect=[0, 0, .9, 1])
        if len(df_agg)==1:
            axs[i].legend(labels=[f'frame{x}' for x in range(3)],loc='upper left',bbox_to_anchor=(1,1))
        else:
            axs[i,j].legend(labels=[f'frame{x}' for x in range(3)],loc='upper left',bbox_to_anchor=(1,1))
        # plt.show()
        fig.savefig(pdf, format='pdf')
        plt.close('all')
    
    pdf.close()
   

def plot_norm_region_density(
    save_dir,
    load_dir,
    ref,
    smpls
):
    save_dir = save_dir / 'norm_region_density'
    if not save_dir.exists():
        save_dir.mkdir()
    
    ref.exp_metadata.df_metadata = ref.exp_metadata.df_metadata.set_index(['sample_name'])
    
    dict_dens_cds = {};dict_dens_utr5 = {};dict_dens_utr3 = {}
    total_reads = {}
    for s in smpls:

        obj = mybin.myBinRibo(
            data_dir=ref.data_dir,
            smpl=s,
            sp=ref.sp,
            save_dir=load_dir
        )
        obj.decode()
        df_data,dict_tr = obj.make_df(tr_cnt_thres=8,is_cds=True)
        dict_tr_now = list(df_data.groupby('tr_id'))

        dens_cds = np.zeros(102)
        dens_utr5 = np.zeros(100)
        dens_utr3 = np.zeros(100)
        
        n_reads = 0
        for i,(tr,df) in enumerate(dict_tr_now):
            print(f'\r{i}/{len(dict_tr_now)} transcript...',end='')
            len_utr5 = ref.annot.annot_dict[tr]['start']
            len_cds = ref.annot.annot_dict[tr]['cds_len']
            len_utr3 = ref.annot.annot_dict[tr]['cdna_len'] - len_utr5 - len_cds

            pos_cds = df['dist5_start'].values / (len_cds-1) - 0.01/2
            pos_cds = pos_cds[ (pos_cds>=-0.015)*(pos_cds<1.005) ]
            dens_cds[ np.round(pos_cds*100).astype(int)+1 ] += 1
            n_reads += len(pos_cds)

            if len_utr5 != 0:
                pos_utr5 = (df['dist5_start'].values + len_utr5) / len_utr5 - 0.01/2
                pos_utr5 = pos_utr5[ (pos_utr5>=-0.005)*(pos_utr5<0.995) ]
                dens_utr5[ np.round(pos_utr5*100).astype(int) ] += 1
                n_reads += len(pos_utr5)

            if len_utr3 != 0:
                pos_utr3 = df['dist5_stop'].values / (len_utr3-1) - 0.01/2
                pos_utr3 = pos_utr3[ (pos_utr3>=-0.005)*(pos_utr3<0.995) ]
                dens_utr3[ np.round(pos_utr3*100).astype(int) ] += 1
                n_reads += len(pos_utr3)
        
        dict_dens_cds[s] = dens_cds
        dict_dens_utr5[s] = dens_utr5
        dict_dens_utr3[s] = dens_utr3
        total_reads[s] = n_reads
    
    outfile_name = save_dir / 'norm_region_density.pdf'
    pdf = PdfPages(outfile_name)
    for j,s in enumerate(smpls):
        fig,ax = plt.subplots(1,1,figsize=(5,2))

        ax.plot(
            np.arange(start=-1,stop=2.0,step=0.01),
            np.hstack((
                dict_dens_utr5[s]/total_reads[s]*1e+6,
                dict_dens_cds[s][1:-1]/total_reads[s]*1e+6,
                dict_dens_utr3[s]/total_reads[s]*1e+6)
                ),
            color="#808080"
        )
        fig.suptitle(s)
        
        ax.set_xlabel('position')
        ax.set_ylabel('RPM density')
        
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
    pdf.close()

def plot_readlen_fastq(
    save_dir:Path,
    dict_fastq:dict,
    col_list={},
    suffix=''
):
    save_dir = save_dir / 'readlen_fastq'
    if not save_dir.exists():
        save_dir.mkdir()
    
    outfile_name = save_dir / f'readlen_{suffix}.pdf'
    pdf = PdfPages(outfile_name)
    dict_read_len = {};smpls = []
    for smpl,load_path in dict_fastq.items():

        read_len_list = []
        with gzip.open(load_path,'rt') as f:
            for j,rec in enumerate(SeqIO.parse(f,'fastq')):
                print(f'\r{j} reads...',end='')
                read_len_list.append(len(rec.seq))
        read_len,read_len_cnt = np.unique(read_len_list,return_counts=True)

        fig,ax = plt.subplots(1,1,figsize=(5,2))
        ax.plot(
            read_len,
            read_len_cnt,
            color="#808080"
        )
        fig.suptitle(smpl)
        
        ax.set_xlabel('read length')
        ax.set_ylabel('count')
        
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')

        dict_read_len[smpl] = {}
        dict_read_len[smpl]['read_len'] = read_len
        dict_read_len[smpl]['read_len_cnt'] = read_len_cnt
        smpls.append(smpl)
    
    if len(col_list)>0:
        fig,ax = plt.subplots(1,1,figsize=(5,2))
        for smpl,v in dict_read_len.items():
            ax.plot(
                v['read_len'],
                v['read_len_cnt'],
                color=col_list[smpl],
                label=smpl
            )
        ax.legend(labels=smpls,loc='upper right')
        
        ax.set_xlabel('read length')
        ax.set_ylabel('count')
    
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')

        # percentage
        fig,ax = plt.subplots(1,1,figsize=(5,2))
        for smpl,v in dict_read_len.items():
            ax.plot(
                v['read_len'],
                v['read_len_cnt']/np.sum(v['read_len_cnt']),
                color=col_list[smpl],
                label=smpl
            )
        ax.legend(labels=smpls,loc='upper right')
        
        ax.set_xlabel('read length')
        ax.set_ylabel('fraction')
    
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')

    pdf.close()

    # data output
    # data output
    read_len = np.arange(
        np.min([x['read_len'][0] for x in dict_read_len.values()]),
        np.max([x['read_len'][-1] for x in dict_read_len.values()])
    )
    pd.DataFrame(
        {
            smpl:[
                v['read_len_cnt'][ np.where(r == v['read_len'])[0][0] ]
                if r in v['read_len'] else 0
                for r in read_len
            ]
            for smpl,v in dict_read_len.items()
        },index=read_len
    ).to_csv(outfile_name.with_suffix('.csv.gz'))

def plot_readlen_bam(
    save_dir:Path,
    dict_bam:dict,
    col_list={},
    suffix='',
    full_align=False
):
    save_dir = save_dir / 'readlen_bam'
    if not save_dir.exists():
        save_dir.mkdir()
    
    outfile_name = save_dir / f'readlen_{suffix}.pdf'
    pdf = PdfPages(outfile_name)
    dict_read_len = {};smpls = []
    for smpl,load_path in dict_bam.items():

        infile = pysam.AlignmentFile(load_path)
        if full_align:
            read_len_list = [
                read.query_length
                for read in infile.fetch(until_eof=True)
                if read.query_length == read.query_alignment_length
            ]
        else:
            read_len_list = [
                read.query_length
                for read in infile.fetch(until_eof=True)
            ]
        read_len,read_len_cnt = np.unique(read_len_list,return_counts=True)

        fig,ax = plt.subplots(1,1,figsize=(5,2))
        ax.plot(
            read_len,
            read_len_cnt,
            color="#808080"
        )
        fig.suptitle(smpl)
        
        ax.set_xlabel('read length')
        ax.set_ylabel('count')
        
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')

        dict_read_len[smpl] = {}
        dict_read_len[smpl]['read_len'] = read_len
        dict_read_len[smpl]['read_len_cnt'] = read_len_cnt
        smpls.append(smpl)
    
    if len(col_list)>0:
        fig,ax = plt.subplots(1,1,figsize=(5,2))
        for smpl,v in dict_read_len.items():
            ax.plot(
                v['read_len'],
                v['read_len_cnt'],
                color=col_list[smpl],
                label=smpl
            )
        ax.legend(labels=smpls,loc='upper right')
        
        ax.set_xlabel('read length')
        ax.set_ylabel('count')
        
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')

    pdf.close()

    # data output
    read_len = np.arange(
        np.min([x['read_len'][0] for x in dict_read_len.values()]),
        np.max([x['read_len'][-1] for x in dict_read_len.values()])
    )
    pd.DataFrame(
        {
            smpl:[
                v['read_len_cnt'][ np.where(r == v['read_len'])[0][0] ]
                if r in v['read_len'] else 0
                for r in read_len
            ]
            for smpl,v in dict_read_len.items()
        },index=read_len
    ).to_csv(outfile_name.with_suffix('.csv.gz'))

def mapped_ref_bam(
    save_dir:Path,
    dict_bam:dict,
    suffix='',
    readlen=[]
):
    save_dir = save_dir / 'mapped_ref_bam'
    if not save_dir.exists():
        save_dir.mkdir()
    
    outfile_name = save_dir / f'mapped_ref_{suffix}.pdf'
    pdf = PdfPages(outfile_name)
    for smpl,load_path in dict_bam.items():

        infile = pysam.AlignmentFile(load_path)
        if len(readlen)>0:
            ref_names_list = [
                read.reference_name
                for read in infile.fetch(until_eof=True)
                if (read.query_length >= readlen[0]) and (read.query_length <= readlen[1])
            ]
        
        else:
            ref_names_list = [
                read.reference_name
                for read in infile.fetch(until_eof=True)
            ]
        ref_names,ref_cnt = np.unique(ref_names_list,return_counts=True)
        dict_ref_cnt = dict(zip(ref_names,ref_cnt))

        # summarize pseudogenes for tRNAs
        # FIXME
        ref_names_uniq_list = []
        dict_ref_uniq_cnt = {}
        for ref_name in ref_names:
            ref_name_ = re.findall(r'Homo_sapiens_(tRNA-[A-Za-z]*-[A-Z]{3})-[0-9]{1,2}-[0-9]{1,2}',ref_name)[0]
            ref_names_uniq_list.append(ref_name_)
            if ref_name_ in dict_ref_uniq_cnt.keys():
                dict_ref_uniq_cnt[ref_name_] += dict_ref_cnt[ref_name]
            else:
                dict_ref_uniq_cnt[ref_name_] = dict_ref_cnt[ref_name]
        
        df_ref_uniq_cnt = pd.DataFrame().from_dict(dict_ref_uniq_cnt,orient='index')\
            .set_axis(['cnt'],axis=1).sort_values(by='cnt',ascending=False)
        
        fig,ax = plt.subplots(1,1,figsize=(10,4))
        df_ref_uniq_cnt.plot.bar(
            color="#808080",
            ax=ax
        )
        fig.suptitle(smpl)
        
        ax.set_xlabel('reference')
        ax.set_ylabel('count')
        
        # for ax in axs:
        #     ax.set_ylabel('reads')
        fig.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close('all')
    pdf.close()

def extract_reads_bylength(
    save_dir:Path,
    dict_fastq_bam:dict,
    readlen:int,
    n_top:int,
    suffix=''
):
    save_dir = save_dir / 'kplogo'
    if not save_dir.exists():
        save_dir.mkdir()

    def func(pct,allvals):
        absolute = int(np.round(pct/100*np.sum(allvals)))
        if pct>1:
            out = f'{pct:.1f}% ({absolute:d})'
        else:
            out = ''
        return out

    _,colors = my._get_spaced_colors(n=n_top)

    for smpl,load_path in dict_fastq_bam.items():

        # seq_pos = {}
        # for i in range(readlen):
        #     seq_pos[i] = dict(zip(['A','T','C','G','N'],np.zeros(5,dtype=int)))


        smpl = smpl.replace(' ','_')
        seqs = []

        if (save_dir / f'kplogo_{smpl}_readlen{readlen}_{suffix}.fa.gz').exists():

            with gzip.open(save_dir / f'kplogo_{smpl}_readlen{readlen}_{suffix}.fa.gz','rt') as f:
                for j,rec in enumerate(SeqIO.parse(f,'fasta')):
                    print(f'\r{j} reads...',end='')
                    seqs.append(str(rec.seq))

        else:
            fout = gzip.open( save_dir / f'kplogo_{smpl}_readlen{readlen}_{suffix}.fa.gz', 'wt' )

            if load_path.suffix == '.gz':

                with gzip.open(load_path,'rt') as f:
                    for j,rec in enumerate(SeqIO.parse(f,'fastq')):
                        print(f'\r{j} reads...',end='')
                        if len(rec.seq) == readlen:
                            SeqIO.write(rec,fout,'fasta')
                            # for i in range(readlen):
                            #     seq_pos[i][rec.seq[i]] += 1
                            seqs.append(str(rec.seq))
            
            elif load_path.suffix == '.bam':

                infile = pysam.AlignmentFile(load_path)
                for read in infile.fetch(until_eof=True):
                    if read.query_length == readlen:
                        fout.write('>' + read.qname + '\n')
                        fout.write(read.query_sequence + '\n')
                        seqs.append(read.query_sequence)
        
            fout.close()

        seq_uniq,cnt_seq = np.unique(seqs,return_counts=True)
        idx = np.argsort(cnt_seq)[::-1]
        pd.DataFrame({
            'count':cnt_seq[idx],
            '%total reads':cnt_seq[idx]/len(seqs)*100
        },index=seq_uniq[idx]).to_csv(save_dir / f'uniq_seq_{smpl}_readlen{readlen}_{suffix}.csv.gz')

        # calculate how many reads are close to top-hit sequence
        seq_top_grps = [];cnt_top_grps = [];cnt_sum_grps = []
        thres = 3 / readlen
        j = 0;cnt = 0
        while cnt < n_top:
            seq_top = seq_uniq[idx[j]]
            if cnt>0:
                
                # FIXME
                dist = np.min([
                        np.min([
                            distance.hamming(list(seq_top),list(seq_top_grps[i][ii]))
                            for ii in range(10)
                        ])
                        for i in range(cnt)
                    ])
            else:
                dist = 1
            if dist <= thres:
                j += 1
            else:
                seq_top_grp = [];cnt_top_grp = []
                for i in idx:
                    dist = distance.hamming(list(seq_top),list(seq_uniq[i]))
                    ################
                    if seq_top[0] != seq_uniq[i][0]:
                        dist += 1
                    ################
                    if dist <= thres:
                        seq_top_grp.append(seq_uniq[i])
                        cnt_top_grp.append(cnt_seq[i])
                seq_top_grps.append(seq_top_grp)
                cnt_top_grps.append(cnt_top_grp)
                cnt_sum_grps.append(np.sum(cnt_top_grp))

                pd.DataFrame({
                    'count':cnt_top_grp,
                    '%total reads':np.array(cnt_top_grp) / len(seqs)*100
                },index=seq_top_grp).to_csv(save_dir / f'uniq_top{cnt}_seq_{smpl}_readlen{readlen}_{suffix}_mm3.csv.gz')
                cnt += 1
        
        vals = cnt_sum_grps + [len(seqs) - np.sum(cnt_sum_grps)]
        val_labels = [x[0] for x in seq_top_grps] + ['others']
        
        # pie chart
        outfile_name = save_dir / f'pie_seq_{smpl}_readlen{readlen}_{suffix}.pdf'
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(1,2,figsize=(8,3))
        wedges,texts,autotexts = axs[0].pie(
            x=vals,
            colors=colors + ['#808080'],
            autopct=lambda pct: func(pct,vals),
            startangle=90,
            radius=1.8*1.2
        )
        plt.setp(
            autotexts,
            size=8
        )
        axs[0].set_title(smpl,pad=50)
        axs[1].set_axis_off()
        axs[1].legend(
            wedges,val_labels,
            bbox_to_anchor = (0.5,0.,0.5,1),
            bbox_transform = fig.transFigure,
            loc='center right')
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close()
        pdf.close()


def find_sequence_bylength(
    save_dir:Path,
    dict_path_uniq_seqs:dict,
    seq_list:list,
    suffix:str,
    colors=[]
):
    save_dir = save_dir / 'sequence_bylength'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if len(colors)==0:
        _,colors = my._get_spaced_colors(n=len(seq_list))
        colors = colors + ["#808080"]
    
    def func(pct,allvals):
        absolute = int(np.round(pct/100*np.sum(allvals)))
        if pct>1:
            out = f'{pct:.1f}% ({absolute:d})'
        else:
            out = ''
        return out
    
    for smpl,load_path in dict_path_uniq_seqs.items():

        dict_seq = pd.read_csv(load_path,index_col=0,header=0).iloc[:,0].to_dict()
        n_reads = np.sum(list(dict_seq.values()))
        readlen = len(list(dict_seq.keys())[0])

        # calculate how many reads are close to top-hit sequence
        seq_top_grps = {};cnt_top_grps = {};cnt_sum_grps = {}
        thres = 3 / readlen
        for seq_top in seq_list:
            readlen_seq_top = len(seq_top)
            seq_top_grps[seq_top] = []
            cnt_top_grps[seq_top] = []
            cnt_sum_grps[seq_top] = 0
            for seq,cnt in dict_seq.items():
                if readlen_seq_top < readlen:
                    dist = np.min([
                        distance.hamming(list(seq_top),list(seq[i:i+readlen_seq_top]))
                        for i in range(readlen - readlen_seq_top+1)
                    ])
                elif readlen_seq_top > readlen:
                    dist = np.min([
                        distance.hamming(list(seq_top[i:i+readlen]),list(seq))
                        for i in range(readlen_seq_top - readlen+1)
                    ])
                else:
                    dist = distance.hamming(list(seq_top),list(seq))
                ###############
                if seq_top[0] != seq[0]:
                    dist = 1
                ###############
                if dist <= thres:
                    seq_top_grps[seq_top].append(seq)
                    cnt_top_grps[seq_top].append(cnt)
                    cnt_sum_grps[seq_top] += cnt
                    
            pd.DataFrame({
                'count':cnt_top_grps[seq_top],
                '%total reads':np.array(cnt_top_grps[seq_top]) / n_reads*100
            },index=seq_top_grps[seq_top]).to_csv(save_dir / f'uniq_seq_{smpl}_readlen{readlen}_{suffix}_mm3_{seq_top}.csv.gz')
        
        vals = list(cnt_sum_grps.values()) + [n_reads - np.sum(list(cnt_sum_grps.values()))]
        val_labels = seq_list + ['others']
        
        # pie chart
        outfile_name = save_dir / f'pie_seq_{smpl}_readlen{readlen}_{suffix}.pdf'
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(1,2,figsize=(8,3))
        wedges,texts,autotexts = axs[0].pie(
            x=vals,
            colors=colors,
            autopct=lambda pct: func(pct,vals),
            startangle=90,
            radius=1.8*1.2
        )
        plt.setp(
            autotexts,
            size=8
        )
        axs[0].set_title(smpl,pad=50)
        axs[1].set_axis_off()
        axs[1].legend(
            wedges,val_labels,
            bbox_to_anchor = (0.5,0.,0.5,1),
            bbox_transform = fig.transFigure,
            loc='center right')
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close()
        pdf.close()


def reads_indv_mapping(
    save_dir:Path,
    dict_fastq_bam:dict,
    smpl:str,
    read_len_range:list,
    fname:str
):
    save_dir = save_dir / 'reads_indv_mapping'
    if not save_dir.exists():
        save_dir.mkdir()
    
    dict_reads = {}
    for i in range(read_len_range[0],read_len_range[1]):
        dict_reads[i] = {}
        for label in dict_fastq_bam.keys():
            dict_reads[i][label] = []
    
    for label,load_path in dict_fastq_bam.items():
        
        if load_path.suffix == '.gz':

            with gzip.open(load_path,'rt') as f:
                for j,rec in enumerate(SeqIO.parse(f,'fastq')):
                    print(f'\r{j} reads...',end='')
                    dict_reads[ len(rec.seq) ][label].append( rec.id )
                   
        elif load_path.suffix == '.bam':

            infile = pysam.AlignmentFile(load_path)
            for read in infile.fetch(until_eof=True):
                dict_reads[ read.query_length ][label].append(read.qname)

    # FIXME
    dict_plot = {}
    for i in range(read_len_range[0],read_len_range[1]):
        for label in dict_fastq_bam.keys():
            dict_reads[i][label] = list(set( dict_reads[i][label] ))
        dict_plot[i] = {
            'unmapped': len(list(set(dict_reads[i]['total'])-(set(dict_reads[i]['tRNA']) | set(dict_reads[i]['mRNA'])))),
            'tRNA_only': len(list(set(dict_reads[i]['tRNA']) - set(dict_reads[i]['mRNA']))),
            'mRNA_only': len(list(set(dict_reads[i]['mRNA']) - set(dict_reads[i]['tRNA']))),
            'tRNA_mRNA_both': len(list(set(dict_reads[i]['mRNA']) & set(dict_reads[i]['tRNA'])))
        }

    outfile_name = save_dir / f'reads_indv_mapping_{fname}_{smpl}.pdf'
    pdf = PdfPages(outfile_name)
    fig,ax = plt.subplots(1,1,figsize=(10,4))

    df_plot = pd.DataFrame().from_dict(dict_plot,orient='index')
    df_plot.plot.bar(stacked=True,ax=ax,rot=0)

    ax.set_xlabel('read length')
    ax.set_ylabel('count')

    fig.tight_layout()
    fig.savefig(pdf,format='pdf')
    pdf.close()
    print("hoge")    


def extract_reads_5end(
    save_dir:Path,
    dict_fastq_bam:dict,
    readlen:int,
    n_top:int,
    suffix=''
):
    save_dir = save_dir / 'extract_reads_5end'
    if not save_dir.exists():
        save_dir.mkdir()

    def func(pct,allvals):
        absolute = int(np.round(pct/100*np.sum(allvals)))
        if pct>1:
            out = f'{pct:.1f}% ({absolute:d})'
        else:
            out = ''
        return out

    _,colors = my._get_spaced_colors(n=n_top)

    for smpl,load_path in dict_fastq_bam.items():

        # seq_pos = {}
        # for i in range(readlen):
        #     seq_pos[i] = dict(zip(['A','T','C','G','N'],np.zeros(5,dtype=int)))


        smpl = smpl.replace(' ','_')
        seqs = []

        if load_path.suffix == '.gz':

            with gzip.open(load_path,'rt') as f:
                for j,rec in enumerate(SeqIO.parse(f,'fastq')):
                    print(f'\r{j} reads...',end='')
                    # for i in range(readlen):
                    #     seq_pos[i][rec.seq[i]] += 1
                    seqs.append(str(rec.seq[:readlen]))
            
        elif load_path.suffix == '.bam':

            infile = pysam.AlignmentFile(load_path)
            for read in infile.fetch(until_eof=True):
                seqs.append(read.query_sequence[:readlen])

        seq_uniq,cnt_seq = np.unique(seqs,return_counts=True)
        idx = np.argsort(cnt_seq)[::-1]
        pd.DataFrame({
            'count':cnt_seq[idx],
            '%total reads':cnt_seq[idx]/len(seqs)*100
        },index=seq_uniq[idx]).to_csv(save_dir / f'uniq_seq_{smpl}_5end{readlen}_{suffix}.csv.gz')

        # calculate how many reads are close to top-hit sequence
        seq_top_grps = [];cnt_top_grps = [];cnt_sum_grps = []
        thres = 0 / readlen
        j = 0;cnt = 0
        while cnt < n_top:
            seq_top = seq_uniq[idx[j]]
            if cnt>0:
            
                dist = np.min([
                        distance.hamming(list(seq_top),list(seq_top_grps[i][0]))
                        for i in range(cnt)
                    ])
            else:
                dist = 1
            if dist <= thres:
                j += 1
            else:
                seq_top_grp = [];cnt_top_grp = []
                for i in idx:
                    dist = distance.hamming(list(seq_top),list(seq_uniq[i]))
                    if dist <= thres:
                        seq_top_grp.append(seq_uniq[i])
                        cnt_top_grp.append(cnt_seq[i])
                seq_top_grps.append(seq_top_grp)
                cnt_top_grps.append(cnt_top_grp)
                cnt_sum_grps.append(np.sum(cnt_top_grp))

                pd.DataFrame({
                    'count':cnt_top_grp,
                    '%total reads':np.array(cnt_top_grp) / len(seqs)*100
                },index=seq_top_grp).to_csv(save_dir / f'uniq_top{cnt}_seq_{smpl}_5end{readlen}_{suffix}_mm0.csv.gz')
                cnt += 1
        
        vals = cnt_sum_grps + [len(seqs) - np.sum(cnt_sum_grps)]
        val_labels = [x[0] for x in seq_top_grps] + ['others']
        
        # pie chart
        outfile_name = save_dir / f'pie_seq_{smpl}_5end{readlen}_{suffix}.pdf'
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(1,2,figsize=(6,3))
        wedges,texts,autotexts = axs[0].pie(
            x=vals,
            colors=colors + ['#808080'],
            autopct=lambda pct: func(pct,vals),
            startangle=90,
            radius=1.8*1.2
        )
        plt.setp(
            autotexts,
            size=8
        )
        axs[0].set_title(smpl,pad=50)
        axs[1].set_axis_off()
        axs[1].legend(
            wedges,val_labels,
            bbox_to_anchor = (0.5,0.,0.5,1),
            bbox_transform = fig.transFigure,
            loc='center right')
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close()
        pdf.close()



def find_sequence_5end(
    save_dir:Path,
    dict_path_uniq_seqs:dict,
    seq_list:list,
    suffix:str,
    colors=[]
):
    save_dir = save_dir / 'sequence_5end'
    if not save_dir.exists():
        save_dir.mkdir()
    
    if len(colors)==0:
        _,colors = my._get_spaced_colors(n=len(seq_list))
        colors = colors + ["#808080"]
    
    def func(pct,allvals):
        absolute = int(np.round(pct/100*np.sum(allvals)))
        if pct>1:
            out = f'{pct:.1f}% ({absolute:d})'
        else:
            out = ''
        return out
    
    for smpl,load_path in dict_path_uniq_seqs.items():

        dict_seq = pd.read_csv(load_path,index_col=0,header=0).iloc[:,0].to_dict()
        n_reads = np.sum(list(dict_seq.values()))
        readlen = len(list(dict_seq.keys())[0])

        # calculate how many reads are close to top-hit sequence
        seq_top_grps = {};cnt_top_grps = {};cnt_sum_grps = {}
        thres = 0 / readlen
        for seq_top in seq_list:
            readlen_seq_top = len(seq_top)
            seq_top_grps[seq_top] = []
            cnt_top_grps[seq_top] = []
            cnt_sum_grps[seq_top] = 0
            for seq,cnt in dict_seq.items():
                if readlen_seq_top < readlen:
                    dist = np.min([
                        distance.hamming(list(seq_top),list(seq[i:i+readlen_seq_top]))
                        for i in range(readlen - readlen_seq_top+1)
                    ])
                elif readlen_seq_top > readlen:
                    dist = np.min([
                        distance.hamming(list(seq_top[i:i+readlen]),list(seq))
                        for i in range(readlen_seq_top - readlen+1)
                    ])
                else:
                    dist = distance.hamming(list(seq_top),list(seq))
                if dist <= thres:
                    seq_top_grps[seq_top].append(seq)
                    cnt_top_grps[seq_top].append(cnt)
                    cnt_sum_grps[seq_top] += cnt
                    
            pd.DataFrame({
                'count':cnt_top_grps[seq_top],
                '%total reads':np.array(cnt_top_grps[seq_top]) / n_reads*100
            },index=seq_top_grps[seq_top]).to_csv(save_dir / f'uniq_seq_{smpl}_readlen{readlen}_{suffix}_mm3_{seq_top}.csv.gz')
        
        vals = list(cnt_sum_grps.values()) + [n_reads - np.sum(list(cnt_sum_grps.values()))]
        val_labels = seq_list + ['others']
        
        # pie chart
        outfile_name = save_dir / f'pie_seq_{smpl}_5end{readlen}_{suffix}.pdf'
        pdf = PdfPages(outfile_name)
        fig,axs = plt.subplots(1,2,figsize=(8,3))
        wedges,texts,autotexts = axs[0].pie(
            x=vals,
            colors=colors,
            autopct=lambda pct: func(pct,vals),
            startangle=90,
            radius=1.8*1.2
        )
        plt.setp(
            autotexts,
            size=8
        )
        axs[0].set_title(smpl,pad=50)
        axs[1].set_axis_off()
        axs[1].legend(
            wedges,val_labels,
            bbox_to_anchor = (0.5,0.,0.5,1),
            bbox_transform = fig.transFigure,
            loc='center right')
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close()
        pdf.close()


    