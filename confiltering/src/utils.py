import torch
import pandas as pd
import os
import warnings
from collections import defaultdict
import re

# create a file for the result of filtering
def extract_relevant_idx(out_path,transcript_path,file_su_name,su_names, relevant_idx, cal_freq,included_abb,num_word_threshold=3,low_frequency=0.001):
    data_transcript = []
    total = 0
    data_path = os.path.join(transcript_path,file_su_name)
    with open(data_path, 'r') as f:
        infos = f.readlines()
        for temp_info in infos:
            content_id, _ = temp_info.split('\t')
            content_id = int(content_id)
            data_transcript.append([content_id])
            total += 1
        data_transcript = pd.DataFrame(data_transcript,columns=['content_id'])
    
    non_label = 0
    for word in su_names:
        if word in relevant_idx.keys():
            add_colum_value = []
            relevant_idx = relevant_idx[word]
            for content_id in list(data_transcript['content_id'].values):
                if content_id in relevant_idx:
                    add_colum_value.append(1)
                else:
                    add_colum_value.append(0)
            data_transcript[word] = add_colum_value
        else:
            non_label += 1
            warnings.warn(f'There is any words that related to {word} in {file_su_name}')

    data_transcript.set_index('content_id',inplace=True,drop=True)

    data_transcript['relevance'] = 0
    for idx in data_transcript.index:
        cont = 0
        for word in su_names:
            if word in data_transcript.columns:
                if data_transcript[word][idx] == 1:
                    cont += 1

        if len(su_names) <= num_word_threshold or included_abb == True:
            if cont == len(su_names):
                data_transcript['relevance'][idx] = 1
        else:
            if cont >= len(su_names) - 1:
                data_transcript['relevance'][idx] = 1
    
    # filtering out the low-frequency-label transcript
    cont_frequency_label = []
    filter_trans_idx = set()
    for trans_idx in cal_freq:
        cont_frequency_label.append({trans_idx : cal_freq[trans_idx].values()})
    for i in cont_frequency_label:
        for trans_idx in i.keys():
            valuelist = list(i[trans_idx])
            total_trans_len = valuelist[0]
            total_label_freq = sum(valuelist[1:])
            if total_trans_len == 0:
                filter_trans_idx.add(trans_idx)
            elif (total_label_freq/total_trans_len) < low_frequency:
                filter_trans_idx.add(trans_idx)

    data_transcript_filter_freq = data_transcript
    for i in data_transcript_filter_freq.index:
        if i in filter_trans_idx and data_transcript_filter_freq['relevance'][i] == 1:
            data_transcript_filter_freq['relevance'][i] = 0
    count = 0
    for i in data_transcript_filter_freq.index:
        if data_transcript_filter_freq['relevance'][i] == 1:
            count+=1
    if count >= 5:
        data_transcript = data_transcript_filter_freq
    else:
        data_transcript = data_transcript
    print(f'{list(data_transcript.index)}')
    cont_rele_trans = sum(data_transcript['relevance'])
    if cont_rele_trans == 0:
        warnings.warn(f'Thre is no relevant transcript in {file_su_name}')
       
    print(f'the number of related transcripts : {cont_rele_trans}')
    cont_rele_trnas_idx = set()
    for i in data_transcript.index:
        if data_transcript['relevance'][i] == 1:
            cont_rele_trnas_idx.add(i)
    print(f'Related Content_id : {list(cont_rele_trnas_idx)}')
    out_path_su = os.path.join(out_path,file_su_name)
    data_transcript.to_csv(out_path_su)

# making category vocabulary file
def saving_category_vocabulary_file(temp_dir,file_su_name,category_vocab):
    category_data_path = temp_dir+'/category_vocabulary'
    vocab_loader_name="category_vocab.pt"
    vocab_save_file = os.path.join(category_data_path+'/', f"{file_su_name}_"+vocab_loader_name)
    torch.save(category_vocab, vocab_save_file)
    print("========================================================================================================================================================================================================")

# making abbreviation list
def making_abb_list(abb_path):
        abbreviation_list = pd.read_excel(abb_path,names=['non_abb','abb1','abb2'])
        abb = defaultdict(list)
        for i in abbreviation_list.index:
            abb1 = abbreviation_list.loc[i,'abb1'].upper()
            if abb1 in abb.keys():
                abb[abb1].append(abbreviation_list.loc[i,'non_abb'])
            else:
                abb[abb1] = [abbreviation_list.loc[i,'non_abb']]

            if type(abbreviation_list.loc[i,'abb2']) == str:
                abb2 = abbreviation_list.loc[i,'abb2'].upper()
                if abb2 in abb.keys():
                    abb[abb2].append(abbreviation_list.loc[i,'non_abb'])
                else:
                    abb[abb2] = [abbreviation_list.loc[i,'non_abb']]

        for i in abb.keys():
            if len(i.split('/')) >1:
                old_abb = i
                old_full_name_list = abb[i]
                new_full_name_list = []
                new_abb = re.sub('/',' ',old_abb)
                abb.pop(i)
                for full_name in old_full_name_list:
                    full_name = re.sub('/',' ',full_name)
                    new_full_name_list.append(full_name)
                abb[new_abb] = new_full_name_list
        abb_sorted = {k:v for k,v in sorted(abb.items(), key = (lambda x : len(x[0])), reverse=True)}

        non_unique_abb = defaultdict(list)
        for i in abb_sorted.keys():
            if len(abb_sorted[i]) > 1:
                non_unique_abb[i] = abb_sorted[i]
        return abb_sorted, non_unique_abb