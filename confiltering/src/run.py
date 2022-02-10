from xml.etree.ElementInclude import include
import os
from tqdm import tqdm
import pymysql
import argparse
from torch.nn import DataParallel
import torch
import json
from utils import extract_relevant_idx,saving_category_vocabulary_file,making_abb_list
from preprocessing import PreproSUTranscript,preprocess_su_name,file_name
from model import SUClassModel
from dataset import SUdataset
from filtering import Filtering

def run(args):
    study_unit_list_path = os.path.join(args.temp_dir,'study_unit_list')
    with open(study_unit_list_path,'r') as f:
        file_list = json.load(f)

    #create abbrevation
    abb, non_unique_abb = making_abb_list(args.abb_path)

    #save prerprocessed data
    for name in tqdm(file_list):
        file_su_name = file_name(args.raw_transcript_path,name)
        if os.path.exists(f'{args.transcript_path}/{file_su_name}') == False:
            print(f'{file_su_name.split(".source")[0]}')
            PreproSUTranscript(raw_transcript_path=args.raw_transcript_path,raw_doc_transcript_path=args.raw_doc_transcript_path,
            study_unit_name=name,abb=abb,non_unique_abb=non_unique_abb).save_processed_transcript(args.transcript_path)
    
    #define model
    model = SUClassModel.from_pretrained(args.pretrained_lm, output_attentions=False, output_hidden_states=False, num_labels=2)
    if torch.cuda.is_available():
        model = DataParallel(model.cuda())
    model.eval()
    #filtering
    for name in tqdm(file_list):
        file_su_name = file_name(args.transcript_path,name)
        if os.path.exists(f'{args.temp_dir}/category_vocabulary/{file_su_name}_category_vocab.pt') == False:
            print(f'===={file_su_name.split(".source")[0]}====')
            included_abb,_,_,_,preprocessed_label_name = preprocess_su_name(name,abb)
            #create dataset
            data = SUdataset(args.temp_dir,args.transcript_path,file_su_name,preprocessed_label_name, args.pretrained_lm,args.truncated_len)
            #filtering
            words, relevant_idx, category_vocab, cal_freq = Filtering(data,args.temp_dir,included_abb,
                                            args.category_vocab_size).making_catevoca_and_classification(model,top_pred_num=args.top_pred_num,
                                            match_threshold=args.match_threshold,doc_weight=args.doc_weight,num_word_threshold=args.num_word_threshold,low_frequency=args.low_frequency)
            #saving results
            extract_relevant_idx(args.out_path,args.transcript_path,file_su_name,words,relevant_idx,cal_freq,included_abb)
            if args.saving_category_vocab_file == True:
                saving_category_vocabulary_file(args.temp_dir,file_su_name,category_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--temp_dir',default=None,help='directory of saving the results')
    parser.add_argument('--raw_transcript_path',default=None,help='directory of raw youtube transcripts')
    parser.add_argument('--raw_doc_transcript_path',default=None,help='directory of raw manually collected document transcripts')
    parser.add_argument('--transcript_path',default=None,help='directory of the preprocessed transcripts')
    parser.add_argument('--abb_path',default='./dataset/abbreviation_list.xlsx',help='load study unit"s abbreviation excel file which should be defined before run')
    parser.add_argument('--pretrained_lm',default='allenai/scibert_scivocab_uncased',help='pretrained model')
    parser.add_argument('--top_pred_num',default=50,help='language model MLM top prediction cutoff')
    parser.add_argument('--category_vocab_size',default=50,help='size of category vocabulary for each study unit')
    parser.add_argument('--match_threshold',default=30,help='matching threshold whether each transcript is relevant to study unit')
    parser.add_argument('--low_frequency',default=0.001,help='criteria for filtering out lower-frequency of label')
    parser.add_argument('--doc_weight',default=2,help='define how much affect manually collected document to create category vocabulary')
    parser.add_argument('--truncated_len',default=100,help='length that documents are padded/truncated to, one unit means 512 length of tokens, 100 -> 512*100')
    parser.add_argument('--num_word_threshold',default=3,help='how many words are related to classify whether transcript is relevant')
    parser.add_argument('--saving_category_vocab_file',default=True,help='saving category vocabulary file or not')
    parser.add_argument('--out_path',default='/home/seokilee/air4all/confiltering/dataset/filtering_doc',help='directory of saving content index with relevance column which indicates filter-in or filter-out')
    args = parser.parse_args()
    print(args)

    run(args)