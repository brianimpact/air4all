from xml.etree.ElementInclude import include
import os
from tqdm import tqdm
import argparse
from torch.nn import DataParallel
import torch
import json
from utils import extract_relevant_idx, saving_category_vocabulary_file, making_abb_list, remove_slash, remove_suid
from preprocessing import PreproSUTranscript, preprocess_su_name
from model import SUClassModel
from dataset import SUdataset
from filtering import Filtering


def preprocess_transcript(args):
    study_unit_list_path = os.path.join(args.temp_dir, 'study_unit_list')
    with open(study_unit_list_path, 'r') as f:
        file_list = json.load(f)
    #create abbrevation
    abb, non_unique_abb = making_abb_list()

    #save prerprocessed data
    for name in tqdm(file_list):
        remove_su_id = remove_suid(name)
        file_name = remove_slash(name)
        if os.path.exists(f'{args.transcript_path}/{file_name}.source') == False:
            print(f'{file_name}')
            PreproSUTranscript(raw_transcript_path=args.raw_transcript_path, raw_doc_transcript_path=args.raw_doc_transcript_path, file_name = file_name,
            remove_su_id=remove_su_id, abb=abb, non_unique_abb=non_unique_abb).save_processed_transcript(args.transcript_path)

def run(args):
    study_unit_list_path = os.path.join(args.temp_dir, 'study_unit_list')
    with open(study_unit_list_path,'r') as f:
        file_list = json.load(f)
    #create abbrevation
    abb,_ = making_abb_list()
    #define model
    model = SUClassModel.from_pretrained(args.pretrained_lm, output_attentions=False, output_hidden_states=False, num_labels=2)
    if torch.cuda.is_available():
        model = DataParallel(model.cuda())
    model.eval()
    #filtering
    for name in tqdm(file_list):
        remove_su_id = remove_suid(name)
        file_name = remove_slash(name)
        if os.path.exists(f'{args.temp_dir}/category_vocabulary/{file_name}_category_vocab.pt') == False:
            print(f'{file_name}')
            included_abb, _, _, _, preprocessed_label_name = preprocess_su_name(remove_su_id,abb)
            print('words that consist label :', preprocessed_label_name)
            #create dataset
            data = SUdataset(args.temp_dir, args.transcript_path, file_name, preprocessed_label_name, args.pretrained_lm, args.truncated_len)
            #filtering
            label_words, relevant_idx, category_vocab, cal_freq, only_manually_cate_vocab, only_youtube_cate_vocab = Filtering(data,args.temp_dir, included_abb,
                                            args.category_vocab_size).making_catevoca_and_classification(model, top_pred_num=args.top_pred_num,
                                            match_threshold=args.match_threshold, doc_weight=args.doc_weight)
            #saving results
            extract_relevant_idx(args.out_path, args.transcript_path, file_name, label_words, relevant_idx, cal_freq, included_abb, num_word_threshold=args.num_word_threshold, low_frequency=args.low_frequency)
            if args.saving_category_vocab_file == True:
                saving_category_vocabulary_file(args.temp_dir, file_name, category_vocab, only_manually_cate_vocab, only_youtube_cate_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('--temp_dir', default=None, help='path of saving the results')
    parser.add_argument('--raw_transcript_path',default= None,help='path of raw youtube transcripts')
    parser.add_argument('--raw_doc_transcript_path',default= None, help='path of raw manually collected document transcripts')
    parser.add_argument('--transcript_path',default=None,help='path of the preprocessed transcripts')
    parser.add_argument('--pretrained_lm',default='allenai/scibert_scivocab_uncased',help='pretrained model')
    parser.add_argument('--top_pred_num', default=50, type=int, help='language model MLM top prediction cutoff')
    parser.add_argument('--category_vocab_size',default=50, type=int, help='size of category vocabulary for each study unit')
    parser.add_argument('--match_threshold',default=30, type=int, help='matching threshold whether each transcript is relevant to study unit')
    parser.add_argument('--low_frequency',default=0.001, type=float, help='criteria for filtering out lower-frequency')
    parser.add_argument('--doc_weight', default=2, type=int, help='define how much affect manually collected document to create category vocabulary')
    parser.add_argument('--truncated_len',default=100, type=int, help='length that documents are padded/truncated to, one unit means 512 length of tokens, 100 -> 512*100')
    parser.add_argument('--num_word_threshold',default=3, type=int, help='how many words are related to classify whether transcript is relevant')
    parser.add_argument('--saving_category_vocab_file',default=True,help='save category vocabulary file')
    parser.add_argument('--out_path',default=None,help='path of saving filter-in content index')
    args = parser.parse_args()
    
    print(args)
    run(args)