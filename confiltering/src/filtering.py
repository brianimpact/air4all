from email.policy import default
import warnings
import torch
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict

class Filtering:
    def __init__(self, data, temp_dir, included_abb, category_vocab_size):
        self.temp_dir = temp_dir 
        self.data = data
        self.tokenizer = self.data.tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.trans_trunc_len = self.data.trans_trunc_len
        self.included_abb = included_abb
        self.category_vocab_size = category_vocab_size

    # remove the stopword in the category vocabulary and sorting
    def filter_stopwords(self,category_words_freq):
        sorted_dict = {k: v for k, v in sorted(category_words_freq.items(), key = (lambda x : x[1]), reverse = True)[:100]}
        category_vocab = np.array(list(sorted_dict.keys()))
        stopwords_vocab = stopwords.words('english')
        delete_idx = []
        for i, word_id in enumerate(category_vocab):
            word = self.inv_vocab[word_id]
            if word in self.data.su_name_dict.keys():
                continue
            if not word.isalpha() or len(word) == 1 or word in stopwords_vocab:
                delete_idx.append(i)
        category_vocab = np.delete(category_vocab, delete_idx)
        category_vocab = category_vocab[:self.category_vocab_size]
        return category_vocab

    # making category vocabulary and filtering each document
    def making_catevoca_and_classification(self, model, top_pred_num=50, match_threshold=30, doc_weight=2):
        result = self.data.make_tensordataset(self.data.label_name_data)
        if result == None:
            return None
        else:   
            datasets, self.words = result[0], result[1]
            word_sep_dataset = defaultdict(list)
            lengths = self.trans_trunc_len
            self.cal_freq = defaultdict()
            for z in range(len(self.words)):
                label_name_dataset_loader = datasets[z]
                cont = 0
                word = self.words[z]
                for trans_idx in lengths.keys():
                    length = lengths[trans_idx]
                    label_name_data = label_name_dataset_loader[cont:cont+length][:-1]
                    sep_attr = defaultdict(list)
                    if label_name_data[0] != None:
                        sep_attr['transcript_idx'] = trans_idx
                        sep_attr['input_ids'] = torch.tensor(label_name_data[0]).reshape(1,-1,512)
                        sep_attr['attention_masks'] = torch.tensor(label_name_data[1]).reshape(1,-1,512)
                        sep_attr['label_idx_pos'] = torch.tensor(label_name_data[2]).reshape(1,-1,512)
                        word_sep_dataset[word].append([sep_attr['input_ids'], sep_attr['attention_masks'], sep_attr['label_idx_pos'], sep_attr['transcript_idx']])
                        cont = cont + length

                        if trans_idx > 0:
                            if trans_idx not in self.cal_freq.keys():
                                length_for_freq = 0
                                for i in sep_attr['attention_masks']:
                                    for j in i:
                                        length_for_freq += sum(j > 0)
                                self.cal_freq[trans_idx] = defaultdict()
                                try:
                                    self.cal_freq[trans_idx]['total_length'] = length_for_freq.item()
                                except:
                                    self.cal_freq[trans_idx]['total_length'] = length_for_freq
                            word_label_freq = 0
                            for i in sep_attr['label_idx_pos']:
                                for j in i:
                                    word_label_freq += sum(j > 0)
                            try:
                                self.cal_freq[trans_idx][word] = word_label_freq.item()  
                            except:
                                self.cal_freq[trans_idx][word] = word_label_freq                  
                    else:
                        return None
        self.category_words_freq = {word: defaultdict(float) for word in self.words}
        self.document_category_word_freq = {word: defaultdict(float) for word in self.words}
        self.only_youtube_category_word_freq = {word: defaultdict(float) for word in self.words}
        self.category_vocab = {}
        self.only_manually_cate_vocab = {}
        self.only_youtube_cate_vocab = {}
        sorted_res_dict = defaultdict(dict)
        for z in range(len(self.words)):
            word = self.words[z]
            one_word_dataset = word_sep_dataset[word]
            for batch in one_word_dataset:
                trans_idx = batch[3]
                if trans_idx < 0:
                    weight = doc_weight
                else:
                    weight = 1
                with torch.no_grad():
                    input_ids = batch[0]
                    input_mask = batch[1]
                    label_pos = batch[2]
                    input_ids_truncated = input_ids.squeeze(0)
                    input_mask_truncated = input_mask.squeeze(0)
                    label_pos_truncated = label_pos.squeeze(0)
                    trans_idx = batch[3]
                    match_idx_label = label_pos_truncated >= 0
                    predictions = model(input_ids_truncated,
                                        token_type_ids=None,
                                        attention_mask=input_mask_truncated)
                    _, sorted_res = torch.topk(predictions[match_idx_label], top_pred_num, dim=-1)
                    sorted_res_dict[trans_idx][word] = sorted_res

                    # for both manually collected transcript and youtube transcript category vocabulary
                    for word_list in sorted_res: 
                        for word_id in word_list: 
                            self.category_words_freq[word][word_id.item()] += (1*weight)
                    # only for manually collected transcript category vocabulary
                    if trans_idx < 0:
                        for word_list in sorted_res: 
                            for word_id in word_list:
                                self.document_category_word_freq[word][word_id.item()]
                    # only for youtube transcript category vocabulary
                    elif trans_idx > 0:
                        for word_list in sorted_res: 
                            for word_id in word_list:
                                self.only_youtube_category_word_freq[word][word_id.item()]

            category_words = self.filter_stopwords(self.category_words_freq[word])
            self.category_vocab[word] = category_words
            
            doc_category_words = self.filter_stopwords(self.document_category_word_freq[word])
            self.only_manually_cate_vocab[word] = doc_category_words

            only_youtube_category_words = self.filter_stopwords(self.only_youtube_category_word_freq[word])
            self.only_youtube_cate_vocab[word] = only_youtube_category_words

        #classification
        relevant_idx = defaultdict(list)
        label_predict_voca = defaultdict()
        check_sorted_dict  = defaultdict(dict)
        for trans_idx in sorted_res_dict:
            if trans_idx < 0:
                continue
            else:
                label_predict_voca[trans_idx] = defaultdict()
                for word in sorted_res_dict[trans_idx]:
                    label_predict_voca[trans_idx][word] = defaultdict()
                    for one_label in sorted_res_dict[trans_idx][word]:
                        for predict_word_id in one_label:
                            if predict_word_id.item() in label_predict_voca[trans_idx][word]:
                                label_predict_voca[trans_idx][word][predict_word_id.item()] += 1
                            else:
                                label_predict_voca[trans_idx][word][predict_word_id.item()] = 1
                    check_sorted_dict = {k: v for k, v in sorted(label_predict_voca[trans_idx][word].items(), key = (lambda x : x[1]), reverse = True)[:50]}
                    cont = 0
                    for word_id in list(check_sorted_dict.keys()):
                        if word_id in list(self.category_vocab[word]):
                            cont += 1
                    if cont >= match_threshold:
                        relevant_idx[word].append(trans_idx)

        for label, category_vocab in self.category_vocab.items():
            print(f"{label}'s category vocabulary: {[self.inv_vocab[w] for w in category_vocab]}")
        result = defaultdict()
        result['label_words'] = self.words
        result['relevant_idx'] = relevant_idx
        result['category_vocab'] = self.category_vocab
        result['cal_freq'] = self.cal_freq
        result['only_manually_cate_vocab'] = self.only_manually_cate_vocab
        result['only_youtube_cate_vocab'] = self.only_youtube_cate_vocab
        return result
    