from distutils.filelist import findall
import torch
import os
from torch.utils.data import TensorDataset
from collections import defaultdict
import warnings
from nltk.corpus import stopwords
import json

class SUdataset(object):
    def __init__(self,temp_dir,transcript_path,file_su_name,su_name_dict, tokenizer, truncated_len):
        self.temp_dir = temp_dir
        self.su_name_dict = su_name_dict
        self.file_su_name = file_su_name
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.max_len = 512
        self.truncated_len = truncated_len
        data_path = os.path.join(transcript_path,file_su_name + '.source')
        with open(data_path,'r') as f:
            self.data_transcript = json.load(f)
        self.read_data(self.temp_dir)

    def read_data(self, dataset_dir):
        file_su_name = self.file_su_name
        label_name_loader = 'loader_data/'+ file_su_name +'_'+ "label_name_data.pt"
        self.label_name_data = self.searching_label_name(dataset_dir, label_name_loader_name=label_name_loader)

    # check the label existence and save the label's position for each transcript
    def find_label_name_in_doc(self, transcript):
        transcript = self.tokenizer.tokenize(transcript)
        tokenlen_transcript = len(transcript) // 512 + 1
        if tokenlen_transcript >= self.truncated_len:
            trunc_transcript_len = self.truncated_len
        else:
            trunc_transcript_len = tokenlen_transcript
        label_idx = {}
        for i in self.su_name_dict.keys():
            label_idx[i] = -1 * torch.ones(trunc_transcript_len * 512, dtype = torch.long)
        new_doc = []
        chunks = []
        idx = 0
        cont = 0
        for i,chunk in enumerate(transcript):
            cont += 1
            chunks.append(chunk[2:] if chunk.startswith("##") else chunk)
            if idx >=self.truncated_len*512:
                break
            if i == len(transcript) - 1 or not transcript[i + 1].startswith("##"):
                word = ''.join(chunks)
                if word in self.su_name_dict.keys():  
                    label_idx[word][idx] = 1
                elif word + 's' in self.su_name_dict.keys(): 
                    word = word + 's'
                    label_idx[word][idx] = 1
                elif (word[-1]=='s' and word[:-1] in self.su_name_dict.keys()):
                    word = word[:-1]
                    label_idx[word][idx] = 1
                    if word not in self.vocab: 
                        chunks = [self.tokenizer.mask_token]
                new_chunk = ''.join(chunks)
                if new_chunk != self.tokenizer.unk_token:
                    idx += len(chunks)
                    new_doc.append(new_chunk)
                chunks = []
        label_idx_pos = {}
        for i in label_idx.keys():
            if (label_idx[i] >= 0).any():
                label_idx_pos[i] = label_idx[i]
            else:
                label_idx_pos[i] = None
        count = 0
        for i in label_idx_pos:
            if label_idx_pos[i] != None:
                count += 1
        if count > 0:
            return ' '.join(new_doc), label_idx_pos, trunc_transcript_len                                         
        else:
            return None

    # encoding the transcript
    def trans_label_name2tensor(self,transcripts):
        new_documents = {}
        label_name_idx = defaultdict(list)
        self.trans_trunc_len = {}
        for doc_idx,doc in transcripts.items():
            result = self.find_label_name_in_doc(doc) 
            if result is not None: 
                self.trans_trunc_len[doc_idx] = result[2]
                new_documents[doc_idx] = result[0] 
                for i in result[1]: 
                    if result[1][i] is not None:
                        label_name_idx[i].append(result[1][i])
                    else:
                        label_name_idx[i].append(-1 * torch.ones(self.trans_trunc_len[doc_idx] * 512, dtype = torch.long))  
        self.len_new_doc = len(new_documents)
        if len(new_documents) > 0:
            for i in self.su_name_dict.keys():
                label_name_idx[i] = torch.cat(label_name_idx[i], dim=0).reshape(-1,512)
            
            result_dict = {'input_ids' : torch.tensor([]) , 'attention_mask' : torch.tensor([])}
            for doc_idx, doc in new_documents.items():
                encoded_dict = self.tokenizer.encode_plus(doc, add_special_tokens= False, max_length= self.trans_trunc_len[doc_idx]*512, padding ='max_length', return_attention_mask=True, truncation=True, return_tensors='pt' )
                input_ids_with_label_name = encoded_dict['input_ids']
                attention_masks_with_label_name = encoded_dict['attention_mask']
                truncated_input_ids = input_ids_with_label_name.reshape(-1,512)
                truncated_attention_masks = attention_masks_with_label_name.reshape(-1,512)
                result_dict['input_ids'] = torch.cat([result_dict['input_ids'], truncated_input_ids], dim = 0) 
                result_dict['attention_mask'] = torch.cat([result_dict['attention_mask'], truncated_attention_masks], dim = 0) 

            result_dict['input_ids'] =  result_dict['input_ids'].long()
            result_dict['attention_mask'] = result_dict['attention_mask'].long()
            transcript_idx_label=[]
            for i in self.trans_trunc_len.keys():
                for _ in range(self.trans_trunc_len[i]):
                    transcript_idx_label.append(i)
            return result_dict['input_ids'], result_dict['attention_mask'], label_name_idx, transcript_idx_label
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
            transcript_idx_label = torch.ones(0, self.max_len, dtype=torch.long)
        
            return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx, transcript_idx_label
                                                                                        
    # searching label names in transcripts
    def searching_label_name(self,dataset_dir, label_name_loader_name):  
        loader_file = os.path.join(dataset_dir, label_name_loader_name)
        if os.path.exists(loader_file):
            print(f"Loading transcripts : {loader_file}")
            transcripts = self.data_transcript
            self.trans_trunc_len = {}
            for idx in transcripts.keys():
                transcript = transcripts[idx]
                int_idx = int(idx)
                transcript = self.tokenizer.tokenize(transcript)
                tokenlen_transcript = len(transcript) // 512 + 1
                if tokenlen_transcript >= self.truncated_len:
                    trans_trunc_len = self.truncated_len
                else:
                    trans_trunc_len = tokenlen_transcript
                self.trans_trunc_len[int_idx] = trans_trunc_len
            self.label_name_data = torch.load(loader_file)
        else:
            print(f"Reading texts : {self.file_su_name}")
            transcripts = {int(idx) : transcript for idx,transcript in self.data_transcript.items()}
            results = self.trans_label_name2tensor(transcripts)
            print("Searching label names in the transcripts.")
            input_ids_with_label_name = results[0]
            attention_masks_with_label_name = results[1]
            transcript_idx_label = [int(idx) for idx in results[3]]
            transcript_idx_label = torch.tensor(transcript_idx_label)
            label_name_pos = results[2]

            self.label_name_data = {"input_ids": input_ids_with_label_name, "attention_masks": attention_masks_with_label_name, "labels": label_name_pos,'transcript_idx' : transcript_idx_label}
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            print(f"Saving texts with label names into {loader_file}")
            torch.save(self.label_name_data, loader_file)
        return self.label_name_data

    # make tensordataset
    def make_tensordataset(self, data_dict):
        data_loaders = []
        if data_dict['input_ids'].shape[0] == 0:
            return None
        else:
            words = list(data_dict["labels"].keys())
            for word in words:        
                dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"][word],data_dict['transcript_idx'])
                data_loaders.append(dataset)
            return data_loaders, words