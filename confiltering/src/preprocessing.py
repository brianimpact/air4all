from distutils.filelist import findall
import re
import torch
import os
from collections import defaultdict
import warnings
import re
from nltk.corpus import stopwords
import jellyfish
import json
import num2words
import copy

class PreproSUTranscript(object):
    def __init__(self, raw_transcript_path, raw_doc_transcript_path, file_name, remove_su_id, abb, non_unique_abb):
        self.raw_transcript_path = raw_transcript_path
        self.raw_doc_trans_path = raw_doc_transcript_path
        self.study_unit_name = remove_su_id.lower()
        self.checking_abb = copy.deepcopy(abb)
        self.non_unique_abb = non_unique_abb
        self.file_name = file_name + '.source'
        data_transcript = defaultdict()
        self.trans_idx = []
        data_path = os.path.join(self.raw_transcript_path,self.file_name)
        with open(data_path, 'r') as f:
            infos = f.readlines()
            for temp_info in infos:
                content_id, transcript = temp_info.split('\t')
                transcript_ = transcript.split('\n')[0]
                content_id = int(content_id)
                self.trans_idx.append(content_id)
                data_transcript[content_id] = [transcript_]
        self.trans_idx = torch.tensor(self.trans_idx).unsqueeze(1)     
        print('the number of youtube transcripts :',len(data_transcript))
        
        count = -1
        length = 0
        if os.path.exists(self.raw_doc_trans_path+'/'+self.file_name):
            data_path = os.path.join(self.raw_doc_trans_path,self.file_name)
            with open(data_path,'r') as f:
                infos = f.readlines()
                for manual_trans in infos[:-1]:
                    if manual_trans is not None and manual_trans != '' and manual_trans != ' ':
                        data_transcript[count] = [manual_trans]
                        count -= 1
                        length += 1
        print(f'the number of manually collected transcripts : {length}')
        
        self.data_transcript = data_transcript
        self.included_abb,self.included_num,self.included_notchar,self.not_chars,self.label_name_list = preprocess_su_name(remove_su_id,abb)

    # change abbreviation that is in the transcript to the full name
    def chagne_abb2full(self,transcript):
        transcript = transcript.strip()
        transcript = ' ' + transcript + ' '
        special_char = {'++' : 'plus plus', '*' : 'star','ⅰ':'one','ⅱ':'two','λ':'lambda'}
        for i in self.checking_abb.keys():
            if i == 'MAP':
                continue
            i_lower = i.lower()
            words = i_lower +'s'

            transcript = re.sub(f'[^a-zA-Z]{words}[^a-zA-Z]',f' {self.checking_abb[i].lower()} ',transcript)
            transcript = re.sub(f'[^a-zA-Z]{i_lower}[^a-zA-Z]',f' {self.checking_abb[i].lower()} ',transcript)
        if self.included_notchar == True:
            for not_char in self.not_chars:
                if not_char in special_char:
                    transcript = transcript.replace(not_char,' '+ special_char[not_char]+' ')
            transcript = re.sub('\s+',' ',transcript)
        transcript = transcript.strip()
        return transcript
    
    # preprocessing transcripts
    def pre_process(self,transcript):
        transcript = transcript[0]
        transcript = re.sub('\n',' ',transcript)

        for i in self.non_unique_abb.keys():
            max_count = 0
            _count = 0
            max_full_name = ''
            for value in self.non_unique_abb[i]:
                value = value.lower()
                count = len(re.findall(f'[^a-zA-Z]{value}[^a-zA-Z]',transcript.lower()))
                if count > max_count:
                    _count += 1
                    max_count = count
                    max_full_name = value
            if _count >= 1:
                self.checking_abb[i] = [max_full_name]
            else:
                self.checking_abb[i] = [value]

        for k,v in self.checking_abb.items():
            self.checking_abb[k] = v[0]

        if self.included_num == True:
            numbers = re.findall('\d+',transcript)
            for num in numbers:
                char = num2words.num2words(num)
                transcript = re.sub(num,char,transcript)
        transcript = transcript.lower()
        transcript = transcript.replace('/',' ')
        transcript = transcript.replace('-',' ')
        transcript = re.sub('\\.+', '.', transcript)
        transcript = re.sub('\\s+', ' ', transcript)
        transcript = self.fix_misrecognition(' '.join(list(self.label_name_list.keys())),transcript)
        w = transcript.split()
        deletion_list = []
        
        for z in range(len(w) - 1):
            bigram = w[z] + w[z + 1]
            for su_word in self.label_name_list.keys():
                if bigram == su_word or (bigram[-1] == 's' and bigram[:-1] == su_word) or bigram + 's' == su_word:
                    w[z] = su_word
                    deletion_list.append(z + 1)
                elif su_word.upper() in self.checking_abb.keys():
                    su_word_ = su_word.upper()
                    if bigram == self.checking_abb[su_word_].lower() or (bigram[-1] == 's' and bigram[:-1] == self.checking_abb[su_word_].lower()) or bigram + 's' == self.checking_abb[su_word_].lower():
                        w[z] = su_word
                        deletion_list.append(z + 1)
        for z in range(len(w) - 2):
            trigram = w[z] + w[z + 1] + w[z + 2]
            for su_word in self.label_name_list.keys():
                if trigram == su_word or trigram[:-1] == su_word or trigram + 's' == su_word:
                    w[z] = su_word
                    deletion_list.append(z + 1)
                    deletion_list.append(z + 2)
                elif su_word.upper() in self.checking_abb.keys():
                    su_word_ = su_word.upper()
                    if trigram == self.checking_abb[su_word_] or (trigram[-1] == 's' and trigram[:-1] == self.checking_abb[su_word_]) or trigram + 's' == self.checking_abb[su_word_]:
                        w[z] = su_word
                        deletion_list.append(z + 1)
                        deletion_list.append(z + 2)
        for d in sorted(deletion_list)[::-1]:
            del w[d]
        transcript = ' '.join(w)
        transcript = transcript.strip()

        transcript = self.chagne_abb2full(transcript)
        transcript = re.sub("&lt;/?.*?&gt;", " ", transcript)
        transcript = re.sub('[^a-zA-Z]', ' ', transcript)
        transcript = re.sub('\\s+', ' ', transcript)
        transcript = transcript.strip()
        return transcript

    # identify words that may be confused by pronunciation in youtube auto-transcript
    def metaphone_synonyms(self,su_name, transcript):
        # FIND WORDS WITH SAME METAPHONE (PHONETIC ALPHABET) AS THE WORDS IN THE TITLE
        # SUCH (DOMAIN SPECIFIC) WORDS ARE EASILY MISRECOGNIZED WHEN USING AUTOMATIC SPEECH RECOGNITION
        # 
        # su_name: name of the transcript (in our case, name of the corresponding study unit)
        # transcript: transcript (in our case, contents describing the study unit)
        #
        # PREPROCESS TITLE (REMOVE STOPWORDS AND NON-ALPHABETIC CHARACTERS, AND APPEND WITH SINGULAR TERMS)
        su_names = re.sub('\s+', ' ', re.sub('[^a-z ]', ' ', su_name)).strip().split() #L1, L2
        removed = stopwords.words('english')
        su_names = [word for word in su_names if word not in removed]
        new_words = []
        for word in su_names:
            if word.endswith('ces'):
                new_words.append(word[:-3] + 'x')
            elif word.endswith('ies'):
                new_words.append(word[:-3] + 'y')
            elif word.endswith('s'):
                new_words.append(word[:-1])
            if word.endswith('ing'):
                new_words.append(word[:-3] + 'e')
            if word.endswith('ings'):
                new_words.append(word[:-4] + 'e')
        su_names = su_names + new_words
        text = re.sub('\s+', ' ', re.sub('[^a-z0-9 ]', ' ', transcript.lower())).strip().split()
        title_metaphones = [jellyfish.metaphone(t) for t in su_names]
        text_metaphones = [jellyfish.metaphone(t) for t in text]
        similar_metaphones = dict()
        for i, title_metaphone in enumerate(title_metaphones):
            similar = set()
            for j, text_metaphone in enumerate(text_metaphones):
                if title_metaphone == text_metaphone and su_names[i] != text[j]:
                    similar.add(text[j])
            similar_metaphones[su_names[i]] = similar
        return similar_metaphones

    # fix synonyms found by metaphone_synonyms
    def fix_misrecognition(self,source_file, source_line):
        synonyms = self.metaphone_synonyms(source_file, source_line)
        inv_synonyms = dict()
        for k in synonyms.keys():
            syns = synonyms[k]
            for w in syns:
                inv_synonyms[w] = k
        words = source_line.split()
        for i in range(len(words)):
            if re.sub('[^a-z]', '', words[i].lower()) in inv_synonyms.keys():
                new_word = inv_synonyms[re.sub('[^a-z]', '', words[i].lower())]
                if words[i].isupper():
                    new_word = new_word.upper()
                elif words[i][0].isupper():
                    new_word = new_word[0].upper() + new_word[1:]
                if words[i].startswith('('):
                    new_word = '(' + new_word
                if ')' in words[i]:
                    new_word = new_word + ')'
                if words[i].endswith('.'):
                    new_word = new_word + '.'
                if words[i].endswith(','):
                    new_word = new_word + ','
                if words[i].endswith(':'):
                    new_word = new_word + ':'
                if words[i].endswith('?'):
                    new_word = new_word + '?'
                if words[i].endswith('!'):
                    new_word = new_word + '!'
                words[i] = new_word
        return ' '.join(words)
    
    # saving preprocessed transcripts
    def save_processed_transcript(self,out_path):
        print('start preprocess')
        data_transcript = defaultdict()
        for trans_idx, trans in self.data_transcript.items():
            data_transcript[trans_idx] = self.pre_process(trans)
        assert len(list(data_transcript.keys())) !=0, 'There is no transcript'
        data_path = os.path.join(out_path,self.file_name)
        with open(data_path,'w') as f:
            json.dump(data_transcript,f)
        print('done preprocess')

# remove study unit index
def remove_suid(study_unit_name):
    search_name = study_unit_name
    if len(search_name.split('/')) > 1:
        if search_name == '1606) A/B Testing':
            search_name = search_name.replace('/','')
        else:
            search_name = search_name.replace('/',' ')
    search_name  = search_name.replace('–','-')
    
    remove_su_id = search_name.split(') ')
    if len(remove_su_id) >= 3:
        remove_su_id = ') '.join(search_name.split(') ')[1:]).lower()
    elif len(remove_su_id) <3:
        remove_su_id =  search_name.split(') ')[1].lower()

    if remove_su_id == 'None':
        warnings.warn(f'There is no transcripts related to {study_unit_name}')
    return remove_su_id

# preprocessing label name
def preprocess_su_name(su_name,abb):
    included_abb = False
    included_num = False
    included_notchar = False

    for abb_word in abb.keys():
        if len(re.findall(f'[^a-zA-Z]{abb_word.lower()}[^a-zA-Z]',su_name)) >= 1:
            included_abb = True
            break
    if len(su_name.split('/')) > 1:
        if su_name == 'a/b testing':
            su_name = su_name.replace('/','')
        else:
            su_name = su_name.replace('/',' ')

    special_char = {'++' : 'plus plus', '*' : 'star','ⅰ':'one','ⅱ':'two','λ':'lambda'}
    not_chars = re.findall('[^a-zA-Z0-9_,()\' :\-]+',su_name)
    if len(not_chars) >= 1:
        included_notchar = True
    if included_notchar == True:
        for not_char in not_chars:
            if not_char in special_char:
                if not_char == '++':
                    su_name = su_name.replace(not_char,' '+special_char[not_char])
                else:
                    su_name = su_name.replace(not_char,special_char[not_char])
    if su_name == 'b- tree' or su_name == 'b-tree':
        su_name = su_name.replace('-','minus ')

    su_name = ' '+su_name+' '
    for i in abb.keys():
        i_lower = i.lower()
        words = i_lower +'s'
        if len(re.findall(f'[^a-zA-Z]{i_lower}[^a-zA-Z]',su_name)) >=1:
            if len(abb[i]) != 1:
                for full in abb[i]:
                    full = full.lower()
                    if len(re.findall(f'[^a-zA-Z]{full}[^a-zA-Z]',su_name)) >=1:
                        su_name = re.sub(f'[^a-zA-Z]+{i_lower}[^a-zA-Z]+',f' {full} ',su_name)
            else:
                su_name = re.sub(f'[^a-zA-Z]+{i_lower}[^a-zA-Z]+',f' {abb[i][0].lower()} ',su_name)

    if '-' in su_name:
        study_unit_name_ = ''
        for i in su_name.split(' '):
            if '-' in i:
                i_ = ''
                count = 0
                idx_count = 0
                for j in i.split('-'):
                    if len(j) == 1 and len(i.split('-')) < 3 or j in ['non','pre','un','off','per','semi']:
                        count += 1
                    elif idx_count+1 <= len(i.split('-')) - 1:
                        if len(j) + len(i.split('-')[idx_count+1]) < 8 and i not in ['text-to-speech','one-shot learning']:
                            count += 1
                    idx_count += 1
                if count != 0:
                    i_ += ''.join(i.split('-'))
                else:
                    i_ = ' '.join(i.split('-'))
                study_unit_name_ += i_+' '
            else:
                study_unit_name_ += i+' '
        su_name = study_unit_name_.strip()

    numbers = re.findall('\d+',su_name)
    if len(numbers) != 0:
        included_num = True
        for num in numbers:
            char = num2words.num2words(num)
            su_name = re.sub(num,char,su_name)
    su_name = su_name.strip()
    su_name = re.sub('[^a-zA-Z]',' ',su_name)
    su_name = re.sub('\\s+', ' ',su_name)

    check_stopword = ['and','or','what','is','by','with','of','any',
    'from','a','in','to','as','for','without','through','other','may',
    'the','between','on','about','not','why','via','when','at','under','code','s','are']
    su_name_dict = {}
    idx = 0
    for word in su_name.strip().split():
        word = word.lower()
        if word not in check_stopword:
            if word[-1] == 's' and word[:-1] in su_name_dict.keys():
                continue
            elif word + 's' in su_name_dict.keys():
                continue
            su_name_dict[word] = idx
            idx += 1
    return included_abb, included_num, included_notchar, not_chars, su_name_dict