# Content Filtering

Content filtering module filters irrelevant english Youtube videos that are not regarded as study material of each study unit.

By automatically filtering the contents, [aistudy.guide](aistudy.guide) is expected to secure reliability without manual effort. 

When filtering, there is no training data for the new study unit, so unsupervised learning should be adopted.

To be specific, we implemented the expansion of [Label-Name-Only Text Classification: LOTClass](https://aclanthology.org/2020.emnlp-main.724.pdf) as our content filtering module.

### 1. Configuration

Arguments of `run.py` are as follows:

- `--temp_dir`: directory of saving the results (str)
- `--raw_transcript_path`: directory of youtube raw transcripts (str)
- `--raw_doc_transcript_path`: directory of raw manually collected document transcripts (str)
- `--transcript_path`: directory of the preprocessed transcripts (str)
- `--abb_path`: directory of study unit's abbreviation excel file (str)
- `--pretrained_lm`: pretrained language model (str)
- `--top_pred_num`: language model MLM top prediction cutoff (int)
- `--category_vocab_size`: size of category vocabulary for each study unit (int)
- `--match_threshold`: matching threshold whether each transcript is relevant to study unit (int)
- `--low_frequency`: criteria for filtering out lower-frequency of label (int)
- `--doc_weight`: define how much affect manually collected document to create category vocabulary (int)
- `--truncated_len`: length that documents are padded/truncated to (int) (e.g. one unit means 512 length of tokens; if set to 100, 512*100 lengths are tokenized)
- `--num_word_threshold` : how many words are related to classify whether transcript is filtered (int)
- `--making_category_vocab_file`: save category vocabulary file or not (boolean)
- `--out_path` : directory of saving content index with relevance column whose value one indicates filter-in or zero indicates filter-out (str)

### 2. Data

#### 2.1 Study Unit List & Abbreviation List

- `study_unit_list` should be in `temp_dir` and this file has the format as follows:
`[<study_unit_id>) <study_unit_name>, ...]`

    e.g. `['0) Scalars, Vectors,...nd Tensors', '1) Matrix Multiplication', ..., '1968) Imitation Learning']`

- `abbreviation_list.xlsx` should be in `abb_path` and this file has the format as follows:  
`<first column - non_abb> : full study unit name which has abbreviaton`  
`<second column - abb1> : first abbreviation of full study unit name`  
`<third column - abb2> : second abbreviation of full study unit name, if it exists, or stay empty`  

    e.g. non_abb, abb1, abb2  
        `Multinoulli Distribution, Categorical Distribution, NaN`  
        ⋮  
        `Stochastic Gradient Descent, SGD, Stochastic GD'  
        ⋮  

#### 2.2 Input/Output

- Input : youtube video transcript or manually collected website posting (below, both are referred to as transcripts) of each study unit

- Output : CSV file which includes whether to filter for each content index for study unit

- transcripts should be in `transcript_path`, where each file in `transcript_path` has filename as follows:
`<study_unit_id>) <study_unit_name>.source`
    
    e.g. `0) Scalars, Vectors, Matrices and Tensors.source`

- In a file, a dictionary consists of content indices and corresponding transcripts. If the content is manually collected content, its content index will be a negative content index, while the youtube transcript has a positive content index.
    
    e.g. `{1: "welcome back to this series...", 2: "in this video we will",...,-2: "matrices and vectors linear algebra review",...}`

- Results of `run.py` are saved in `out_path` with filename like the files in `data_path`, but with different extension.
    
    e.g. `0) Scalars, Vectors, Matrices and Tensors.csv`

- Results are CSV files, which include content indices, include whether each word consisting of study unit name is relevant to the corresponding content, and comprehensively whether to filter for each content for a study unit.

`0 : non-relevant, 1 : relevant`  
    e.g. `1) Matrix Multiplication`  

        - column :  
            content_id, matrix, multiplication, relevance  

        - value :  
            10, 1, 0, 0  
            11, 0, 0, 0  
            ⋮  
            898, 1, 1, 1  
            1075, 1, 0, 0   

### 3. Training

- This task should be unsupervised learning, so training is not required.

- Use pre-trained language model MLM to form the category vocabulary.

### 4. Requirements

The following libraries are required for running content filtering module: `numpy, pandas, torch, nltk, tqdm, transformers, jellyfish, num2words`.

### 5. References

[SCIBERT: A Pretrained Language Model for Scientific Text (Beltagy et al., EMNLP 2019)](https://aclanthology.org/D19-1371.pdf)

[Label-Name-Only Text Classification: LOTClass](https://aclanthology.org/2020.emnlp-main.724.pdf)

Content filtering module is implemented based on [LOTClass](https://github.com/yumeng5/LOTClass) and we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).