# Study Unit Summarization

Study unit summarization module generates the summarized text (or the abstraction) of each study unit based on the corresponding contents.

Such summaries provided by [aistudy.guide](aistudy.guide) help users decide whether to study each study unit or not.

Study unit summarization proceeds in the following two steps: content summarization and multi-document summarization.


## 1. Content Summarization
The first phase of the study unit summarization module is content summarization.

In order to preserve information in long contents, sliding windows of sizes 1,024, 512, and 256 are used to summarize the tokens in the windows, where each window's stride is half of its window size.

SciTLDR, a pretrained extreme summarization model ([download](https://github.com/allenai/scitldr)), is adopted for content summarization, where extreme summarization refers to the task of generating a single sentence that abstracts the given document.

Many contents collected for [aistudy.guide](aistudy.guide) are from YouTube, and while their automatically generated transcripts had high quality in general, they shared the following weaknesses.

First, sentences without most of the punctuation marks (including commas, exclamation marks, question marks, and full stops) were generated.

Second, domain-specific terms were incorrectly transcribed due to their low prior.

Thus, in `scitldr.py`, domain-specific terms are restored using metaphone-based correction and each content is summarized into separated sentences so that they can be used for unsupervised extractive multi-document summarization.

### 1.1. Arguments

Arguments of `scitldr.py` are as follows:

- `batch_size`: minibatch size for text summarization (int)
- `data_path`: directory of the contents (str)
- `out_path`: directory of the summaries (str)
- `checkpoint_dir`: directory of the pretrained checkpoints (str)
- `checkpoint_name`: name of the pretrained checkpoints (str) (recommended to use `scitldr_catts.tldr-ao.pt`)
- `beam`: beam size for better candidate search (int) (see [wikipedia page on beam search](https://en.wikipedia.org/wiki/Beam_search))
- `lenpen`: penalty term for short candidates in beam search (float)
- `max_len_b`: upper limit on number of tokens in summary (int)
- `min_len`: lower limit on number of tokens in summary (int)
- `no_repeat_ngram_size`: term to prevent repeating n-grams (int) (e.g. if set to 3, repeating trigrams are not generated)

For more information on `beam, lenpen, max_len_b, min_len, no_repeat_ngram_size`, see [documentation on fairseq's BART generation](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#Generation).

### 1.2. Data

- As described above, contents (to be summarized) should be in `data_path`, where each file in `data_path` has filename as follows:
`<study_unit_id>) <study_unit_name>.txt`
    
    e.g. `0) Scalars, Vectors, Matrices and Tensors.txt`

- Each line in a file consists of a content index and corresponding text, separated by a tab.
    
    e.g. `0\tThis is an example document....`

- Results of `scitldr.py` are saved in `out_path` with filename like the files in `data_path`, but with different extension.
    
    e.g. `0) Scalars, Vectors, Matrices and Tensors.out`

### 1.3. Requirements
`scitldr.py` requires the following libraries: `torch, fairseq, tqdm`.

### 1.4. References
[TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP 2020)](https://aclanthology.org/2020.findings-emnlp.428.pdf)

[Hanging on the Metaphone (Philips, Computer Language Magazine 1990)](https://en.wikipedia.org/wiki/Metaphone)

## 2. Multi-Document Summarization

The second phase of the study unit summarization module is multi-document summarization, performed in an unsupervised extractive fashion.

`textrank.py` provides various methods for text representation and extractive summarization.

### 2.1. Arguments

Arguments of `textrank.py` are as follows:

- `sentence_embedding`: specifies how to represent each sentence (str)
    
    - `specter`: [pretrained document-level embedding of scientific papers](https://aclanthology.org/2020.acl-main.207.pdf)
    - `bert11avg`: representation obtained by average-pooling outputs of 11-th layer in [SciBERT](https://aclanthology.org/D19-1371.pdf)
    - `bert11max`: representation obtained by max-pooling outputs of 11-th layer in [SciBERT](https://aclanthology.org/D19-1371.pdf)
    - `bert11cls`: representation obtained from \[CLS\] token of 11-th layer in [SciBERT](https://aclanthology.org/D19-1371.pdf)
    - `bert12avg`: representation obtained by average-pooling outputs of 12-th layer in [SciBERT](https://aclanthology.org/D19-1371.pdf)
    - `bert12max`: representation obtained by max-pooling outputs of 12-th layer in [SciBERT](https://aclanthology.org/D19-1371.pdf)
    - `bert12cls`: representation obtained from \[CLS\] token of 12-th layer in [SciBERT](https://aclanthology.org/D19-1371.pdf)
    - `None`: use text generation evaluation metric (ROUGE, BLEU, METEOR) as sentence similarity

- `similarity`: specifies how to measure similarities between sentences (str)
    
    - `cosine`: cosine similarity between SciBERT or SPECTER-based representations
    - `exponential`: [exponential similarity](https://link.springer.com/content/pdf/10.1007/s42452-019-1142-8.pdf) between SciBERT or SPECTER-based representations
    - `inverse`: similarity between SciBERT or SPECTER-based representations, proportional to (1 + distance)^{-1}
    - `rouge`: symmetric [ROUGE-L score](https://aclanthology.org/W04-1013.pdf)
    - `bleu`: [BLEU score](https://aclanthology.org/P02-1040.pdf)
    - `meteor`: [METEOR score](https://aclanthology.org/W05-0909.pdf)

- `merge`: specifies how to merge sentences to generate multi-document summarization (str)

    - `line`: [TextRank algorithm](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
    - `specter`: integrates SPECTER-based content-level TextRank and sentence-level TextRank
    - `maxpool`: integrates content-level TextRank using max-pooled representations and sentence-level TextRank
    - `avgpool`: integrates content-level TextRank using average-pooled representations and sentence-level TextRank
    - `clustering`: sentence clustering
    - `clique`: graph-based sentence clustering introduced in [SummPip](https://dl.acm.org/doi/pdf/10.1145/3397271.3401327)

- `threshold`: clustering threshold used for `clustering` or `clique` (float)
- `data_path`: directory of the SciTLDR results obtained by content summarization (str)
- `out_path`: directory of the study unit summarization (multi-document summarization) results (str)

### 2.2. Data

- content summarization results should be in `data_path`, where each file in `data_path` has filename as follows:
`<study_unit_id>) <study_unit_name>.out`
    
    e.g. `0) Scalars, Vectors, Matrices and Tensors.out`

- Results of `textrank.py` are saved in `out_path` with filename like the files in `data_path`, but with different extension.
    
    e.g. `0) Scalars, Vectors, Matrices and Tensors.txt`

- In each file in `out_path`, ranking scores of each sentence and the resulting summary (consisting up to 500 letters) of the study unit are reported.

### 2.3. Requirements
`textrank.py` requires the following libraries: `networkx, numpy, torch, nltk, rouge, tqdm, transformers`.

### 2.4. References
[SPECTER: Document-level Representation Learning using Citation-informed Transformers (Cohan et al., ACL 2020)](https://aclanthology.org/2020.acl-main.207.pdf)

[SCIBERT: A Pretrained Language Model for Scientific Text (Beltagy et al., EMNLP 2019)](https://aclanthology.org/D19-1371.pdf)

[SummPip: Unsupervised Multi-Document Summarization with Sentence Graph Compression (Zhao et al., SIGIR 2020)](https://dl.acm.org/doi/pdf/10.1145/3397271.3401327)
