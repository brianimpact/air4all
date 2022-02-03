# Study Unit Summarization

Study unit summarization module generates the summarized text (or the abstraction) of each study unit based on the corresponding contents.

Such summaries provided by [aistudy.guide](aistudy.guide) help users decide whether to study each study unit or not.


## 0. References
References for our two-phase study unit summarizer is as follows:

1. Content Summarization

[TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP 2020)](https://aclanthology.org/2020.findings-emnlp.428.pdf)

2. Multi-Document Summarization

[SPECTER: Document-level Representation Learning using Citation-informed Transformers (Cohan et al., ACL 2020)](https://aclanthology.org/2020.acl-main.207.pdf)

[SummPip: Unsupervised Multi-Document Summarization with Sentence Graph Compression (Zhao et al., SIGIR 2020)](https://dl.acm.org/doi/pdf/10.1145/3397271.3401327)

## 1. Content Summarization
The first phase of the study unit summarization module is content summarization.

In order to preserve information in long contents, sliding windows of sizes 1,024, 512, and 256 are used to summarize the tokens in the windows, where each window's stride is half of its window size.

SciTLDR, a pretrained extreme summarization model ([download](https://github.com/allenai/scitldr)), is adopted for content summarization, where extreme summarization refers to the task of generating a single sentence that abstracts the given document.

Many contents collected for [aistudy.guide](aistudy.guide) are from YouTube, and while their automatically generated transcripts had high quality in general, they shared the following weaknesses.

First, sentences without most of the punctuation marks (including commas, exclamation marks, question marks, and full stops) were generated.

Second, domain-specific terms were incorrectly transcribed due to their low prior.

Thus, in scitldr.py, domain-specific terms are restored using metaphone-based correction and each content is summarized into separated sentences so that they can be used for unsupervised extractive multi-document summarization.

### 1.1. Arguments

Arguments of scitldr.py are as follows:
```
batch_size: minibatch size for text summarization (int)
data_path: directory of the contents (str)
out_path: directory of the summaries (str)
checkpoint_dir: directory of the pretrained checkpoints (str)
checkpoint_name: name of the pretrained checkpoints (str, recommended to use scitldr_catts.tldr-ao.pt)
beam: beam size for better candidate search (int, see https://en.wikipedia.org/wiki/Beam_search)
lenpen: penalty term for short candidates in beam search (float)
max_len_b: upper limit on number of tokens in summary (int)
min_len: lower limit on number of tokens in summary (int)
no_repeat_ngram_size: term to prevent repeating n-grams (int, e.g. if set to 3, repeating trigrams are not generated)
```

For more information on ```beam, lenpen, max_len_b, min_len, no_repeat_ngram_size```, see [this documentation on fairseq's BART generation](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#Generation).

### 1.2. Data

As described above, contents (to be summarized) should be in ```data_path```, where each file in ```data_path``` has filename as follows:
```<study_unit_id>) <study_unit_name>.txt```

e.g. ```0) Scalars, Vectors, Matrices and Tensors.txt```

Each line in a file consists of a content index and corresponding text, separated by a tab.

e.g. ```0\tThis is an example document....```

Results of scitldr.py are saved in ```out_path``` with filename like the files in ```data_path```, but with different extension.

e.g. ```0) Scalars, Vectors, Matrices and Tensors.out```

### 1.3. Requirements
scitldr.py requires the following libraries: ```torch, fairseq, tqdm```.
