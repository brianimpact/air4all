# Content Classifier

Content classifier infers the topics and the study units of a given content.

By doing so, [aistudy.guide](aistudy.guide) is expected to assign automatically-collected documents to their corresponding study units without manual intervention.

In order to make use of the hierarchical structure of topics, a hierarchical text classification model is adopted.

Specifically, [Hierarchy-Aware Global Model (HiAGM)](https://aclanthology.org/2020.acl-main.104.pdf) with enhanced text feature extraction is implemented as our content classifier.

While the original HiAGM makes use of [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), this version utilizes [SciBERT](https://aclanthology.org/D19-1371.pdf)

### 1. Configuration

`run.py` takes the path to a configuration file as its argument. (e.g. `python run.py configs/air4all_config.json`)

An example configuration file can be found in `configs/air4all_config.json`.

Attributes of a configuration are as follows:

- `config.mode`: specifies whether to train a model or to test an existing model (str) ("train" or "test")
- `config.model.embedding`: specifies the BERT-based text feature extraction method

    - includes `type` (which checkpoint from [Huggingface](https://huggingface.co/models) to use), `dimension` (dimension of the text features), `dropout`, and `additional layer` (whether to add an additional fully-connected layer after text feature extraction)
- `config.model.rnn`: specifies the GRU architecture in the classifier

    - includes `in_dimension`, `out_dimension`, `layers`, `bidirectional`, and `dropout`

- `config.model.cnn`: specifies the CNN architecture in the classifier

    - includes `kernels` (list of kernel sizes used for parallel CNNs), `dimension`, and `pooling_k` (used for top-k pooling)

- `config.model.structure_encoder`: specifies the hierarchy-GCN architecture in the classifier

    - includes `dimension` and `dropout`

- `config.model.feature_aggregation`: specifies the text feature propagation process of the classifier

    - includes `dropout`

- `config.training`: specifies the training process of the model (used only if `config.mode=="train"`)

    - includes `batch_size`, `num_epochs`, `max_text_length` (documents longer than this are truncated), and `recursive_regularization_penalty` (see [this paper](http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf))
    - `config.training.optimizer`: specifies the optimizer used while training (includes `type` and `learning_rate`)
    - `config.training.schedule`: specifies the training scheduling process (includes `patience`, `decay`, `early_stopping`, and `auto_save`)

- `config.path`: specifies where checkpoints, logs, and data are saved

    - includes `checkpoints` (path to the checkpoints), `initial_checkpoint` (name of the pretrained checkpoint, empty for clean slate), and `log`
    
    - `config.path.data`: specifies where the files containing required data are saved (includes `train`, `val`, `test`, `labels`, `prior`, and `hierarchy`)


### 2. Data

- `config.path.data.train`, `config.path.data.val`, and `config.path.data.test`

    - These files are .json files where each line is a JSON object composed as follows:

        `{"text": "this is an example document", "label": ["<study_unit_label_name>", "<parent_topic_label_name>", ancester topics]}`

    - The labels are required to be sorted from the leaf node to the node just below the root.

        e.g. `["Vector, Matrix, and Tensor", "Basics of Linear Algebra", "Linear Algebra", "AI Prerequisites"]`

- `config.path.labels`

    - This is a json file containing a dictionary where label names and label indices are keys and values, respectively.

- `config.path.prior`

    - This is a json file containing a dictionary of the following format:

        `{"<parent_label_name>": {"<child_1_label_name>": <prior 1>, "<child_2_label_name>": <prior 2> ...} ...}`

        e.g. `{"AI Prerequisites": {"Calculus": 0.4, "Linear Algebra": 0.3, "Probability": 0.2, "Statistics": 0.1} ...}`
    
    - For each parent node - child node pair, prior is simply obtained as the proportion of the number of instances.

- `config.path.hierarchy`

    - This is a tsv file where each line is written as follows:

        `<parent_label_name>\t<child_1_label_name>\t<child_2_label_name>...`


### 3. Requirements

`run.py` requires the following libraries: `numpy, torch, transformers`.

### 4. References

[SCIBERT: A Pretrained Language Model for Scientific Text (Beltagy et al., EMNLP 2019)](https://aclanthology.org/D19-1371.pdf)

[Hierarchy-Aware Global Model for Hierarchical Text Classification (Zhou et al., ACL 2020)](https://aclanthology.org/2020.acl-main.104.pdf)

[Recursive Regularization for Large-scale Classification with Hierarchical and Graphical Dependencies (Gopal and Yang, KDD 2013)](http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf)