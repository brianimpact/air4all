# Topic Graph Expansion

Topic graph expansion module finds an appropriate position for a new concept in an existing topic graph to capture the emerging knowledge and keep the topic graph dynamically updated.

By automatically expanding the topic graph, [aistudy.guide](aistudy.guide) is expected to maintain high coverage without manual curation. 

In order to insert an emerging concept into the most appropriate position, a one-to-pair(new concept-to-candidate parent and child) matching model is adopted.

To be specific, [Triplet Matching Network (TMN)](https://www.aaai.org/AAAI21Papers/AAAI-3030.ZhangJ.pdf) with enhanced node feature is implemented as our taxonomy expansion module.

We mainly leverage two types of concept information to obtain initial concept features: the concept's name and description.

While the original TMN paper only uses name-based concept feature (with fasttext) as input, this version utilizes [SPECTER](https://aclanthology.org/2020.acl-main.207.pdf) to get the initial concept feature and then combines with [Graph Neural Network (GNN)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4700287) encoder to incorporate hierarchical positional information.

### 1. Configuration

`train.py` takes the path to a configuration file as its argument. (e.g. `python train.py --config configs/AI/config.ai.json`)

An example configuration file can be found in `configs/AI/config.ai.json`.

Attributes of a configuration are as follows:

- `model`: specifies model name (str)
- `path`: specifies the path to data file (str)
- `n_gpu`: specifies the number of gpus to use (int)
- `input_mode`: specifies feature encoder. This version uses tg, combination of initial feature vector and GNN, where t indicates initial feature vector and g indicates GNN, respectively. (str)

- `architecture`: specifies the model architecture
    - includes arguments such as `in_dim`, `hidden_dim`, `out_dim`, `num_layers`, and `feat_drop`
   
- `train_data_loader` : specifies data loader for training
    - includes arguments such as `batch_size`, `max_pos_size` (maximum positive positions for training), `neg_size` (negative sampling), `num_neighbors` (number of neighbors for GNN), and `normalize_embed` (whether to normalize initial feature vector)

- `trainer`: specifies the training process of the model
    - includes `epochs`, `test_batch_size` (batch size for validation phase in training), `save_dir` (path to save checkpoint), and `monitor` (used for early stop and lr_scheduler e.g. max val_hit_at_10 indicates metric hit_at_19 is used and max is better)
  
- `loss`: specifies loss (str, default is bce_loss)
- `metrics`: specifies metrics (str, e.g. hit_at_k, mean_rank, scaled_mean_reciprocal_rank etc.)

- `optimizer`: specifies the optimizer used while training
    - includes `type` and arguments such as `lr` (learning rate) and `weight_decay`
    
- `lr_scheduler`: specifies the learning rate scheduling
    - includes `type` and arguments such as `mode` and `patience`
    
### 2. Data

For topic graph expansion, your datasets need to be prepared and formatted accordingly.

- `<taxonomy_name>.terms`, `<taxonomy_name>.relations`, and `<taxonomy_name>.terms.embedding`
    
    - These input files are required to organize your topic graph and node features.

    - `<taxonomy_name>.terms`
      
        - Each line represents one concept (term) in the topic graph composed as follows: `<concept_id>\t<concept_name>`
    
    - `<taxonomy_name>.relations`
      
        - Each line represents one relation in the existing topic graph: `<parent_concept_id>\t<child_concept_id>`
    
    - `<taxonomy_name>.terms.embedding`
      
        - In the first line of this file, the two numbers indicate vocabulary size(total number of concepts) and embedding dimension(in our case, 768), respectively.

        - Each line represents one concept with its pretrained embedding: `<concept_id> <concept_embedding>`

### 3. Training

- For training, specify a config file containing all parameter settings for your dataset.

    e.g. `python train.py --config configs/AI/config.ai.json`

### 4. Inference

- For inference, specify a model checkpoint, a path to a new concept list, and a path to save prediction results.

- The configuration file will set all other configurations.

    e.g. `python infer.py --resume <model save directory>/model_best.pth --taxo <new concept data directory>/new_concepts.txt --save <result save directory>/infer_result.tsv`
    
### 5. Requirements

The following libraries are required for running topic graph expansion module: `torch, tqdm, dgl, networkx`.

### 6. References

[SPECTER: Document-level Representation Learning using Citation-informed Transformers (Cohan et al., ACL 2020)](https://aclanthology.org/2020.acl-main.207.pdf)

[Taxonomy Completion via Triplet Matching Network (Zhang et al., AAAI 2021)](https://www.aaai.org/AAAI21Papers/AAAI-3030.ZhangJ.pdf)

[TaxoExpan: Self-supervised Taxonomy Expansion with Position-Enhanced Graph Neural Network (Shen et al., WWW 2020)](https://dl.acm.org/doi/pdf/10.1145/3366423.3380132)

Topic graph expansion module is implemented based on [TMN](https://github.com/JieyuZ2/TMN) and we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).