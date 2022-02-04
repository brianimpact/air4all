import argparse
import os

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from filter_summ import MyOwnLemmatizer, check_overlap, remove_redundant
from utils import *


class EmbeddingExtractor:
    def __init__(self, sentence_embedding, similarity, merge, threshold):
        self.sentence_embedding = sentence_embedding
        self.lemmatizer = MyOwnLemmatizer()
        # SCIBERT
        # SciBERT: A Pretrained Language Model for Scientific Text (I. Beltagy, K. Lo, and A. Cohan, EMNLP 2019)
        if self.sentence_embedding.startswith('bert'):
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.language_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        # SPECTER
        # SPECTER: Document-level Representation Learning using Citation-informed Transformers (A. Cohan, S. Feldman, I. Beltagy, D. Downey, and D. S. Weld, ACL 2020)
        elif sentence_embedding == 'specter':
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            self.language_model = AutoModel.from_pretrained('allenai/specter')
        else:
            raise NotImplementedError
        self.language_model.eval()
        self.language_model.cuda()

        if similarity == 'cosine':
            self.simfunc = cosine_similarity
        elif similarity == 'exponential':
            self.simfunc = exponential_similarity
        elif similarity == 'inverse':
            self.simfunc = inverse_similarity
        else:
            raise NotImplementedError
        # SENTENCE MERGE BASED ON SPECTER EMBEDDING SIMILARITY
        self.merge = merge
        if self.merge == 'specter':
            self.merge_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            self.merge_language_model = AutoModel.from_pretrained('allenai/specter')
            self.merge_language_model.cuda()
            self.merge_language_model.eval()
        
        self.threshold = threshold
    
    def __call__(self, documents, title):
        with torch.no_grad():
            doc_list = documents.split('\n\n')
            doc_list = [x for x in doc_list if len(x) > 0]
            out_list = []
            source_lines = []
            for doc in doc_list:
                sentences = [simple_preprocess(s) for s in doc.split('\n') if len(s) > 0]
                out_list.append([])
                # REMOVE OVERLAPPING SENTENCES
                sentences = remove_redundant(self.lemmatizer, sentences)
                for sentence in sentences:
                    # REMOVE IRRELEVANT SENTENCES
                    overlap = check_overlap(self.lemmatizer, sentence, title)
                    if sum(list(overlap.values())) == 0:
                        continue
                    source_lines.append(sentence)
                    if self.sentence_embedding.startswith('bert'):
                        inputs = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0).cuda()
                        outputs = self.language_model(inputs, output_hidden_states=True)[2][int(self.sentence_embedding[4:6])]
                        # WHERE TO EXTRACT SENTENCE EMBEDDING FROM BERT: CLS TOKEN, TOKEN AVERAGE OR TOKEN MAX POOLING
                        if self.sentence_embedding[6:] == 'cls':
                            out_list[-1].append(outputs[0, 0, :])
                        elif self.sentence_embedding[6:] == 'avg':
                            out_list[-1].append(outputs[0, :, :].mean(0))
                        elif self.sentence_embedding[6:] == 'max':
                            out_list[-1].append(outputs[0, :, :].max(0)[0])
                        else:
                            raise NotImplementedError
                    # SPECTER EMBEDDING EXTRACTION
                    elif self.sentence_embedding == 'specter':
                        inputs = self.tokenizer([title + self.tokenizer.sep_token + sentence], padding=False, truncation=True, return_tensors="pt", max_length=512)
                        for k in inputs.keys():
                            inputs[k] = inputs[k].cuda()
                        outputs = self.language_model(**inputs)
                        out_list[-1].append(outputs.last_hidden_state[0, 0, :])
                    else:
                        raise NotImplementedError
            values = []
            results = [y for x in out_list for y in x]
            for i in range(len(results) - 1):
                for j in range(i + 1, len(results)):
                    values.append((results[i] - results[j]).norm().cpu().item())
            t = sum(values) / len(values)
            
            # TEXTRANK
            if self.merge == 'line':
                results = [y for x in out_list for y in x]
                matrix = np.zeros([len(results), len(results)], dtype=np.float64)
                for i in range(len(results)):
                    for j in range(len(results)):
                        if i != j:
                            matrix[i, j] = self.simfunc(results[i], results[j], t)
                graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
                score_dict = nx.pagerank(graph)
                scores = [score_dict[idx] for idx in range(len(score_dict))]
                assert len(scores) == len(source_lines)
                ranked_list = sorted(((scores[i], s) for i, s in enumerate(source_lines)), reverse=True)
            # SENTENCE RANKING AFTER DOCUMENT RANKING
            elif self.merge in ['specter', 'maxpool', 'avgpool']:
                score_list = []
                for result in out_list:
                    matrix = np.zeros([len(result), len(result)], dtype=np.float64)
                    for i in range(len(result)):
                        for j in range(len(result)):
                            if i != j:
                                matrix[i, j] = self.simfunc(result[i], result[j], t)
                    graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
                    score_dict = nx.pagerank(graph)
                    score_list.append([score_dict[idx] for idx in range(len(score_dict))])
                if self.merge == 'maxpool':
                    doc_representation = [torch.stack(o).max(0)[0] for o in out_list]
                elif self.merge == 'avgpool':
                    doc_representation = [torch.stack(o).mean(0) for o in out_list]
                elif self.merge == 'specter':
                    doc_representation = []
                    for doc in doc_list:
                        inputs = self.merge_tokenizer([title + self.tokenizer.sep_token + doc.replace('\n', ' ')], padding=False, truncation=True, return_tensors="pt", max_length=512)
                        for k in inputs.keys():
                            inputs[k] = inputs[k].cuda()
                        outputs = self.merge_language_model(**inputs)
                        doc_representation.append(outputs.last_hidden_state[:, 0, :])
                matrix = np.zeros([len(doc_representation), len(doc_representation)], dtype=np.float64)
                for i in range(len(doc_representation)):
                    for j in range(len(doc_representation)):
                        if i != j:
                            matrix[i, j] = ((doc_representation[i] * doc_representation[j]).sum() / (doc_representation[i].norm() * doc_representation[j].norm())).item()
                graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
                score_dict = nx.pagerank(graph)
                doc_score = [score_dict[idx] for idx in range(len(score_dict))]
                
                for i in range(len(doc_list)):
                    score_list[i] = [s * doc_score[i] / len(score_list[i]) for s in score_list[i]]
                scores = [y for x in score_list for y in x]
                ranked_list = sorted(((scores[i], s) for i, s in enumerate(source_lines)), reverse=True)
            # SENTENCE CLUSTERING BASED APPROACH
            elif self.merge == 'clustering':
                results = [y for x in out_list for y in x]
                cluster_id = [[idx] for idx in range(len(results))]
                cluster_vec = [[v] for v in results]
                while True:
                    similarities = dict()
                    for i in range(len(cluster_id) - 1):
                        for j in range(i + 1, len(cluster_id)):
                            # CENTROID DISTANCE CALCULATION
                            center_i = torch.stack(cluster_vec[i]).mean(0)
                            center_j = torch.stack(cluster_vec[j]).mean(0)
                            similarities[(i, j)] = self.simfunc(center_i, center_j, t)
                    clustered = sorted(list(similarities.keys()), key=lambda x: -similarities[x])[0]
                    if similarities[clustered] < self.threshold:
                        break
                    cluster_id[clustered[0]] = cluster_id[clustered[0]] + cluster_id[clustered[1]]
                    cluster_vec[clustered[0]] = cluster_vec[clustered[0]] + cluster_vec[clustered[1]]
                    del cluster_id[clustered[1]]
                    del cluster_vec[clustered[1]]
                    if len(cluster_id) < 5:
                        break
                # PRIORITIZE BIGGER CLUSTERS
                cluster_size_order = sorted(list(range(len(cluster_id))), key=lambda x: -len(cluster_id[x]))
                ranked_list = []
                for idx in cluster_size_order:
                    cluster_center = torch.stack(cluster_vec[idx]).mean(0)
                    intra_cluster = [self.simfunc(v, cluster_center, t) for v in cluster_vec[idx]]
                    max_idx = cluster_id[idx][sorted(list(range(len(cluster_id[idx]))), key=lambda x: -intra_cluster[x])[0]]
                    ranked_list.append((len(cluster_id[idx]), source_lines[max_idx]))
            # GRAPH CLUSTERING (CLIQUE) BASED APPROACH
            # SummPip: Unsupervised Multi-Document Summarization with Sentence Graph Compression (J. ZHAO, M. LIU, L. GAO, Y. JIN, L. DU, H. ZHANG, AND G. HAFFARI, SIGIR 2020)
            elif self.merge == 'clique':
                results = [y for x in out_list for y in x]
                matrix = np.zeros([len(results), len(results)], dtype=np.float64)
                for i in range(len(results)):
                    for j in range(len(results)):
                        if i != j:
                            sim = self.simfunc(results[i], results[j], t)
                            matrix[i, j] = sim if sim > self.threshold else np.NaN
                graph = nx.from_numpy_array(matrix)
                loc_nan = np.argwhere(np.isnan(matrix))
                graph.remove_edges_from([(loc_nan[i, 0], loc_nan[i, 1]) for i in range(loc_nan.shape[0])])
                remaining_nodes = list(range(len(results)))
                clique_nodes = []
                while True:
                    clique = list(nx.algorithms.approximation.clique.max_clique(graph))
                    if len(clique) == 1:
                        break
                    clique_nodes.append([remaining_nodes[i] for i in clique])
                    graph.remove_nodes_from(clique)
                ranked_list = []
                for clique in clique_nodes:
                    similarities = []
                    for node in clique:
                        similarities.append(sum([matrix[node, n2] for n2 in clique if n2 != node]))
                    representing_node = clique[sorted(list(range(len(clique))), key=lambda x: -similarities[x])[0]]
                    ranked_list.append((len(clique), source_lines[representing_node]))
                for node in remaining_nodes:
                    ranked_list.append((1, source_lines[node]))
            else:
                raise NotImplementedError

            return ranked_list


class MetricBasedExtractor:
    # CALCULATES SENTENCE SIMILARITY USING TEXT EVALUATION METRICS (ROUGE, BLEU, METEOR)
    def __init__(self, similarity, merge, threshold):
        self.lemmatizer = MyOwnLemmatizer()

        if similarity == 'rouge':
            self.simfunc = calc_rouge
        elif similarity == 'bleu':
            self.simfunc = calc_bleu
        elif similarity == 'meteor':
            self.simfunc = calc_meteor
        else:
            raise NotImplementedError
        
        self.merge = merge
        self.threshold = threshold
    
    def __call__(self, documents, title):
        doc_list = documents.split('\n\n')
        doc_list = [x for x in doc_list if len(x) > 0]
        out_list = []
        source_lines = []
        for doc in doc_list:
            sentences = [simple_preprocess(s) for s in doc.split('\n') if len(s) > 0]
            out_list.append([])
            sentences = remove_redundant(self.lemmatizer, sentences)
            for sentence in sentences:
                overlap = check_overlap(self.lemmatizer, sentence, title)
                if sum(list(overlap.values())) == 0:
                    continue
                source_lines.append(sentence)
                out_list[-1].append(sentence)
        # TEXTRANK
        if self.merge == 'line':
            matrix = np.zeros([len(source_lines), len(source_lines)], dtype=np.float64)
            for i in range(len(source_lines)):
                for j in range(len(source_lines)):
                    if i != j:
                        matrix[i, j] = self.simfunc(source_lines[i], source_lines[j])
            graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            score_dict = nx.pagerank(graph)
            scores = [score_dict[idx] for idx in range(len(score_dict))]
            assert len(scores) == len(source_lines)
            ranked_list = sorted(((scores[i], s) for i, s in enumerate(source_lines)), reverse=True)
        # SENTENCE CLUSTERING BASED APPROACH
        elif self.merge == 'clustering':
            cluster_id = [[idx] for idx in range(len(source_lines))]
            sentence_similarities = dict()
            for i in range(len(source_lines) - 1):
                for j in range(i + 1, len(source_lines)):
                    sentence_similarities[(i, j)] = self.simfunc(source_lines[i], source_lines[j])
                    sentence_similarities[(j, i)] = self.simfunc(source_lines[i], source_lines[j])
                    # intercluster_similarity([source_lines[x] for x in cluster_id[i]], [source_lines[x] for x in cluster_id[j]], self.simfunc)
            while True:
                if len(cluster_id) < 5:
                    break
                cluster_similarities = dict()
                for i in range(len(cluster_id) - 1):
                    for j in range(i + 1, len(cluster_id)):
                        values = []
                        for x1 in cluster_id[i]:
                            for x2 in cluster_id[j]:
                                values.append(sentence_similarities[x1, x2])
                        cluster_similarities[(i, j)] = sum(values) / len(values)
                clustered = sorted(list(cluster_similarities.keys()), key=lambda x: -cluster_similarities[x])[0]
                if cluster_similarities[clustered] < self.threshold:
                    break
                cluster_id[clustered[0]] = cluster_id[clustered[0]] + cluster_id[clustered[1]]
                del cluster_id[clustered[1]]
                
            cluster_size_order = sorted(list(range(len(cluster_id))), key=lambda x: -len(cluster_id[x]))
            ranked_list = []
            for idx in cluster_size_order:
                centroid = cluster_centroid([source_lines[x] for x in cluster_id[idx]], self.simfunc)
                ranked_list.append((len(cluster_id[idx]), centroid))
        # GRAPH CLUSTERING (CLIQUE) BASED APPROACH
        # SummPip: Unsupervised Multi-Document Summarization with Sentence Graph Compression (J. ZHAO, M. LIU, L. GAO, Y. JIN, L. DU, H. ZHANG, AND G. HAFFARI, SIGIR 2020)
        elif self.merge == 'clique':
            matrix = np.zeros([len(source_lines), len(source_lines)], dtype=np.float64)
            for i in range(len(source_lines)):
                for j in range(len(source_lines)):
                    if i != j:
                        sim = self.simfunc(source_lines[i], source_lines[j])
                        matrix[i, j] = sim if sim > self.threshold else np.NaN
            graph = nx.from_numpy_array(matrix)
            loc_nan = np.argwhere(np.isnan(matrix))
            graph.remove_edges_from([(loc_nan[i, 0], loc_nan[i, 1]) for i in range(loc_nan.shape[0])])
            remaining_nodes = list(range(len(source_lines)))
            clique_nodes = []
            while True:
                clique = list(nx.algorithms.approximation.clique.max_clique(graph))
                if len(clique) == 1:
                    break
                clique_nodes.append([remaining_nodes[i] for i in clique])
                graph.remove_nodes_from(clique)
            ranked_list = []
            for clique in clique_nodes:
                similarities = []
                for node in clique:
                    similarities.append(sum([matrix[node, n2] for n2 in clique if n2 != node]))
                representing_node = clique[sorted(list(range(len(clique))), key=lambda x: -similarities[x])[0]]
                ranked_list.append((len(clique), source_lines[representing_node]))
            for node in remaining_nodes:
                ranked_list.append((1, source_lines[node]))

        else:
            raise NotImplementedError

        return ranked_list


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_embedding', default=None, type=str, choices=['specter', 'bert11avg', 'bert11max', 'bert11cls', 'bert12avg', 'bert12max', 'bert12cls', None])
    parser.add_argument('--similarity', default='rouge', type=str, choices=['cosine', 'exponential', 'inverse', 'rouge', 'bleu', 'meteor'])
    parser.add_argument('--merge', default='clustering', type=str, choices=['line', 'specter', 'maxpool', 'avgpool', 'clustering', 'clique'])
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--data_path', default='/data5/assets/jinhyun95/air4all/single_doc_summ/v220126', help='Path to data directory')
    parser.add_argument('--out_path', default='/data5/assets/jinhyun95/air4all/multi_doc_summ/v220126', help='Path to output directory')

    args = parser.parse_args()
    # CHECK PATHS
    if not os.path.exists(args.data_path):
        print(f'{args.data_path} does not exist')
        exit(0)
    os.makedirs(args.out_path, exist_ok=True)

    # SENTENCE EMBEDDING BASED APPROACH AND TEXT EVALUATION METRIC BASED APPROACH
    if args.sentence_embedding is None:
        extractor = MetricBasedExtractor(args.similarity, args.merge, args.threshold)
    else:
        extractor = EmbeddingExtractor(args.sentence_embedding, args.similarity, args.merge, args.threshold)
    fnames = sorted([x for x in os.listdir(args.data_path) if x.endswith('.out')], key=lambda x: int(x.split(')')[0]))
    for fname in tqdm(fnames):
        with open(os.path.join(args.data_path, fname), 'r', encoding='utf8') as f:
            lines = f.read()
        if len(lines) == 0:
            continue
        title = fname.split(') ', 1)[1].replace('.out', '')
        ranked_list = extractor(lines, title)
        summ, verbose = make_string(ranked_list)
        with open(os.path.join(args.out_path, fname.replace('.out', '.txt')), 'w', encoding='utf8') as f:
            f.write('# SUMMARY UP TO 500 CHARACTERS (DATABASE FORMAT)\n')
            f.write(summ + '\n\n')
            f.write('# SENTENCE RANKING\n')
            f.write(verbose)
