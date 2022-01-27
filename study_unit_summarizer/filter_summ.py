import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.translate.meteor_score import meteor_score

class MyOwnLemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, word):
        new_word = self.lemmatizer.lemmatize(word)
        if new_word != word:
            return new_word
        else:
            if word.endswith('ces'):
                word = word[:-3] + 'x'
            elif word.endswith('ies'):
                word = word[:-3] + 'y'
            elif word.endswith('s'):
                word = word[:-1]
        return word

def prep_text(lemmatizer, text):
    text = text.replace('(', ' ').replace(')', ' ').lower().strip()
    text = re.sub('\s+', ' ', re.sub('[^a-z\s]', '', text))
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer(w) for w in word_tokenize(text) if w not in stop_words]
    return words

def check_overlap(lemmatizer, doc, title):
    # COUNT OVERLAPPING WORDS BETWEEN TITLE AND DOCUMENT
    abbreviations = {
        'Local Interpretable Model-Agnostic Explanation': 'Local Surrogate', 'Categorical Distribution': 'Multinoulli Distribution', 'Normal Distribution': 'Gaussian Distribution',
        'Ancestral Sampling': 'Forward Sampling', 'Probabilistic PCA': 'Probabilistic Principal Components Analysis', 'Virtual Assistant': 'Conversational artificial intelligence',
        'Speech-to-Text': 'Speech Recognition', 'Loss Function': 'Cost Function', 'Bagged Tree': 'Bootstrap Aggregated Tree', 'Scoped Rule': 'Anchor', 'Double DQN': 'Double Deep Q-Network',
        'Inception': 'GoogLeNet', 'AdaDelta': 'Adaptive delta', 'Grad-CAM': 'Gradient-weighted class activation mapping', 'PixelCNN': 'Pixel convolutional neural network',
        'PixelRNN': 'Pixel recurrent neural network', 'word2vec': 'Word-to-vector', 'DenseNet': 'Densely Connected Network', 'Word2vec': 'Word-to-Vector', 'AdaGrad': 'adaptive gradient',
        'AutoAug': 'Automated data augmentation', 'RMSProp': 'root mean square propagation', 'seq2seq': 'sequence-to-sequence', 'Seq2seq': 'Sequence-to-Sequence',
        'AutoML': 'Automated machine learning', 'AutoRL': 'Automated reinforcement learning', 'METEOR': 'metric for the evaluation of translation with explicit ordering',
        'ResNet': 'Residual network', 'GloVe': 'Global Vectors', 'R-CNN': 'regions with CNN features', 'ROUGE': 'recall-oriented understudy for gisting evaluation',
        'SARSA': 'State–action–reward–state–action', 't-SNE': 'T-stochastic neighbor embedding', 'Sarsa': 'State–Action–Reward–State–Action', 'Adam': 'Adaptive moment estimation',
        'BERT': 'bidirectional encoder representation from transformers', 'BFGS': 'Broyden-Fletcher-Goldfarb-Shanno', 'BLEU': 'Bilingual evaluation understudy',
        'BPTT': 'Backpropagation through time', 'DDPG': 'Deep Deterministic Policy Gradient', 'DDQN': 'Double deep Q network', 'ELBO': 'Evidence lower bound',
        'ELMo': 'Embedding from language model', 'LIME': 'linear interpretable model agnostic interpretations', 'LSTM': 'long short-term memory', 'MAML': 'model-agnostic meta learning',
        'MCMC': 'Markov chain Monte Carlo', 'NICE': 'nonlinear independent components estimation', 'PPCA': 'Probabilistic principal component analysis', 'RCNN': 'regions with CNN features',
        'ReLU': 'Rectified Linear Unit', 'SHAP': 'Shapley additive explanations', 'TCAV': 'Testing with Concept Activation Vectors', 'TRPO': 'Trust region policy optimization',
        'k-NN': 'k nearest neighbors', 'tSNE': 'T-stochastic neighbor embedding', 'STFT': 'Short Time Fourier Transformation', 'AIS': 'Annealed Importance Sampling',
        'ALE': 'Accumulated local effects', 'CAE': 'Contractive autoencoder', 'CNN': 'Convolutional neural network', 'CTC': 'Connectionist temporal classification',
        'DAE': 'Denoising autoencoder', 'DAG': 'directed acyclic graph', 'DBM': 'deep Boltzmann machine', 'DBN': 'deep belif network', 'DQN': 'deep Q network', 'DTW': 'Dynamic time warping',
        'GAN': 'generative adversarial network', 'GMM': 'Gaussian mixture model', 'GPI': 'Generalized policy iteration', 'GRU': 'Gated recurrent unit', 'GSN': 'Generative stochastic network',
        'HMM': 'hidden Markov model', 'ICA': 'Independent component analysis', 'ICE': 'Individual conditional expectation', 'KKT': 'Karush–Kuhn–Tucker', 'KLD': 'Kullback-Leibler divergence',
        'KRR': 'Knowledge representation and reasoning', 'L2L': 'Learning-to-learn', 'LDA': 'latent Dirichlet allocation', 'LLN': 'law of large numbers', 'LRP': 'layer-wise relevance propagation',
        'MAB': 'multi-armed bandits', 'MAP': 'maximum a posteriori', 'MDP': 'Markov decision process', 'MLE': 'Maximum likelihood estimation', 'MLP': 'multilayer perceptron',
        'MRF': 'Markov random field', 'MSE': 'mean squared error', 'NAS': 'Neural architecture search', 'NCE': 'Noise contrastive estimation', 'NER': 'Named-entity recognition',
        'NIN': 'Network-in-network', 'NLM': 'Neural Language Modeling', 'NLP': 'Natural Language Processing', 'NMT': 'Neural machine translation', 'NTM': 'neural turing machine',
        'OCR': 'Optical character recognition', 'PCA': 'principal component analysis', 'PDP': 'partial dependence plots', 'RBF': 'Radial basis function', 'RBM': 'restricted Boltzmann machine',
        'RNN': 'Recurrent Neural Network', 'SAC': 'Soft actor-critic', 'SGD': 'stochastic gradient descent', 'SSD': 'Single-shot multibox detector', 'STT': 'Speech recognition',
        'SVM': 'support vector machine', 'VAE': 'variational autoencoder', 'XAI': 'explainable AI', 'kNN': 'k nearest neighbors', 'CRF': 'Conditional Random Field', 'TTS': 'Text-to-Speech',
        'POS': 'Part-of-Speech', 'FFT': 'Fast Fourier Transformation', 'LLE': 'Locally Linear Embedding', 'GCN': 'Graph Convolutional Network', 'BM': 'Boltzmann machine',
        'CD': 'Contrastive divergence', 'CV': 'computer vision', 'DP': 'Dynamic Programming', 'EM': 'expectation-maximization', 'EP': 'Expectation propagation', 'KL': 'Kullback-Leibler',
        'LM': 'Language Modeling', 'MI': 'Mutual information', 'MM': 'minorize-maximization', 'RL': 'reinforcement learning', 'TD': 'temporal difference', 'AE': 'Autoencoder',
        'DL': 'Deep Learning', 'ML': 'Machine Learning', 'AI': 'Artificial Intelligence'
    }
    for abb in abbreviations.keys():
        if abb in title:
            title = title + ' ' + abbreviations[abb]
    doc_clean = prep_text(lemmatizer, doc)
    title_clean = prep_text(lemmatizer, title)
    overlaps = dict()
    for query in list(set(title_clean)):
        cnt = 0
        for word in doc_clean:
            if word == query:
                cnt += 1
        overlaps[query] = cnt
    return overlaps

def remove_redundant(lemmatizer, lines):
    # IF TWO LINES ARE ALMOST THE SAME, REMOVE TO REDUCE REDUNDANCY
    clean_lines = [' '.join(prep_text(lemmatizer, line)) for line in lines]
    overlaps = dict()
    for i in range(len(clean_lines) - 1):
        overlap = []
        for j in range(i + 1, len(clean_lines)):
            if (meteor_score([clean_lines[i]], clean_lines[j]) + meteor_score([clean_lines[j]], clean_lines[i])) / 2 > 0.9:
                overlap.append(j)
        overlaps[i] = overlap
    key = sorted(list(overlaps.keys()), key=lambda x: -len(overlaps[x]))
    for k in key:
        if k in overlaps.keys():
            for v in overlaps[k]:
                if v in overlaps.keys():
                    del overlaps[v]
        
    return [lines[k] for k in overlaps.keys()]