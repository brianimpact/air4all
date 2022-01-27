import re
from nltk.corpus import stopwords
import jellyfish


def metaphone_synonyms(title, doc):
    # FIND WORDS WITH SAME METAPHONE (PHONETIC ALPHABET) AS THE WORDS IN THE TITLE
    # SUCH (DOMAIN SPECIFIC) WORDS ARE EASILY MISRECOGNIZED WHEN USING AUTOMATIC SPEECH RECOGNITION
    # 
    # title: topic of the document (in our case, name of the corresponding study unit)
    # doc: document (in our case, contents describing the study unit)
    #
    # PREPROCESS TITLE (REMOVE STOPWORDS AND NON-ALPHABETIC CHARACTERS, AND APPEND WITH SINGULAR TERMS)
    title_words = title.split(')', 1)[1].replace('.txt', '').lower()
    title_words = re.sub('\s+', ' ', re.sub('[^a-z ]', ' ', title_words)).strip().split()
    removed = stopwords.words('english')
    title_words = [word for word in title_words if word not in removed]
    new_words = []
    for word in title_words:
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
    title_words = title_words + new_words
    text = re.sub('\s+', ' ', re.sub('[^a-z0-9 ]', ' ', doc.lower())).strip().split()
    title_metaphones = [jellyfish.metaphone(t) for t in title_words]
    text_metaphones = [jellyfish.metaphone(t) for t in text]
    similar_metaphones = dict()
    for i, title_metaphone in enumerate(title_metaphones):
        similar = set()
        for j, text_metaphone in enumerate(text_metaphones):
            if title_metaphone == text_metaphone and title_words[i] != text[j]:
                similar.add(text[j])
        similar_metaphones[title_words[i]] = similar
    return similar_metaphones


def fix_misrecognition(source_file, source_line):
    # FIX SYNONYMS FOUND BY METAPHONE_SYNONYMS
    synonyms = metaphone_synonyms(source_file, source_line)
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


def inverse_abbreviate(source_file, source_line):
    # DE-ABBREVIATION FOR BETTER CONTEXT UNDERSTANDING AND LESS OOV
    # INPUT: SOURCE_FILE TODO: REMOVE, SOURCE_LINE: DOCUMENT
    abb_dict = {'Speech-to-Text': 'Speech Recognition', 'Inception': 'GoogLeNet', 'AdaDelta': 'Adaptive delta', 'Grad-CAM': 'Gradient-weighted class activation mapping',
                'GradCAM': 'Gradient-weighted class activation mapping', 'PixelCNN': 'Pixel convolutional neural network', 'PixelRNN': 'Pixel recurrent neural network',
                'word2vec': 'Word-to-vector', 'DenseNet': 'Densely Connected Network', 'Word2vec': 'Word-to-Vector', 'AdaGrad': 'adaptive gradient',
                'AutoAug': 'Automated data augmentation', 'RMSProp': 'root mean square propagation', 'seq2seq': 'sequence-to-sequence', 'Bagged': 'Bootstrap Aggregated',
                'AutoML': 'Automated machine learning', 'AutoRL': 'Automated reinforcement learning', 'METEOR': 'metric for the evaluation of translation with explicit ordering',
                'ResNet': 'Residual network', 'GloVe': 'Global Vectors', 'R-CNN': 'regions with CNN features', 'ROUGE': 'recall-oriented understudy for gisting evaluation',
                'SARSA': 'State–action–reward–state–action', 't-SNE': 'T-stochastic neighbor embedding', 'Sarsa': 'State–Action–Reward–State–Action','Adam': 'Adaptive moment estimation',
                'BERT': 'bidirectional encoder representation from transformers', 'BFGS': 'Broyden-Fletcher-Goldfarb-Shanno', 'BLEU': 'Bilingual evaluation understudy',
                'BPTT': 'Backpropagation through time', 'DDPG': 'Deep Deterministic Policy Gradient', 'DDQN': 'Double deep Q network', 'ELBO': 'Evidence lower bound',
                'ELMo': 'Embedding from language model', 'LIME': 'linear interpretable model agnostic interpretations', 'LSTM': 'long short-term memory',
                'MAML': 'model-agnostic meta learning', 'MCMC': 'Markov chain Monte Carlo','CRNN': 'convolutional-recurrent neural network',
                'PPCA': 'Probabilistic principal component analysis', 'RCNN': 'regions with CNN features', 'ReLU': 'Rectified Linear Unit', 'SHAP': 'Shapley additive explanations',
                'TCAV': 'Testing with Concept Activation Vectors', 'TRPO': 'Trust region policy optimization', 'k-NN': 'k nearest neighbors', 'tSNE': 'T-stochastic neighbor embedding',
                'STFT': 'Short Time Fourier Transformation', 'AIS': 'Annealed Importance Sampling', 'ALE': 'Accumulated local effects', 'CAE': 'Contractive autoencoder',
                'CNN': 'Convolutional neural network', 'CTC': 'Connectionist temporal classification', 'DAE': 'Denoising autoencoder', 'DAG': 'directed acyclic graph',
                'DBM': 'deep Boltzmann machine', 'DBN': 'deep belif network', 'DQN': 'deep Q network', 'DTW': 'Dynamic time warping', 'GAN': 'generative adversarial network',
                'GMM': 'Gaussian mixture model', 'GPI': 'Generalized policy iteration', 'GRU': 'Gated recurrent unit', 'GSN': 'Generative stochastic network', 'HMM': 'hidden Markov model',
                'ICA': 'Independent component analysis', 'ICE': 'Individual conditional expectation', 'KKT': 'Karush–Kuhn–Tucker', 'KLD': 'Kullback-Leibler divergence',
                'KRR': 'Knowledge representation and reasoning', 'L2L': 'Learning-to-learn', 'LDA': 'latent Dirichlet allocation', 'LLN': 'law of large numbers',
                'LRP': 'layer-wise relevance propagation', 'MAB': 'multi-armed bandits', 'MDP': 'Markov decision process', 'MLE': 'Maximum likelihood estimation',
                'MLP': 'multilayer perceptron', 'MRF': 'Markov random field', 'MSE': 'mean squared error', 'NAS': 'Neural architecture search', 'NCE': 'Noise contrastive estimation',
                'NER': 'Named-entity recognition', 'NIN': 'Network-in-network', 'NLM': 'Neural Language Modeling', 'NLP': 'Natural Language Processing', 'NMT': 'Neural machine translation',
                'NTM': 'neural turing machine', 'OCR': 'Optical character recognition', 'PCA': 'principal component analysis', 'PDP': 'partial dependence plots', 'RBF': 'Radial basis function',
                'RBM': 'restricted Boltzmann machine', 'RNN': 'Recurrent Neural Network', 'SAC': 'Soft actor-critic', 'SGD': 'stochastic gradient descent', 'SSD': 'Single-shot multibox detector',
                'STT': 'Speech recognition', 'ASR': 'Speech recognition', 'SVM': 'support vector machine', 'VAE': 'variational autoencoder', 'XAI': 'explainable AI', 'kNN': 'k nearest neighbors',
                'CRF': 'Conditional Random Field', 'TTS': 'Text-to-Speech', 'POS': 'Part-of-Speech', 'FFT': 'Fast Fourier Transformation', 'LLE': 'Locally Linear Embedding',
                'GCN': 'Graph Convolutional Network', 'BM': 'Boltzmann machine', 'CD': 'Contrastive divergence', 'CV': 'computer vision', 'DP': 'Dynamic Programming',
                'EM': 'expectation-maximization', 'EP': 'Expectation propagation', 'KL': 'Kullback-Leibler', 'LM': 'Language Modeling', 'MI': 'Mutual information', 'MM': 'minorize-maximization',
                'RL': 'reinforcement learning', 'TD': 'temporal difference', 'AE': 'Autoencoder', 'DL': 'Deep Learning', 'ML': 'Machine Learning', 'AI': 'Artificial Intelligence',
                'RMS': 'root mean square', 'prop': 'propagation'}
    
    source_split = source_line.split()
    source_split_lower = [x.lower() for x in source_split]
    for abb in sorted(list(abb_dict.keys()), key=lambda x: -len(x)):
        for i, ssl in enumerate(source_split_lower):
            if (re.sub('[^a-z]', '', ssl) == abb.lower()) or (re.sub('[^a-z]', '', ssl) == abb.lower() + 's'):
                new_word = abb_dict[abb].lower()
                if re.sub('[^a-z]', '', ssl) == abb + 's':
                    new_word = new_word + 's'
                for letter in ssl:
                    if ord(letter) < 97 or ord(letter) > 122:
                        new_word = letter + new_word
                    else:
                        break
                for letter in ssl[::-1]:
                    if ord(letter) < 97 or ord(letter) > 122:
                        new_word = new_word + letter
                    else:
                        break
                source_split[i] = new_word
                source_split_lower[i] = new_word

    return ' '.join(source_split)