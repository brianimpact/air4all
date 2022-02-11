import torch
import pandas as pd
import os
import warnings
from collections import defaultdict
import json

# create a file for the result of filtering
def extract_relevant_idx(out_path, transcript_path, file_su_name, su_names, relevant_idx, cal_freq, included_abb, num_word_threshold=3, low_frequency=0.001):
    data_transcript = []
    data_path = os.path.join(transcript_path,file_su_name)
    with open(data_path,'r') as f:
        infos = json.load(f)
        for content_id in infos.keys():
            content_id = int(content_id)
            if content_id >=0:
                data_transcript.append([content_id])
    data_transcript = pd.DataFrame(data_transcript,columns=['content_id'])
    
    non_label = 0
    for word in su_names:
        if word in relevant_idx.keys():
            add_colum_value = []
            one_word_relevant_idx = relevant_idx[word]
            for content_id in list(data_transcript['content_id'].values):
                if content_id in one_word_relevant_idx:
                    add_colum_value.append(1)
                else:
                    add_colum_value.append(0)
            data_transcript[word] = add_colum_value
        else:
            non_label += 1
            warnings.warn(f'There is any words that related to {word} in {file_su_name}')

    data_transcript.set_index('content_id',inplace=True,drop=True)

    data_transcript['relevance'] = 0
    for idx in data_transcript.index:
        cont = 0
        for word in su_names:
            if word in data_transcript.columns:
                if data_transcript[word][idx] == 1:
                    cont += 1

        if len(su_names) <= num_word_threshold or included_abb == True:
            if cont == len(su_names):
                data_transcript['relevance'][idx] = 1
        else:
            if cont >= len(su_names) - 1:
                data_transcript['relevance'][idx] = 1
    
    # filtering out the low-frequency-label transcript
    cont_frequency_label = []
    filter_trans_idx = set()
    for trans_idx in cal_freq:
        cont_frequency_label.append({trans_idx : cal_freq[trans_idx].values()})
    for i in cont_frequency_label:
        for trans_idx in i.keys():
            valuelist = list(i[trans_idx])
            total_trans_len = valuelist[0]
            total_label_freq = sum(valuelist[1:])
            if total_trans_len == 0:
                filter_trans_idx.add(trans_idx)
            elif (total_label_freq/total_trans_len) < low_frequency:
                filter_trans_idx.add(trans_idx)

    data_transcript_filter_freq = data_transcript
    for i in data_transcript_filter_freq.index:
        if i in filter_trans_idx and data_transcript_filter_freq['relevance'][i] == 1:
            data_transcript_filter_freq['relevance'][i] = 0
    count = 0
    for i in data_transcript_filter_freq.index:
        if data_transcript_filter_freq['relevance'][i] == 1:
            count+=1
    if count >= 5:
        data_transcript = data_transcript_filter_freq
    else:
        data_transcript = data_transcript
    print(f'{list(data_transcript.index)}')
    cont_rele_trans = sum(data_transcript['relevance'])
    if cont_rele_trans == 0:
        warnings.warn(f'Thre is no relevant transcript in {file_su_name}')
       
    print(f'the number of related transcripts : {cont_rele_trans}')
    cont_rele_trnas_idx = set()
    for i in data_transcript.index:
        if data_transcript['relevance'][i] == 1:
            cont_rele_trnas_idx.add(i)
    print(f'Related Content_id : {list(cont_rele_trnas_idx)}')
    out_path_su = os.path.join(out_path,file_su_name)
    data_transcript.to_csv(out_path_su)

# making category vocabulary file
def saving_category_vocabulary_file(temp_dir,file_su_name,category_vocab, only_manually_cate_vocab, only_youtube_cate_vocab):
    category_data_path = temp_dir+'/category_vocabulary'
    vocab_loader_name="category_vocab.pt"
    vocab_save_file = os.path.join(category_data_path, f"{file_su_name}_"+vocab_loader_name)
    torch.save(category_vocab, vocab_save_file)

    manual_loader="manual_category_vocab.pt"
    manual_category_data_path = temp_dir+'/manual_category_vocabulary'
    manual_vocab_save_file = os.path.join(manual_category_data_path, f"{file_su_name}_"+manual_loader)
    torch.save(only_manually_cate_vocab, manual_vocab_save_file)

    only_youtube_vocab_loader_name="only_youtube_category_vocab.pt"
    only_youtube_category_data_path = temp_dir+'/youtube_category_vocabulary'
    only_youtube_vocab_save_file = os.path.join(only_youtube_category_data_path, f"{file_su_name}_"+only_youtube_vocab_loader_name)
    torch.save(only_youtube_cate_vocab, only_youtube_vocab_save_file)
    print("========================================================================================================================================================================================================")

# making abbreviation list
def making_abb_list():
        abb_sorted = {'LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS': ['Local Surrogate'], 'CONVERSATIONAL ARTIFICIAL INTELLIGENCE': ['Conversational AI'], 'EXPLAINABLE ARTIFICIAL INTELLIGENCE': ['Explainable AI'], 'SELF-RECONFIGURABLE MODULAR ROBOTS': ['Reconfigurable Robots'], 'SUBJECT MATTER EXPERT TURING TEST': ['Feigenbaum Test'], 'GRAM-SCHMIDT ORTHONORMALIZATION': ['Gram-Schmidt Process'], 'FIXED-SIZE BLOCKS ALLOCATION': ['Memory Pool'], 'RASTER OPERATIONS PIPELINE': ['Render Output Unit'], 'MARKOV DECISION PROCESSES': ['Markov Decision Process'], 'INTERIOR EXTREMUM THEOREM': ["Fermat's Theorem"], 'CATEGORICAL DISTRIBUTION': ['Multinoulli Distribution'], 'INTEREST POINT DETECTION': ['Interest Operator'], 'BIO-INSPIRED LOCOMOTION': ['Biomimetic Locomotion'], 'COCKTAIL PARTY EFFECT': ['Selective Auditory Attention'], 'NORMAL DISTRIBUTION': ['Gaussian Distribution'], 'MEANS-ENDS ANALYSIS': ['Reasoning as Search'], 'MAPPING CARDINALITY': ['Cardinality Ratio'], 'ANCESTRAL SAMPLING': ['Forward Sampling'], 'DECLARATIVE MEMORY': ['Explicit Memory'], 'PROBABILISTIC PCA': ['Probabilistic Principal Components Analysis'], 'VIRTUAL ASSISTANT': ['Conversational AI'], 'INTENTIONAL AGENT': ['Deliberative Agent'], 'CONVOLUTIONAL NN': ['Convolutional Neural Network'], 'BAYESIAN NETWORK': ['Directed Acyclic Graphical Model'], 'INLINE EXPANSION': ['Inlining'], 'BP THROUGH TIME': ['Backpropagation Through Time'], 'NORMAL EQUATION': ['Ordinary Least Squares'], 'ROTARY ENCODERS': ['Shaft Encoders'], 'MARKOV PROCESS': ['Markov Chain'], 'SPEECH-TO-TEXT': ['Speech Recognition'], 'CONTRACTIVE AE': ['Contractive Autoencoder'], 'VARIATIONAL AE': ['Variational Autoencoders'], 'BAYES THEOREM': ["Bayes' Rule"], 'LOSS FUNCTION': ['Cost Function'], 'STOCHASTIC GD': ['Stochastic Gradient Descent'], 'RESTRICTED BM': ['Restricted Boltzmann Machine'], 'MARKOV MATRIX': ['Stochastic Matrix'], 'RECURRENT NN': ['Recurrent Neural Network'], 'DENOISING AE': ['Denoising Autoencoder'], 'AUTOMATED RL': ['Automated Reinforcement Learning'], 'SCOPED RULES': ['Anchors'], 'AUTOMATED ML': ['Automated Machine Learning'], 'BAGGED TREE': ['Bootstrap Aggregated Tree'], 'UNIT MATRIX': ['Identity Matrix'], 'MEMOISATION': ['Memoization'], 'DOUBLE DQN': ['Double Deep Q-Network'], 'LINEAR MAP': ['Linear Transformation'], 'ERGONOMICS': ['Human Factors'], 'DRILL DOWN': ['Zoom'], 'INCEPTION': ['GoogLeNet'], 'SOJOURNER': ['Robotic Rover'], 'DENSENET': ['Densely Connected Network'], 'ADADELTA': ['Adaptive Delta'], 'WORD2VEC': ['Word-to-Vector'], 'GRAPH NN': ['Graph Neural Network'], 'GRAD-CAM': ['Gradient-Weighted Class Activation Mapping'], 'UD CHAIN': ['Use-Define Chain'], 'PTHREADS': ['POSIX Threads'], 'SNAKEBOT': ['Limbless Biomimetic Locomotion'], 'RIJNDAEL': ['Advanced Encryption Standard'], 'TEARDROP': ['IP Fragmentation Attacks'], 'ADAGRAD': ['Adaptive Gradient'], 'RMSPROP': ['Root Mean Square Propagation'], 'SEQ2SEQ': ['Encoder-Decoder Structure in Sequence-to-Sequence', 'Sequence-to-Sequence'], 'DEEP BM': ['Deep Boltzmann Machine'], 'AUTOAUG': ['Automated Data Augmentation'], 'GRADCAM': ['Gradient-Weighted Class Activation Mapping'], 'SYSCALL': ['System Call'], 'FIREBEE': ['Target Drones'], 'INFOVIS': ['Information Visualization'], 'SQL CLI': ['SQL/Call-Level Interface'], 'SQL PSM': ['SQL/Persistent Stored Modules'], 'AUTORL': ['Automated Reinforcement Learning'], 'RESNET': ['Residual Network'], 'AUTOML': ['Automated Machine Learning'], 'METEOR': ['Metric for Evaluation of Translation with Explicit Ordering'], 'OPENMP': ['Open Multiprocessing'], 'OPENCL': ['Open Computing Language'], 'SHAKEY': ['First AI Robot'], 'ROOMBA': ['Cleaning Robots'], 'STRIPS': ['Stanford Research Institute Problem Solver'], 'SCIVIS': ['Scientific Visualization'], 'COCOMO': ['Constructive Cost Model'], 'TOCTOU': ['Time-of-Check to Time-of-Use'], 'SARSA': ['State-Action-Reward-State-Action', 'State-Action-Reward-State-Action'], 'R-CNN': ['Region-Based CNN'], 'GLOVE': ['Global Vectors'], 'T-SNE': ['t-Stochastic Neighbor Embedding'], 'ROUGE': ['Recall-Oriented Understudy for Gisting Evaluation'], 'SNARC': ['Stochastic Neural Analog Reinforcement Calculator'], 'ENIAC': ['Colossus Computer and Electronic Numerical Integrator And Computer'], 'COBOT': ['Collaborative Robotics in Manufacturing'], 'SDRAM': ['Synchronous Dynamic Random-Access Memory'], 'GDRAM': ['Graphic Dynamic Random-Access Memory'], 'NVRAM': ['Non-Volatile Random-Access Memory'], 'GPGPU': ['General-Purpose computing on Graphics Processing Units'], 'FLOPS': ['Floating-Point Operations Per Second'], 'MTURK': ['Amazon Mechanical Turk'], 'IPSEC': ['Internet Protocol Security'], 'HAZOP': ['Hazard and Operability Analysis'], '3.5NF': ['Boyce-Codd Normal Form'], 'K-NN': ['k-Nearest Neighbors'], 'RELU': ['Rectified Linear Unit'], 'ADAM': ['Adaptive Moment Estimation'], 'BFGS': ['Broyden-Fletcher-Goldfarb-Shanno'], 'BPTT': ['Backpropagation Through Time'], 'LSTM': ['Long Short-Term Memory'], 'PPCA': ['Probabilistic Principal Components Analysis'], 'NICE': ['Non-Linear Independent Components Estimation'], 'MCMC': ['Markov Chain Monte Carlo'], 'MAML': ['Model-Agnostic Meta-Learning'], 'LIME': ['Local Surrogate'], 'SHAP': ['Shapley Additive Explanations'], 'TCAV': ['Testing with Concept Activation Vectors'], 'DDQN': ['Double Deep Q-Network'], 'DDPG': ['Deep Deterministic Policy Gradient'], 'TRPO': ['Full Trust Region Policy Optimization'], 'RCNN': ['Region-Based CNN'], 'BERT': ['Bidirectional Encoder Representations from Transformers'], 'ELMO': ['Embeddings from Language Model'], 'TSNE': ['t-Stochastic Neighbor Embedding'], 'STFT': ['Short Time Fourier Transformation'], 'BLEU': ['Bilingual Evaluation Understudy'], 'ELBO': ['Evidence Lower Bound'], 'SRAM': ['Static Random-Access Memory'], 'DRAM': ['Dynamic Random-Access Memory'], 'MRAM': ['Magnetoresistive Random-Access Memory'], 'PRAM': ['Phase-change Random-Access Memory'], 'RRAM': ['Resistive Random-Access Memory'], 'FRAM': ['Ferroelectric Random-Access Memory'], 'RAID': ['Redundant Array of Independent Disks'], 'RISC': ['Reduced Instruction Set Computer'], 'CISC': ['Complex Instruction Set Computer'], 'VLIW': ['Very Long Instruction Word'], 'EPIC': ['Explicitly Parallel Instruction Computing'], 'BIOS': ['Basic Input/Output System'], 'RTOS': ['Real-Time Operating System'], 'SIMD': ['Single Instruction Stream, Multiple Data Streams'], 'MISD': ['Multiple Instruction Streams, Single Data Stream'], 'MIMD': ['Multiple Instruction Streams, Multiple Data Streams'], 'ASIC': ['Application-Specific Integrated Circuit'], 'FPGA': ['Field-Programmable Gate Array'], 'MIPS': ['Million Instructions Per Second'], 'CUDA': ['Compute Unified Device Architecture'], 'OOTL': ['Human Out-of-the-Loop'], 'BEAM': ['Biology, Electronics, Aesthetics, and Mechanics'], 'FDIR': ['Fault Detection, Isolation, and Recovery'], 'SLAM': ['Simultaneous Localization and Mapping'], 'DTED': ['Digital Terrain Elevation Data'], 'HAII': ['What is Human-AI Interaction'], 'HITL': ['Human-in-the-Loop'], 'GOMS': ['Goals, Operator, Methods, and Selection Rules'], 'RITE': ['Rapid Iterative Testing and Evaluation'], 'UHRS': ['Universal Human Relevance System'], 'RBAC': ['Role-Based Access Control'], 'MITM': ['Man-in-the-Middle'], 'MITB': ['Man-in-the-Browser'], 'DDOS': ['Distributed Denial of Service'], 'E2EE': ['End-to-End Encryption'], 'HIDS': ['Host-Based Intrusion Detection Systems'], 'NIDS': ['Network Intrusion Detection Systems'], 'SIEM': ['Security Information and Event Management'], 'FMEA': ['Failure Modes and Effects Analysis'], 'FGSM': ['Fast Gradient Sign Method'], 'PPML': ['Privacy Preserving Machine Learning'], 'SAML': ['Security Assertion Markup Language'], 'DBMS': ['Database Management System'], 'EKNF': ['Elementary Key Normal Form'], 'BCNF': ['Boyce-Codd Normal Form'], 'DKNF': ['Domain-Key Normal Form'], 'ODMG': ['Object Data Management Group'], 'JPQL': ['Java Persistence Query Language'], 'HDFS': ['Hadoop Distributed File System'], 'SVM': ['Support Vector Machine'], 'KKT': ['Solving Constrained Optimization Problem with Karush-Kuhn Tucker'], 'RBF': ['Radial Basis Function'], 'KNN': ['k-Nearest Neighbors'], 'PCA': ['Principal Components Analysis'], 'LDA': ['Latent Dirichlet Allocation'], 'HMM': ['Hidden Markov Model'], 'GMM': ['Gaussian Mixture Model'], 'MRF': ['Markov Random Field'], 'CRF': ['Conditional Random Field'], 'MLP': ['Multi-Layer Perceptron'], 'SGD': ['Stochastic Gradient Descent'], 'CNN': ['Convolutional Neural Network'], 'RNN': ['Recurrent Neural Network'], 'NTM': ['Neural Turing Machine'], 'GRU': ['Gated Recurrent Unit'], 'STT': ['Speech Recognition'], 'NMT': ['Neural Machine Translation'], 'ICA': ['Independent Component Analysis'], 'DAE': ['Denoising Autoencoder'], 'CAE': ['Contractive Autoencoder'], 'MLE': ['Maximum Likelihood Estimation'], 'NCE': ['Noise-Contrastive Estimation'], 'MAP': ['Maximum A Posteriori'], 'RBM': ['Restricted Boltzmann Machine'], 'DBN': ['Deep Belief Networks'], 'DBM': ['Deep Boltzmann Machine'], 'VAE': ['Variational Autoencoders'], 'GAN': ['Generative Adversarial Networks'], 'GSN': ['Generative Stochastic Network'], 'NAS': ['Neural Architecture Search', 'Network-Attached Storage'], 'PDP': ['Partial Dependence Plot'], 'ICE': ['Individual Conditional Expectation'], 'ALE': ['Accumulated Local Effects'], 'MSE': ['Mean Squared Error'], 'MDP': ['Markov Decision Process'], 'GPI': ['Generalized Policy Iteration'], 'DQN': ['Deep Q-Network'], 'NIN': ['Network in Network'], 'SSD': ['Single Shot Multibox Detector', 'Solid-State Drive'], 'GNN': ['Graph Neural Network'], 'TTS': ['Text-to-Speech'], 'POS': ['Part-of-Speech'], 'NER': ['Named Entity Recognition'], 'SAC': ['Soft Actor-Critic'], 'CTC': ['Connectionist Temporal Classification'], 'DTW': ['Dynamic Time Warping'], 'LLN': ['Law of Large Numbers'], 'FFT': ['Fast Fourier Transformation'], 'LLE': ['Locally Linear Embedding'], 'LRP': ['Layer-Wise Relevance Propagation'], 'GCN': ['Graph Convolutional Networks'], 'OCR': ['Optical Character Recognition'], 'VCG': ['Vickrey-Clarke-Groves'], 'SVD': ['Singular Value Decomposition'], 'CLT': ['Central Limit Theorem'], 'IQR': ['Range and Inter-Quantile Range'], 'DFS': ['Depth-First Search'], 'BFS': ['Breadth-First Search'], 'MST': ['Minimum Spanning Tree'], 'KMP': ['Knuth-Morris-Pratt'], 'HDL': ['Hardware Description Language'], 'FSM': ['Finite-State Machine'], 'XAI': ['Explainable AI'], 'PBD': ['Privacy by Design'], 'PMF': ['Probability Mass Function'], 'PDF': ['Probability Density Function'], 'OLS': ['Ordinary Least Squares'], 'DFT': ['Discrete Fourier Transform'], 'MSS': ['Music Source Separation'], 'CGI': ['Computer Generated Imagery'], 'NPC': ['Non-Player Character'], 'IIS': ['Intelligent Irrigation System'], 'PSA': ['Public Safety Assessment'], 'EOD': ['Explosive Ordnance Disposal'], 'RAM': ['Random-Access Memory'], 'HDD': ['Hard Disk Drive'], 'SAN': ['Storage Area Network'], 'CPU': ['Central Processing Unit'], 'ALU': ['Arithmetic Logic Unit'], 'FPU': ['Floating-Point Unit'], 'MMU': ['Memory Management Unit'], 'ISA': ['What is Instruction Set Architecture'], 'GPU': ['What is a Graphics Processing Unit'], 'TMU': ['Texture Mapping Unit'], 'ROP': ['Render Output Unit'], 'DOS': ['Distributed Operating System', 'Denial of Service'], 'IDE': ['Intergrated Development Environment'], 'TCO': ['Tail Call Optimization'], 'IPO': ['Interprocedural Optimization'], 'DCE': ['Dead Code Elimination'], 'JIT': ['Just-In-Time'], 'CFG': ['Control-Flow Graph'], 'CSE': ['Common Subexpression Elimination'], 'SSA': ['Static Single Assignment'], 'GVN': ['Global Value Numbering'], 'ILP': ['Instruction-Level Parallelism'], 'STM': ['Software Transactional Memory'], 'HPC': ['High-Performance Computing'], 'NPU': ['Neural Processing Unit'], 'TPU': ['Tensor Processing Unit'], 'SMP': ['Symmetric Multiprocessing'], 'IPC': ['Instructions Per Clock'], 'MPI': ['Message Passing Interface'], 'AGV': ['Automatic Guided Vehicles'], 'LOA': ['Levels of Automation'], 'ACL': ['Autonomous Control Levels', 'Access Control List'], 'FSA': ['Finite State Automata'], 'SIT': ['Structural Information Theory'], 'IRM': ['Innate Releasing Mechanism'], 'INS': ['Inertial Navigation Systems'], 'GPS': ['Global Positioning Systems', 'General Problem Solver'], 'GVG': ['Generalized Voronoi Graphs'], 'RRT': ['Rapidly-Exploring Random Tree'], 'BCD': ['Boustrophedon Cell Decomposition'], 'EKF': ['Extended Kalman Filter'], 'MCL': ['Monte Carlo Localization'], 'DEM': ['Digital Elevation Model'], 'DSM': ['Digital Surface Model'], 'MRS': ['MultiRobot Systems'], 'HSV': ['Hue, Saturation, Value'], 'HSI': ['Hue, Saturation, Intensity'], 'SCT': ['Spherical Coordinate Transform'], 'FOV': ['Field of View'], 'TOF': ['Time of Flight'], 'HCI': ['Human-Computer Interaction'], 'HBC': ['Human-Based computation'], 'S-R': ['Stimulus-Response'], 'PRP': ['Psychological Refractory Period'], 'NDM': ['Naturalistic Decision Making'], 'RPD': ['Recognition-Primed Decisions'], 'PCP': ['Parallel Coordinates Plot'], 'ACD': ['Activity-Centered Design'], 'VSD': ['Value-Sensitive Design'], 'GUI': ['Graphic User Interface'], 'TUI': ['Tangible User Interface'], 'VUI': ['Virtual User Interface'], 'KLM': ['Keystroke-Level Model'], 'ICC': ['Intraclass Correlation Coefficient'], 'SSO': ['Single Sign-On'], 'DES': ['Data Encryption Standard'], 'AES': ['Advanced Encryption Standard'], 'RSA': ['Rivest-Shamir-Adelman'], 'OSI': ['Open System Interconnection'], 'XSS': ['Cross-Site Scripting'], 'SQL': ['Structured Query Language'], 'SSI': ['Server-Side Includes'], 'DNS': ['Domain Name Server'], 'WEP': ['Wired Equivalent Privacy'], 'WPA': ['WiFi Protected Access'], 'VPN': ['Virtual Private Networks'], 'NAT': ['Network Address Translation'], 'DLP': ['Data Loss Prevention'], 'IDS': ['Intrusion Detection Systems'], 'TCB': ['Trusted Computing Base'], '2PC': ['Two-Phase Commit'], 'DAM': ['Database Activity Monitoring'], 'IAM': ['Identity and Access Management'], 'BIA': ['Business Impact Analysis'], 'FTA': ['Fault Tree Analysis'], 'PGD': ['Projected Gradient Descent'], 'C&W': ['Carlini and Wagner'], 'UML': ['Unified Modeling Language'], 'EER': ['Enhanced Entity-Relationship'], '1NF': ['First Normal Form'], '2NF': ['Second Normal Form'], '3NF': ['Third Normal Form'], '4NF': ['Fourth Normal Form'], '5NF': ['Fifth Normal Form'], '6NF': ['Sixth Normal Form'], 'DDL': ['Data Definition Language'], 'DML': ['Data Manipulation Language'], 'DCL': ['Data Control Language'], 'SDL': ['Storage Definition Language'], 'VDL': ['View Definition Language'], 'ORD': ['Object-Relational Database'], 'DDB': ['Distributed Database'], 'GDB': ['Graph Database'], 'MDM': ['Master Data Management'], '3PC': ['Three-Phase Commit'], '2PL': ['Two-Phase Locking'], 'MI': ['Mutual Information'], 'EM': ['Expectation-Maximization'], 'CD': ['Contrastive Divergence'], 'BM': ['Boltzmann Machine'], 'DP': ['Dynamic Programming'], 'TD': ['Temporal Difference'], 'MM': ['Minorize-Maximization'], 'CV': ['Coefficient of Variation'], 'SE': ['Standard Error'], 'LU': ['Lower-Upper'], 'CU': ['Control Unit'], 'OS': ['What is an Operating System'], 'UI': ['User Interface'], 'BT': ['Behavior Trees'], 'ER': ['Entity-Relationship'], 'AI': ['Artificial Intelligence'], 'ML': ['Machine Learning']}

        non_unique_abb = defaultdict(list)
        for i in abb_sorted.keys():
            if len(abb_sorted[i]) > 1:
                non_unique_abb[i] = abb_sorted[i]
        return abb_sorted, non_unique_abb