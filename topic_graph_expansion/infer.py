import torch
import argparse
from parse_config import ConfigParser
import numpy as np
import more_itertools as mit
from tqdm import tqdm
import data_loader.data_loaders as model_data
import model.model as model_arch
from gensim.models import KeyedVectors

from pathlib import Path
from collections import OrderedDict

def main(config, args):
    nf = []
    vocab = []
    logger = config.get_logger('test')
    with open(args.taxo, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                seg = line.split("\t")
                nf.append([float(e) for e in seg[1].split(" ")])
                vocab.append("_".join(seg[0].split(" ")))
    nf = np.array(nf)

    if config['train_data_loader']['args']['normalize_embed']:
        sums = nf.sum(axis=1)
        nf = nf/sums[:, np.newaxis]
    kv = KeyedVectors(vector_size=nf.shape[1])
    kv.add_vectors(vocab, nf)

    input_mode = config['input_mode']
    test_data_loader = model_data.TaxoCompDataLoader(
        input_mode=input_mode,
        path=config['path'],
        sampling=0,
        batch_size=1,
        num_neighbors=config['train_data_loader']['args']['num_neighbors'],
        drop_last=False,
        normalize_embed=config['train_data_loader']['args']['normalize_embed'],
        cache_refresh_time=config['train_data_loader']['args']['cache_refresh_time'],
        num_workers=8,
        test=1
    )
    logger.info(test_data_loader)
    test_dataset = test_data_loader.dataset
    idx2word = test_dataset.vocab

    model = config.initialize('architecture', model_arch, input_mode)
    node_features = test_dataset.node_features
    vocab_size, embed_dim = node_features.size()
    model.set_embedding(vocab_size=vocab_size, embed_dim=embed_dim)
    logger.info(model)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v

    device, device_ids = prepare_device(config['n_gpu'], logger)
    model = model.to(device)
    if config['n_gpu'] == 1:
        model.set_device(device)
        model.load_state_dict(state_dict)
    elif config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
        model.module.set_device(device)
        model.load_state_dict(new_state_dict)
    model.eval()

    """Start inference"""
    candidate_positions = test_dataset.all_edges
    if 'g' in input_mode:
        edge2subgraph = {e: test_dataset.get_egonet_node_feature(-1, e[0], e[1]) for e in tqdm(candidate_positions)}

    batched_model = []
    batched_positions = []
    for edges in tqdm(mit.sliced(candidate_positions, args.batch_size), desc="graph encoding"):
        edges = list(edges)
        us, vs, bgu, bgv, lens = None, None, None, None, None
        if 't' in input_mode:
            us, vs = zip(*edges)
            us = torch.tensor(us)
            vs = torch.tensor(vs)
        if 'g' in input_mode:
            bgs = [edge2subgraph[e] for e in edges]
            bgu, bgv = zip(*bgs)

        ur, vr = model.forward_encoders(us, vs, bgu, bgv, lens)
        batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
        batched_positions.append(len(edges))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), open(args.save_path, "w") as fout:
        fout.write(f"Query concept\tPredicted positions\n")
        for i, query in tqdm(enumerate(vocab), desc="predict positions"):
            batched_scores = []
            qf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
            for (ur, vr), n_position in zip(batched_model, batched_positions):
                expanded_qf = qf.expand(n_position, -1)
                ur = ur.to(device)
                vr = vr.to(device)
                if config['n_gpu'] > 1:
                    scores = model.module.match(ur, vr, expanded_qf)
                else:
                    scores = model.match(ur, vr, expanded_qf)
                batched_scores.append(scores)
            batched_scores = torch.cat(batched_scores)
            predicted_scores = batched_scores.cpu().squeeze_().tolist()
            predicted_candidate_positions = [candidate_positions[e[0]] for e in sorted(enumerate(predicted_scores), key=lambda x: -x[1])[:args.topn]]
            predict_parents = "\t".join([f'({idx2word[u].split("@")[0]}, {idx2word[v].split("@")[0]})' for (u, v) in predicted_candidate_positions])
            fout.write(f"{query}\t{predict_parents}\n")

def prepare_device(n_gpu_use, logger):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: No GPU available on this machine,"
                            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    device_idx_list = list(range(n_gpu_use))
    return device, device_idx_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxo', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--save_path', default="./result/inference_results.tsv", type=str)
    parser.add_argument('--device', default=None, type=str)

    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--topn', default=10, type=int)

    args = parser.parse_args()
    config = ConfigParser(parser)
    main(config, args)