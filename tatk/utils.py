import argparse
import os
import logging
from pathlib import Path
import random
import pickle
import copy

import yaml
import dgl
import torch
import numpy as np
import pandas as pd


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs='+', default=[0])
    parser.add_argument("--data", type=str, default="WIKI")
    parser.add_argument("--model", type=str, default="TGN")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--attack", type=str, default="proposed", choices=["none", "random", "proposed", "degree", "pagerank", "preference", "jaccard"])
    parser.add_argument("--surrogate", type=str, default="SURROGATE")
    parser.add_argument("--ptb_rate", type=float, default=0.3)
    parser.add_argument("--use_hungarian", action="store_true", default=True)
    parser.add_argument("--xpool", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--eval_neg_samples", type=int, default=100)
    parser.add_argument("--rebuild", action="store_true", default=False)

    parser.add_argument("--robust", type=str, default="none", choices=["none", "cosine", "svd", "proposed"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--threshold_s", type=float, default=0.6)
    parser.add_argument("--threshold_e", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--temporal", type=float, default=0.00)
    parser.add_argument("--cosine_threshold", type=float, default=0.5)
    parser.add_argument("--svd_rank", type=int, default=50)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args


def pre_exp_setting(exp_id, args, evaluation=False):
    if args.robust == "proposed":
        exp_dir = f'ROBUST/{args.data}/{args.attack}'
    elif args.robust == "cosine":
        exp_dir = f'COSINE/{args.data}/{args.attack}'
    elif args.robust == "svd":
        exp_dir = f'SVD/{args.data}/{args.attack}'
    elif args.robust == "none": 
        exp_dir = f'NON_ROBUST/{args.data}/{args.attack}'

    if args.attack == "none":
        exp_file = f'{exp_dir}/{args.model}'
    elif args.attack == "proposed":
        exp_file = f'{exp_dir}/{args.model}_{args.surrogate}_{args.ptb_rate}_{args.use_hungarian}_{args.xpool}'
    # elif args.attack != "random":
    #     exp_file = f'{exp_dir}/{args.model}_{args.ptb_rate}_{args.xpool}'
    else:
        exp_file = f'{exp_dir}/{args.model}_{args.ptb_rate}'

    if args.robust == "proposed":
        exp_file += f'_{args.scheduler}_{args.threshold_s}_{args.threshold_e}_{args.mf}_{args.mf_window}_{args.epoch_adj}_{args.temporal}_{args.theta}'
    elif args.robust == "cosine":
        exp_file += f'_{args.scheduler}_{args.cosine_threshold}'
    elif args.robust == "svd":
        exp_file += f'_{args.svd_rank}_{args.svd_threshold}_{args.mf_window}'

    args.exp_id = exp_id
    args.exp_dir = exp_dir
    args.exp_file = exp_file
    Path(f'./LOG/{exp_id}/{exp_dir}/').mkdir(parents=True, exist_ok=True)
    Path(f'./MODEL/{exp_id}/{exp_dir}/').mkdir(parents=True, exist_ok=True)

    if evaluation:
        Path(f'./EVAL/{exp_id}/{exp_dir}/').mkdir(parents=True, exist_ok=True)
        if args.use_clean_eval:
            logger = create_logger(f'./EVAL/{exp_id}/{args.exp_file}_clean_eval.log')
        elif args.use_clean_test:
            logger = create_logger(f'./EVAL/{exp_id}/{args.exp_file}_clean_test.log')
        else:
            logger = create_logger(f'./EVAL/{exp_id}/{args.exp_file}.log')
    else:
        logger = create_logger(f'./LOG/{exp_id}/{args.exp_file}.log')

    logger.warning(args)
    args.logger = logger


def create_logger(file_name="./log/run.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_data(data, args):
    rand_edge_features = 0 if not hasattr(args, 'rand_edge_features') else args.rand_edge_features
    rand_node_features = 0 if not hasattr(args, 'rand_node_features') else args.rand_node_features
    # if data == "MOOC" or data == "LASTFM" or data == "UCI" or data == "LASTFM_SMALL":
    #     rand_edge_features = 128
    node_feats, edge_feats = load_feat(data, rand_edge_features, rand_node_features)
    # node_feats, edge_feats = None, None
    g, df = load_graph(data)
    # if data == "UCI":
    #     edge_feats = torch.randn(len(df), 128)
    return node_feats, edge_feats, g, df


def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'LASTFM_SMALL':
            edge_feats = torch.randn(300000, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
        elif d == "UCI":
            edge_feats = torch.randn(59835, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'LASTFM_SMALL':
            node_feats = torch.randn(1845, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
        elif d == "UCI":
            node_feats = torch.randn(1900, rand_dn)

    return node_feats, edge_feats


def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    # g = np.load('DATA/{}/ext_full.npz'.format(d))
    with open(f'DATA/{d}/ext_full.pkl', 'rb') as f:
        g = pickle.load(f)
    return g, df


def get_attacked_data_dir(args):
    if args.attack == "proposed":
        data_dir = f'DATA/{args.data}/ATTACKED/{args.attack}_{args.surrogate}_{args.ptb_rate}_{args.use_hungarian}'
        if args.use_hungarian:
            data_dir += f'_{args.xpool}/'
    else:
        data_dir = f'DATA/{args.data}/ATTACKED/{args.attack}_{args.ptb_rate}'
        # if args.attack != "random":
        #     data_dir += f'_{args.xpool}/'
    return data_dir



def load_attacked_data(data, args):
    rand_edge_features = 0 if not hasattr(args, 'rand_edge_features') else args.rand_edge_features
    rand_node_features = 0 if not hasattr(args, 'rand_node_features') else args.rand_node_features
    node_feats, edge_feats = load_attacked_feat(data, args, rand_edge_features, rand_node_features)
    # node_feats, edge_feats = None, None 
    g, df = load_attacked_graph(data, args)
    # if data == "UCI":
    #     edge_feats = torch.randn(len(df), 128)
    return node_feats, edge_feats, g, df


def load_attacked_feat(d, args, rand_de=0, rand_dn=0):
    data_dir = get_attacked_data_dir(args)
    node_feats = None
    if os.path.exists(f'{data_dir}/node_features.pt'):
        node_feats = torch.load(f'{data_dir}/node_features.pt')
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists(f'{data_dir}/edge_features.pt'):
        edge_feats = torch.load(f'{data_dir}/edge_features.pt')
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
        elif d == "UCI":
            edge_feats = torch.randn(59835, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
        elif d == "UCI":
            node_feats = torch.randn(1900, rand_dn)
    return node_feats, edge_feats


def load_attacked_graph(d, args):
    data_dir = get_attacked_data_dir(args)
    df = pd.read_csv(f'{data_dir}/edges.csv')
    with open(f'{data_dir}/ext_full.pkl', 'rb') as f:
        g = pickle.load(f)
    return g, df


def parse_config(model):
    config_path = f'./config/{model}.yml'
    conf = yaml.safe_load(open(config_path, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param


def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs


def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs


def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs


def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs


