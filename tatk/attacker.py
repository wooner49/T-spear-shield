import os
import copy
import time
import itertools
import math

from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity

from .utils import (
    parse_config,
    to_dgl_blocks,
    node_to_dgl_blocks,
    prepare_input
)
from .utils_train import (
    create_model_mailbox_sampler,
    train_model_link_pred,
)
from .sampler import ParallelSampler


class TemporalAttack:
    def __init__(self, args):
        self.args = args
    
    def attack(self, orig_node_feats, orig_edge_feats, orig_g, orig_df, ptb_rate, args, seed=0, **kwargs):
        tot_num_ptb = int(len(orig_df) * ptb_rate)
        print(f'>> [{args.attack} attack] len_df: {len(orig_df)}, ptb_rate: {ptb_rate}, tot_num_ptb: {tot_num_ptb}')
        if ptb_rate <= 0.0:
            print(f'>> [{args.attack} attack] elapsed: 0s')
            return orig_node_feats, orig_edge_feats, orig_g, orig_df

        node_feats = None if orig_node_feats is None else orig_node_feats.clone().detach()
        edge_feats = None if orig_edge_feats is None else orig_edge_feats.clone().detach()
        g = copy.deepcopy(orig_g)
        df = orig_df.copy()
        df['adv'] = np.zeros(len(df)).astype(np.int32)


        t_attack_start = time.time()
        train_df = orig_df[orig_df['ext_roll'] == 0]
        valid_df = orig_df[orig_df['ext_roll'] == 1]
        test_df = orig_df[orig_df['ext_roll'] == 2]

        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')

        if args.attack == "proposed":
            ####################### Load surrogate model #######################
            sample_param, memory_param, gnn_param, train_param = parse_config(args.surrogate)
            model, mailbox, sampler = create_model_mailbox_sampler(node_feats, edge_feats, g, df, sample_param, memory_param, gnn_param, train_param)

            seed = seed if seed is not None else 0
            self.args.model_path = f'./MODEL/AAAI/NON_ROBUST/{args.data}/none/{args.surrogate}_seed_{seed}_best_from51.pt'
            if not os.path.isfile(self.args.model_path):
                raise NotImplementedError
            model.load_state_dict(torch.load(self.args.model_path))
            model.eval()

            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                model.memory_updater.last_updated_nid = None

        elif args.attack != "random":
            ####################### Make adjacency matrix ######################
            src_set, dst_set = set(df.src), set(df.dst)
            if args.data == "UCI" and args.data == "BITCOIN":
                src_set = src_set.union(dst_set)
                dst_set = dst_set.union(src_set)
            i = 0
            src_to_idx, idx_to_src = {}, {}
            dst_to_idx, idx_to_dst = {}, {}
            for s in src_set:
                src_to_idx[s] = i
                idx_to_src[i] = s
                i += 1
            for d in dst_set:
                if d in src_to_idx:
                    dst_to_idx[d] = src_to_idx[d]
                    idx_to_dst[src_to_idx[d]] = d
                else:
                    dst_to_idx[d] = i
                    idx_to_dst[i] = d
                    i += 1
            adj = np.zeros((i, i)).astype(np.int32) 
            graph = nx.Graph(adj)

        tot_num_ptb = 0

        for i_df, x_df in enumerate([train_df, valid_df, test_df]):
            ptb_ts, ptb_src, ptb_dst = np.array([]), np.array([]), np.array([])
            ptb_edge_feats = np.array([]).reshape(-1, edge_feats.shape[1]) if edge_feats is not None else None

            for i_rows, rows in x_df.groupby(x_df.index // args.batch_size):
                if i_rows > 0:
                    ################################################################
                    ############## timestamp (common for all attacks) ##############
                    ################################################################
                    num_ptb_batch = int(len(rows) * ptb_rate)
                    kde.fit(rows.time.values.reshape(-1, 1))
                    ptb_ts_batch = np.trunc(kde.sample(num_ptb_batch).squeeze())
                    ptb_ts_batch = np.clip(ptb_ts_batch, rows.time.iloc[0], rows.time.iloc[-1])



                    ################################################################
                    ############ edge features (common for all attacks) ############
                    ################################################################
                    t_batch_start = time.time()
                    if edge_feats is not None:
                        kde.fit(edge_feats[prev_rows['Unnamed: 0'].values])
                        ptb_edge_feats_batch = np.round(kde.sample(num_ptb_batch), 4)

                    ################################################################
                    ####################### (src, dst) pairs #######################
                    ################################################################
                    if args.attack == "random":
                        ptb_src_batch = np.random.choice(list(row_src_set), num_ptb_batch, replace=True)
                        ptb_dst_batch = np.random.choice(list(row_dst_set), num_ptb_batch, replace=True)
                        if args.data == "UCI" or args.data == "BITCOIN":
                            mask = ptb_src_batch == ptb_dst_batch
                            while mask.sum() != 0:
                                ptb_src_batch[mask] = np.random.choice(list(row_src_set), mask.sum(), replace=True)
                                ptb_dst_batch[mask] = np.random.choice(list(row_dst_set), mask.sum(), replace=True)
                                mask = ptb_src_batch == ptb_dst_batch
                
                    elif args.attack == "preference" or args.attack == "jaccard" or args.attack == "degree" or args.attack == "pagerank":
                        if args.attack == "preference":
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            score = list(nx.preferential_attachment(graph, src_dst_pair))
                            src_dst_score = np.array([i for _, _, i in score])
                            # src_dst_score = np.log(src_dst_score + 1)
                        elif args.attack == "jaccard": # Not for bipartite graph
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            score = list(nx.jaccard_coefficient(graph, src_dst_pair))
                            src_dst_score = np.array([i for _, _, i in score])
                        elif args.attack == "degree":
                            deg = adj.sum(1)
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            src_dst_score = deg[src_dst_pair[:, 0]] + deg[src_dst_pair[:, 1]]
                        elif args.attack == "pagerank":
                            pr = nx.pagerank(graph, max_iter=10, tol=1e-3)
                            pr = np.array([pr[i] for i in range(graph.number_of_nodes())]).astype(np.float32)
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            src_dst_score = pr[src_dst_pair[:, 0]] + pr[src_dst_pair[:, 1]]

                        num_hungarian = 0
                        cost = src_dst_score.reshape(len(row_src_set), len(row_dst_set))
                        for elem in inter: ## Remove self-interactions from candidates
                            cost[np.where(row_src_array == elem)[0], np.where(row_dst_array == elem)[0]] = 999999

                        cost1 = cost.copy()
                        row_cand, col_cand = np.array([]).astype(np.int32), np.array([]).astype(np.int32)
                        while len(row_cand) < num_ptb_batch:
                            row_ind, col_ind = linear_sum_assignment(cost)
                            row_cand = np.concatenate((row_cand, row_ind)) 
                            col_cand = np.concatenate((col_cand, col_ind))
                            cost[row_ind, col_ind] = 999999
                            num_hungarian += 1

                        for i_h in range(args.xpool):
                            row_ind, col_ind = linear_sum_assignment(cost)
                            row_cand = np.concatenate((row_cand, row_ind)) 
                            col_cand = np.concatenate((col_cand, col_ind))
                            cost[row_ind, col_ind] = 999999
                            num_hungarian += 1

                        idx_batch = cost1[row_cand, col_cand].argsort()[:num_ptb_batch]
                        ptb_src_batch = np.array([idx_to_src[i] for i in row_src_array[row_cand[idx_batch]]]).astype(np.int32)
                        ptb_dst_batch = np.array([idx_to_dst[j] for j in row_dst_array[col_cand[idx_batch]]]).astype(np.int32)
                        assert num_ptb_batch == len(ptb_src_batch) and num_ptb_batch == len(ptb_dst_batch)
                
                    elif args.attack == "proposed":
                        ######################## Edge scores #######################
                        src_dst_pair = np.array(list(itertools.product(row_src_array, row_dst_array))).astype(np.int32)
                        root_nodes = src_dst_pair.T.reshape(1, -1).squeeze()
                        # ts = rows.time.iloc[0] * np.ones_like(root_nodes).astype(np.float32)
                        ts = ptb_ts_batch.min() * np.ones_like(root_nodes).astype(np.float32)
                        if sampler is not None:
                            sampler.sample(root_nodes, ts)
                            ret = sampler.get_ret()
                        if gnn_param['arch'] != 'identity':
                            mfgs = to_dgl_blocks(ret, sample_param['history'])
                        else:
                            mfgs = node_to_dgl_blocks(root_nodes, ts)
                        mfgs = prepare_input(mfgs, node_feats, edge_feats)
                        if mailbox is not None:
                            mailbox.prep_input_mails(mfgs[0])
                        with torch.no_grad():
                            src_dst_score, _ = model(mfgs, neg_samples=0)

                        num_hungarian = 0
                        if not args.use_hungarian:
                            idx_batch = src_dst_score.squeeze().topk(num_ptb_batch, largest=False).indices.cpu().numpy()
                            ptb_src_batch = src_dst_pair[idx_batch][:, 0]
                            ptb_dst_batch = src_dst_pair[idx_batch][:, 1]
                        else:
                            cost = src_dst_score.reshape(len(row_src_set), len(row_dst_set)).cpu().numpy()
                            for elem in inter: ## Remove self-interactions from candidates
                                cost[np.where(row_src_array == elem)[0], np.where(row_dst_array == elem)[0]] = 999999

                            cost1 = cost.copy()
                            row_cand, col_cand = np.array([]).astype(np.int32), np.array([]).astype(np.int32)
                            while len(row_cand) < num_ptb_batch:
                                row_ind, col_ind = linear_sum_assignment(cost)
                                row_cand = np.concatenate((row_cand, row_ind)) 
                                col_cand = np.concatenate((col_cand, col_ind))
                                cost[row_ind, col_ind] = 999999
                                num_hungarian += 1

                            for i_h in range(args.xpool):
                                row_ind, col_ind = linear_sum_assignment(cost)
                                row_cand = np.concatenate((row_cand, row_ind)) 
                                col_cand = np.concatenate((col_cand, col_ind))
                                cost[row_ind, col_ind] = 999999
                                num_hungarian += 1

                            idx_batch = cost1[row_cand, col_cand].argsort()[:num_ptb_batch]
                            ptb_src_batch = row_src_array[row_cand[idx_batch]]
                            ptb_dst_batch = row_dst_array[col_cand[idx_batch]]
                            assert num_ptb_batch == len(ptb_src_batch) and num_ptb_batch == len(ptb_dst_batch)

                    ptb_ts = np.concatenate((ptb_ts, ptb_ts_batch))
                    ptb_src = np.concatenate((ptb_src, ptb_src_batch))
                    ptb_dst = np.concatenate((ptb_dst, ptb_dst_batch))
                    ptb_edge_feats = np.concatenate((ptb_edge_feats, ptb_edge_feats_batch)) if edge_feats is not None else None
                    num_src_id = len(set(ptb_src_batch)) 
                    num_dst_id = len(set(ptb_dst_batch))
                    t_batch_elapsed = time.time() - t_batch_start
                    if args.attack == "random":
                        print(f'* [Batch {i_df}-{i_rows}] t_batch_elapsed: {t_batch_elapsed:.4f}s, num_ptb_batch: {num_ptb_batch}, num_src_id: {num_src_id}/{len(row_src_set)}, num_dst_id: {num_dst_id}/{len(row_dst_set)}')
                    elif args.attack == "proposed":
                        print(f'* [Batch {i_df}-{i_rows}] t_batch_elapsed: {t_batch_elapsed:.4f}s, num_ptb_batch: {num_ptb_batch}, xpool: {args.xpool}, num_hungarian: {num_hungarian}, num_src_id: {num_src_id}/{len(row_src_set)}, num_dst_id: {num_dst_id}/{len(row_dst_set)}')
                    else:
                        print(f'* [Batch {i_df}-{i_rows}] t_batch_elapsed: {t_batch_elapsed:.4f}s, num_ptb_batch: {num_ptb_batch}, num_hungarian: {num_hungarian}, num_src_id: {num_src_id}/{len(row_src_set)}, num_dst_id: {num_dst_id}/{len(row_dst_set)}')

                ################################################################
                ################### Save previous batch info ###################
                ################################################################
                prev_rows = rows
                if args.attack == "random":
                    row_src_set = set(rows.src)
                    row_dst_set = set(rows.dst)
                    if args.data == "UCI" or args.data == "BITCOIN":
                        row_src_set = row_src_set.union(row_dst_set)
                        row_dst_set = row_dst_set.union(row_src_set)
                elif args.attack == "preference" or args.attack == "jaccard" or args.attack == "degree" or args.attack == "pagerank":
                    src_idx = [src_to_idx[s] for s in rows.src.values]
                    dst_idx = [dst_to_idx[d] for d in rows.dst.values]
                    for s, d in zip(src_idx, dst_idx):
                        adj[s, d] = 1
                        adj[d, s] = 1
                    graph.add_edges_from(list(zip(src_idx, dst_idx)))
                    row_src_set = set(src_idx)
                    row_dst_set = set(dst_idx)
                    if args.data == "UCI" or args.data == "BITCOIN":
                        row_src_set = row_src_set.union(row_dst_set)
                        row_dst_set = row_dst_set.union(row_src_set)
                    row_src_array = np.array(list(row_src_set)).astype(np.int32)
                    row_dst_array = np.array(list(row_dst_set)).astype(np.int32)
                    inter = row_src_set.intersection(row_dst_set)
                elif args.attack == "proposed":
                    ####################### Process batch ######################
                    root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
                    ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)
                    if sampler is not None:
                        sampler.sample(root_nodes, ts)
                        ret = sampler.get_ret()
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts)
                    mfgs = prepare_input(mfgs, node_feats, edge_feats)
                    if mailbox is not None:
                        mailbox.prep_input_mails(mfgs[0])
                    with torch.no_grad():
                        _, _ = model(mfgs, neg_samples=0)

                    if mailbox is not None:
                        eid = rows['Unnamed: 0'].values
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        block = None
                        if memory_param['deliver_to'] == 'neighbors':
                            block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                        mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=0)
                        mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=0)

                    row_src_set = set(rows.src)
                    row_dst_set = set(rows.dst)
                    if args.data == "UCI" or args.data == "BITCOIN":
                        row_src_set = row_src_set.union(row_dst_set)
                        row_dst_set = row_dst_set.union(row_src_set)
                    row_src_array = np.array(list(row_src_set)).astype(np.int32)
                    row_dst_array = np.array(list(row_dst_set)).astype(np.int32)
                    inter = row_src_set.intersection(row_dst_set)


            num_ptb = len(ptb_ts)
            tot_num_ptb += num_ptb 
            ptb_df = pd.DataFrame({
                'Unnamed: 0': np.arange(len(df), len(df) + num_ptb), 
                'src': ptb_src, 
                'dst': ptb_dst, 
                'time': ptb_ts, 
                'int_roll': 0, 
                'ext_roll': i_df,
                'adv': 1
            })
            ptb_edge_feats = torch.from_numpy(ptb_edge_feats).to(dtype=torch.float32) if ptb_edge_feats is not None else None
            node_feats, edge_feats, g, df = self.add_perturbations(node_feats, edge_feats, g, df, ptb_edge_feats, ptb_df)
        
        t_attack_elapsed = time.time() - t_attack_start
        print(f'>> [{args.attack} attack] ptb_rate: {ptb_rate}, tot_num_ptb: {tot_num_ptb}, elapsed: {t_attack_elapsed:.4f}s')
        return node_feats, edge_feats, g, df

    def add_perturbations(self, node_feats, edge_feats, g, df, ptb_edge_feats, ptb_df, verbose=false):
        df = df.append(ptb_df)
        df = df.sort_values('time')
        edge_feats = torch.vstack((edge_feats, ptb_edge_feats)) if ptb_edge_feats is not none else edge_feats
        df = df.reset_index(drop=true)

        num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
        ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
        ext_full_indices = g['ext_full_indices']
        ext_full_ts = g['ext_full_ts']
        ext_full_eid = g['ext_full_eid']

        for i, row in tqdm(ptb_df.iterrows(), total=len(ptb_df), disable=not verbose):
            src = int(row['src'])
            dst = int(row['dst'])
            idx = int(row['Unnamed: 0'])
            ext_full_indices[src].append(dst)
            ext_full_ts[src].append(row['time'])
            ext_full_eid[src].append(idx)
            ext_full_indices[dst].append(src)
            ext_full_ts[dst].append(row['time'])
            ext_full_eid[dst].append(idx)

        for i in tqdm(range(num_nodes), disable=True):
            ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])
        
        def ext_sort(i, indices, t, eid):
            idx = np.argsort(t[i])
            indices[i] = np.array(indices[i])[idx].tolist()
            t[i] = np.array(t[i])[idx].tolist()
            eid[i] = np.array(eid[i])[idx].tolist()
        
        for i in tqdm(range(num_nodes), disable=True):
            ext_sort(i, ext_full_indices, ext_full_ts, ext_full_eid)

        np_ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
        np_ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
        np_ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

        def tsort(i, indptr, indices, t, eid):
            beg = indptr[i]
            end = indptr[i + 1]
            sidx = np.argsort(t[beg:end])
            indices[beg:end] = indices[beg:end][sidx]
            t[beg:end] = t[beg:end][sidx]
            eid[beg:end] = eid[beg:end][sidx]

        for i in tqdm(range(ext_full_indptr.shape[0] - 1), disable=True):
            tsort(i, ext_full_indptr, np_ext_full_indices, np_ext_full_ts, np_ext_full_eid)

        g['indptr'] = ext_full_indptr
        g['indices'] = np_ext_full_indices
        g['ts'] = np_ext_full_ts
        g['eid'] = np_ext_full_eid
        return node_feats, edge_feats, g, df

