import math
import time
from pathlib import Path
import copy
import itertools
from collections import deque
# import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from deeprobust.graph.defense.pgd import PGD, prox_operators

from .modules import GeneralModel, EstimateAdj
from .memorys import MailBox, EmbeddingBox
from .sampler import ParallelSampler, NegLinkSampler, NegLinkSamplerDST
from .utils import (
    to_dgl_blocks,
    node_to_dgl_blocks,
    prepare_input,
)
from pympler.tracker import SummaryTracker


def create_model_mailbox_sampler(node_feats, edge_feats, g, df, sample_param, memory_param, gnn_param, train_param):
    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()
    
    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    
    return model, mailbox, sampler


def train_model_link_pred(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, seed=0):
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    if args.data != "UCI" and args.data != "BITCOIN":
        neg_link_sampler = NegLinkSamplerDST(df.dst.values)
    else:
        src_set = set(df.src.values)
        dst_set = set(df.dst.values)
        node_set = src_set.union(dst_set)
        neg_link_sampler = NegLinkSamplerDST(node_set)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

    train_df = df[df['ext_roll'] == 0]

    if args.robust == "svd":
        src_set, dst_set = set(train_df.src), set(train_df.dst)
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
        src_idx = [src_to_idx[s] for s in train_df.src.values]
        dst_idx = [dst_to_idx[d] for d in train_df.dst.values]
        for s, d in zip(src_idx, dst_idx):
            adj[s, d] = 1
            adj[d, s] = 1
        
        U, S, V = np.linalg.svd(adj)
        rec_adj = U[:, :args.svd_rank] @ np.diag(S[:args.svd_rank]) @ V[:args.svd_rank, :]
        rec_adj = (rec_adj + rec_adj.T) / 2
        rec_adj = np.clip(rec_adj, 0, 1)


    for e in range(1, train_param['epoch'] + 1):
        time_sample = 0
        time_prep = 0
        time_tot = 0
        total_loss = 0

        # training
        model.train()
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
            model.memory_updater.last_updated_ts = None
            model.memory_updater.last_updated_memory = None

        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            t_tot_s = time.time()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples=1)]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                time_sample += ret[0].sample_time()
            t_prep_s = time.time()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            time_prep += time.time() - t_prep_s

            ## Default Loss function
            optimizer.zero_grad()
            if args.robust == "none":
                pred_pos, pred_neg = model(mfgs)
                loss = criterion(pred_pos, torch.ones_like(pred_pos))
                loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            elif args.robust == "svd":
                node_emb = model.get_embeddings(mfgs)
                pred_pos, pred_neg = model.edge_predictor(node_emb, neg_samples=1)
                pred_pos, pred_neg = pred_pos.squeeze(), pred_neg.squeeze()

                src_idx = [src_to_idx[s] for s in rows.src.values]
                dst_idx = [dst_to_idx[d] for d in rows.dst.values]

                weights = rec_adj[src_idx, dst_idx]
                weights = torch.from_numpy(weights).cuda()
            
                loss_pos = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos), reduction="none")
                loss_pos = torch.mean(loss_pos * weights)
                loss_neg = F.binary_cross_entropy_with_logits(pred_neg, torch.zeros_like(pred_neg), reduction="none")
                loss_neg = torch.mean(loss_neg * weights)
                loss = loss_pos + loss_neg

            total_loss += float(loss) * train_param['batch_size']
            loss.backward()
            optimizer.step()

            t_prep_s = time.time()
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors': ## for APAN
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
            time_prep += time.time() - t_prep_s
            time_tot += time.time() - t_tot_s

        model_path = f'./MODEL/{args.exp_id}/{args.exp_file}_seed_{seed}_{e}.pt'
        torch.save(model.state_dict(), model_path)

        t_val_s = time.time()
        val_ap, val_auc, val_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', evaluation=False)
        time_val = time.time() - t_val_s
        # args.logger.debug(f'Epoch: {e}, train_loss: {total_loss:.1f}, total_time: {(time_tot + time_val):.2f}s (sample: {time_sample:.2f}s, prep: {time_prep:.2f}s, val: {time_val:.2f}s)')
        args.logger.debug(f'Epoch: {e}, train_loss: {total_loss:.1f}, val_ap: {val_ap:.4f}, val_auc: {val_auc:.4f}, total_time: {(time_tot + time_val):.2f}s (sample: {time_sample:.2f}s, prep: {time_prep:.2f}s, val: {time_val:.2f}s)')

    model.eval()
    # val_ap, val_auc, val_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', evaluation=False)
    test_ap, test_auc, test_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='test', evaluation=False)
    torch.cuda.empty_cache()
    
    return val_ap, val_auc, val_hit, test_ap, test_auc, test_hit


def robust_train_model_link_pred_v2(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, seed=0):
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    if args.data != "UCI" and args.data != "BITCOIN":
        neg_link_sampler = NegLinkSamplerDST(df.dst.values)
    else:
        src_set = set(df.src.values)
        dst_set = set(df.dst.values)
        node_set = src_set.union(dst_set)
        neg_link_sampler = NegLinkSamplerDST(node_set)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

    if args.scheduler == "cosine":
        threshold_scheduler = lambda epoch: args.threshold_s + (1 - np.cos(epoch * np.pi / train_param['epoch'])) * (args.threshold_e - args.threshold_s) * 0.5
    elif args.scheduler == "linear":
        threshold_scheduler = lambda epoch: args.threshold_s + (epoch / train_param['epoch']) * (args.threshold_e - args.threshold_s)
    elif args.scheduler == "fix":
        threshold_scheduler = lambda epoch: args.cosine_threshold

    embedding_box = None
    if args.temporal > 0:
        embedding_box = EmbeddingBox(g['indptr'].shape[0] - 1, model.dim_out)
        embedding_box.move_to_gpu()
        
    for e in range(1, train_param['epoch'] + 1):
        threshold = threshold_scheduler(e - 1)
        if args.robust == "cosine":
            threshold = args.cosine_threshold
        time_filter = 0
        time_mf = 0
        time_sample = 0
        time_prep = 0
        time_tot = 0
        time_temp = 0
        total_loss = 0

        clean_g = copy.deepcopy(g)
        clean_df = df.copy()
        train_df = df[df['ext_roll'] == 0]

        # training
        model.train()

        if sampler is not None:
            sampler.update_dataset(clean_g['indptr'], clean_g['indices'], clean_g['eid'], clean_g['ts'])
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
            model.memory_updater.last_upeated_ts = None
            model.memory_updater.last_upeated_memory = None

        if embedding_box is not None:
            embedding_box.reset()

        if args.mf:
            prev_dfs = deque([], args.mf_window - 1)
            prev_dfs.clear()
        
        num_edge_filter = 0
        for i_rows, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            t_tot_s = time.time()

            ### Filtering
            t_filter_s = time.time()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)
            if sampler is not None:
                sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            with torch.no_grad():
                node_emb = model.get_embeddings(mfgs)
                pred_pos, _ = model.edge_predictor(node_emb, neg_samples=0)

            if args.robust == "proposed":
                edge_score = torch.sigmoid(pred_pos.squeeze()).cpu().numpy()
                remove_df = rows[edge_score < threshold]
                num_edge_filter += len(remove_df)
            elif args.robust == "cosine":
                src_emb = node_emb[:len(rows)]
                dst_emb = node_emb[len(rows):len(rows) * 2]
                edge_score = F.cosine_similarity(src_emb, dst_emb).cpu().numpy()
                remove_df = rows[edge_score < args.cosine_threshold]
                num_edge_filter += len(remove_df)

            if len(remove_df) > 0:
                clean_g, clean_df = remove_noise(clean_g, clean_df, remove_df)
            if sampler is not None:
                sampler.update_dataset(clean_g['indptr'], clean_g['indices'], clean_g['eid'], clean_g['ts'])
                sampler.reset()
            rows = rows.drop(remove_df.index)
            time_filter += time.time() - t_filter_s
            if len(rows) == 0:
                continue

            weights = torch.ones(len(rows)).cuda()

            ## Train after filtering
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples=1)]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                time_sample += ret[0].sample_time()
            t_prep_s = time.time()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            time_prep += time.time() - t_prep_s
                
            ## Default Loss function
            optimizer.zero_grad()
            node_emb = model.get_embeddings(mfgs)
            pred_pos, pred_neg = model.edge_predictor(node_emb, neg_samples=1)
            if len(rows) == 1:
                pass 
            else:
                pred_pos, pred_neg = pred_pos.squeeze(), pred_neg.squeeze()
            
            ## TODO: 
            loss_pos = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos), reduction="none")
            loss_pos = torch.mean(loss_pos * weights)
            loss_neg = F.binary_cross_entropy_with_logits(pred_neg, torch.zeros_like(pred_neg), reduction="none")
            loss_neg = torch.mean(loss_neg * weights)

            # loss_pos = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos))
            # loss_neg = F.binary_cross_entropy_with_logits(pred_neg, torch.zeros_like(pred_neg))

            loss_temporal = 0
            t_temp_s = time.time()
            if args.temporal > 0:
                true_src_dst = root_nodes[:2 * len(rows)]
                tmp_before_emb = embedding_box.node_embedding[true_src_dst]
                tmp_before_ts = embedding_box.node_embedding_ts[true_src_dst]
                tmp_node_mask = embedding_box.node_activated[true_src_dst]
                tmp_cur_emb = node_emb[:2 * len(rows), :]
                tmp_cur_ts = torch.from_numpy(ts[:2 * len(rows)]).cuda()
                tmp_before_emb = tmp_before_emb[tmp_node_mask]
                tmp_before_ts = tmp_before_ts[tmp_node_mask]
                tmp_cur_emb = tmp_cur_emb[tmp_node_mask]
                tmp_cur_ts = tmp_cur_ts[tmp_node_mask]
                if tmp_node_mask.sum() != 0:
                    tmp_time_diff = tmp_cur_ts - tmp_before_ts
                    # tmp_weight = 1 / (1 + torch.log(tmp_time_diff + 1e-6))
                    tmp_weight = torch.exp(-args.theta * tmp_time_diff)
                    loss_temporal = - tmp_weight * F.cosine_similarity(tmp_cur_emb, tmp_before_emb)
                    loss_temporal = loss_temporal.mean()
            time_temp += time.time() - t_temp_s
            
            if embedding_box is not None:
                true_src_dst = root_nodes[:2 * len(rows)]
                embedding_box.node_embedding[true_src_dst] = node_emb[:2 * len(rows), :].detach()
                embedding_box.node_embedding_ts[true_src_dst] = torch.from_numpy(ts[:2 * len(rows)]).cuda()
                embedding_box.node_activated[true_src_dst] = True

            loss = loss_pos + loss_neg + args.temporal * loss_temporal
            loss.backward()
            optimizer.step()

            total_loss += float(loss_pos + loss_neg) * len(pred_pos)
            if args.temporal > 0:
                total_loss += float(loss_temporal) * int(tmp_node_mask.sum())
            # print(f"iter: {i_rows}, loss_pos: {loss_pos:.4f}, loss_neg: {loss_neg:.4f}, loss_temporal: {loss_temporal:.4f}, time: {(time.time() - t_tot_s):.2f}s")

            t_prep_s = time.time()
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors': ## for APAN
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=1)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=1)
            time_prep += time.time() - t_prep_s
            time_tot += time.time() - t_tot_s

        len_filtered_df = len(clean_df[clean_df['ext_roll'] == 0])
        cleaning_rate = len_filtered_df / len(train_df) * 100
        diff = len(train_df) - len_filtered_df

        model_path = f'./MODEL/{args.exp_id}/{args.exp_file}_seed_{seed}_{e}.pt'
        torch.save(model.state_dict(), model_path)

        t_val_s = time.time()
        val_ap, val_auc, val_hit, val_g, val_df = robust_link_pred_evaluation_v2(node_feats, edge_feats, clean_g, clean_df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', evaluation=False)
        # val_ap, val_auc, val_hit = 0, 0, 0
        time_val = time.time() - t_val_s

        # args.logger.debug(f'Epoch: {e}, threshold: {threshold:.3f}, train_loss: {total_loss:.1f}, best_auc: {best_auc:.4f}, val_ap: {ap:.4f}, val_auc: {auc:.4f}, val_hit: {hit:.4f}, test_ap: {test_ap:.4f}, test_auc: {test_auc:.4f}, test_hit: {test_hit:.4f}, len_filtered_df: {len_filtered_df}/{len(train_df)}({cleaning_rate:.1f}%), diff: {diff}, num_emb_filter: {num_emb_filter}, total_time: {(time_tot + time_val):.2f}s (filter: {time_filter:.2f}s, mf:{time_mf:.2f}s, sample: {time_sample:.2f}s, prep: {time_prep:.2f}s, temp: {time_temp:.2f}s, val: {time_val:.2f}s)')
        args.logger.debug(f'Epoch: {e}, threshold: {threshold:.3f}, train_loss: {total_loss:.1f}, val_ap: {val_ap:.4f}, val_auc: {val_auc:.4f}, val_hit: {val_hit:.4f}, len_filtered_df: {len_filtered_df}/{len(train_df)}({cleaning_rate:.1f}%), diff: {diff}, num_edge_filter: {num_edge_filter}, total_time: {(time_tot + time_val):.2f}s (filter: {time_filter:.2f}s, sample: {time_sample:.2f}s, prep: {time_prep:.2f}s, temp: {time_temp:.2f}s, val: {time_val:.2f}s)')

    model.eval()
    # val_ap, val_auc, val_hit, val_g, val_df = robust_link_pred_evaluation_v2(node_feats, edge_feats, clean_g, clean_df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', evaluation=False)
    test_ap, test_auc, test_hit, _, _ = robust_link_pred_evaluation_v2(node_feats, edge_feats, val_g, val_df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='test', evaluation=False)
    # test_ap, test_auc, test_hit = 0, 0, 0
    return val_ap, val_auc, val_hit, test_ap, test_auc, test_hit


def remove_noise(g, df, remove_df):
    df = df.drop(remove_df.index)
    idx_fix = np.zeros_like(g['indptr'])
    idx_rmvs = []
    for i, row in remove_df.iterrows():
        src = int(row['src'])
        dst = int(row['dst'])
        ts = row['time']
        eid = int(row['Unnamed: 0'])

        for node in [src, dst]:
            idx_start = g['indptr'][node]
            idx_end = g['indptr'][node + 1]
            idx_rmv = idx_start + np.where(g['eid'][idx_start:idx_end] == eid)[0][0]
            idx_rmvs.append(idx_rmv)
            idx_fix[node + 1:] += 1
        
    idx_rmvs = np.array(idx_rmvs)
    mask = np.ones_like(g['eid']).astype(np.bool)
    mask[idx_rmvs] = False

    g['indptr'] = g['indptr'] - idx_fix
    g['indices'] = g['indices'][mask]
    g['eid'] = g['eid'][mask]
    g['ts'] = g['ts'][mask]

    return g, df 


@torch.no_grad()
def link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', seed=None, evaluation=False):
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    if args.data != "UCI" and args.data != "BITCOIN":
        neg_link_sampler = NegLinkSamplerDST(df.dst.values, seed=seed)
    else:
        src_set = set(df.src.values)
        dst_set = set(df.dst.values)
        node_set = src_set.union(dst_set)
        neg_link_sampler = NegLinkSamplerDST(node_set, seed=seed)

    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    hits = list()

    if mode == 'val':
        eval_df = df[df['ext_roll'] == 1]
        neg_samples = negs
        if evaluation and args.data == "MOOC": 
            neg_samples = len(neg_link_sampler.dsts) - 1
    elif mode == 'test':
        eval_df = df[df['ext_roll'] == 2]
        neg_samples = negs
        if evaluation and args.data == "MOOC": 
            neg_samples = len(neg_link_sampler.dsts) - 1
    elif mode == 'train':
        eval_df = df[df['ext_roll'] == 0]

    with torch.no_grad():
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            if evaluation and args.data == "MOOC": 
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v3(rows.dst.values)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            elif evaluation and args.data != "MOOC":
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v2(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            else:
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)

            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])

            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).squeeze().sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)

            #### TODO: only test for non adv
            if args.attack != "none" and mode == "test":
                pos_mask = (rows['adv'] == 0).values
                neg_mask = np.tile(pos_mask, neg_samples)
                pos_neg_mask = np.concatenate((pos_mask, neg_mask))
                pred_pos = pred_pos[pos_mask]
                pred_neg = pred_neg[neg_mask]
                y_pred = y_pred[pos_neg_mask]
                y_true = y_true[pos_neg_mask]

            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                ranks = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1
                aucs_mrrs.append(torch.reciprocal(ranks).type(torch.float))
                hits.append(ranks <= 10)
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)

    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
        hit10 = float(torch.cat(hits).sum() / len(torch.cat(hits)))
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
        hit10 = 0
    return ap, auc_mrr, hit10


@torch.no_grad()
def robust_link_pred_evaluation_v2(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', seed=None, evaluation=False, threshold=0.9):
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    if args.data != "UCI" and args.data != "BITCOIN":
        neg_link_sampler = NegLinkSamplerDST(df.dst.values, seed=seed)
    else:
        src_set = set(df.src.values)
        dst_set = set(df.dst.values)
        node_set = src_set.union(dst_set)
        neg_link_sampler = NegLinkSamplerDST(node_set, seed=seed)
    # neg_link_sampler = NegLinkSamplerDST(df.dst.values)

    neg_samples = 1
    model.eval()
    aps = []
    aucs_mrrs = []
    hits = []

    if mode == 'val':
        eval_df = df[df['ext_roll'] == 1]
        neg_samples = negs
        if evaluation and args.data == "MOOC": 
            neg_samples = len(neg_link_sampler.dsts) - 1
    elif mode == 'test':
        eval_df = df[df['ext_roll'] == 2]
        neg_samples = negs
        if evaluation and args.data == "MOOC": 
            neg_samples = len(neg_link_sampler.dsts) - 1
    elif mode == 'train':
        eval_df = df[df['ext_roll'] == 0]

    if sampler is not None:
        sampler.update_dataset(g['indptr'], g['indices'], g['eid'], g['ts'])
        sampler.reset()

    for i_rows, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):

        if mode == "train":
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            ts = np.tile(rows.time.values, 2).astype(np.float32)

            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            with torch.no_grad():
                # pred_pos, pred_neg = model(mfgs, neg_samples=0)
                node_emb = model.get_embeddings(mfgs)
                pred_pos, _ = model.edge_predictor(node_emb, neg_samples=0)

            if args.robust == "proposed":
                edge_score = torch.sigmoid(pred_pos.squeeze()).cpu().numpy()
                remove_df = rows[edge_score < args.threshold_e]
                # remove_df = rows[edge_score < threshold]
            elif args.robust == "cosine":
                src_emb = node_emb[:len(rows)]
                dst_emb = node_emb[len(rows):len(rows) * 2]
                edge_score = F.cosine_similarity(src_emb, dst_emb).cpu().numpy()
                remove_df = rows[edge_score < args.cosine_threshold]

            if len(remove_df) > 0:
                g, df = remove_noise(g, df, remove_df)
            if sampler is not None:
                sampler.update_dataset(g['indptr'], g['indices'], g['eid'], g['ts'])
                sampler.reset()
            rows = rows.drop(remove_df.index)
            if len(rows) == 0:
                continue

            # root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            # ts = np.tile(rows.time.values, 2).astype(np.float32)
            if evaluation and args.data == "MOOC":
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v3(rows.dst.values)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            elif evaluation and args.data != "MOOC":
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v2(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            else:
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)

            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
        
            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)

            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                ranks = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1
                aucs_mrrs.append(torch.reciprocal(ranks).type(torch.float))
                hits.append(ranks <= 10)
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            # aps.append(0)
            # aucs_mrrs.append(0)

            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
    
        elif mode == "val" or mode == "test":
            if evaluation and args.data == "MOOC":
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v3(rows.dst.values)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            elif evaluation and args.data != "MOOC":
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v2(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            else:
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)

            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
        
            with torch.no_grad():
                node_emb = model.get_embeddings(mfgs)
                pred_pos, pred_neg = model.edge_predictor(node_emb, neg_samples=neg_samples)

            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)

            edge_score = torch.sigmoid(pred_pos.squeeze()).cpu().numpy()

            #### TODO: only test for non adv
            if args.attack != "none" and mode == "test":
                pos_mask = (rows['adv'] == 0).values
                neg_mask = np.tile(pos_mask, neg_samples)
                pos_neg_mask = np.concatenate((pos_mask, neg_mask))
                pred_pos = pred_pos[pos_mask]
                pred_neg = pred_neg[neg_mask]
                y_pred = y_pred[pos_neg_mask]
                y_true = y_true[pos_neg_mask]

            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                ranks = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1
                aucs_mrrs.append(torch.reciprocal(ranks).type(torch.float))
                hits.append(ranks <= 10)
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            ####### Filtering
            if args.robust == "proposed":
                remove_df = rows[edge_score < args.threshold_e]
                # remove_df = rows[edge_score < threshold]
            elif args.robust == "cosine":
                src_emb = node_emb[:len(rows)]
                dst_emb = node_emb[len(rows):len(rows) * 2]
                edge_score = F.cosine_similarity(src_emb, dst_emb).cpu().numpy()
                remove_df = rows[edge_score < args.cosine_threshold]

            # remove_df = rows[edge_score < args.threshold_e]
            if len(remove_df) > 0:
                g, df = remove_noise(g, df, remove_df)
            if sampler is not None:
                sampler.update_dataset(g['indptr'], g['indices'], g['eid'], g['ts'])
                sampler.reset()
            rows = rows.drop(remove_df.index)
            if len(rows) == 0:
                continue

            ##### Update mailbox
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            ts = np.tile(rows.time.values, 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs, neg_samples=0)

            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=0)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=0)

    ap = float(torch.tensor(aps, dtype=torch.float).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
        hit10 = float(torch.cat(hits).sum() / len(torch.cat(hits)))
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs, dtype=torch.float).mean())
        hit10 = 0
    return ap, auc_mrr, hit10, g, df
