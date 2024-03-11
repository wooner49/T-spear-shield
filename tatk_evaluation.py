import time
import os
import copy
from pathlib import Path
import pickle
import numpy as np
import torch

from tatk.utils import (
    set_random_seed,
    parse_argument,
    parse_config,
    pre_exp_setting,
    load_data,
    load_attacked_data,
)
from tatk.utils_train import (
    create_model_mailbox_sampler,
    link_pred_evaluation,
    robust_link_pred_evaluation_v2,
    remove_noise,
)

EXP_ID = 'AAAI'

def evaluation(args):
    pre_exp_setting(EXP_ID, args, evaluation=True)

    val_aps, val_aucs, val_hits = [], [], []
    test_aps, test_aucs, test_hits = [], [], []

    for i, seed in enumerate(args.seeds):
        set_random_seed(seed)

        if args.attack == "none":
            node_feats, edge_feats, g, df = load_data(args.data, args)
        else:
            node_feats, edge_feats, g, df = load_attacked_data(args.data, args)
    
        ## Model train
        sample_param, memory_param, gnn_param, train_param = parse_config(args.model)
        model, mailbox, sampler = create_model_mailbox_sampler(node_feats, edge_feats, g, df, sample_param, memory_param, gnn_param, train_param)

        best_epoch = 0
        best_ap = 0
        best_auc = 0
        best_hit = 0
        best_mailbox = None
        best_g = None
        best_df = None
        patience = 0

        patience_target = 10
        best_model_path = f'./MODEL/{args.exp_id}/{args.exp_file}_seed_{seed}_best_from51.pt'

        start_epoch = 1
        for e in range(start_epoch, train_param['epoch'] + 1):
            model_path = f'./MODEL/{args.exp_id}/{args.exp_file}_seed_{seed}_{e}.pt'
            model.load_state_dict(torch.load(model_path))
            model.eval()
            t_eval_s = time.time() 

            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                model.memory_updater.last_updated_nid = None
                model.memory_updater.last_updated_ts = None
                model.memory_updater.last_updated_memory = None

            if args.robust == "none" or args.robust == "svd":
                if mailbox is not None:
                    link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='train', seed=seed)
                val_ap, val_auc, val_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=args.eval_neg_samples, mode='val', evaluation=True, seed=seed)

            elif args.robust == "proposed" or args.robust == "cosine":
                orig_g = copy.deepcopy(g)
                orig_df = df.copy()
                if mailbox is not None:
                    _, _, _, train_g, train_df = robust_link_pred_evaluation_v2(node_feats, edge_feats, orig_g, orig_df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='train', seed=seed, threshold=threshold)
                val_ap, val_auc, val_hit, val_g, val_df = robust_link_pred_evaluation_v2(node_feats, edge_feats, train_g, train_df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=args.eval_neg_samples, mode='val', evaluation=True, seed=seed, threshold=threshold)

            t_eval_e = time.time() - t_eval_s

            if val_auc >= best_auc:
                best_epoch = e
                best_ap = val_ap
                best_auc = val_auc
                best_hit = val_hit
                if mailbox is not None:
                    best_mailbox = copy.deepcopy(mailbox)
                if args.robust == "proposed" or args.robust == "cosine":
                    best_g = copy.deepcopy(val_g)
                    best_df = val_df.copy()
                torch.save(model.state_dict(), best_model_path)
                patience = 0
            else:
                patience += 1
                
            args.logger.debug(f'Epoch: {e}, best_epoch: {best_epoch}, best_auc: {best_auc:.4f}, val_ap: {val_ap:.4f}, val_auc: {val_auc:.4f}, val_hit: {val_hit:.4f}, eval_time: {t_eval_e:.2f}s')
            # args.logger.debug(f'Epoch: {e}, best_epoch: {best_epoch}, best_ap: {best_ap:.4f}, val_ap: {val_ap:.4f}, val_auc: {val_auc:.4f}, val_hit: {val_hit:.4f}, eval_time: {t_eval_e:.2f}s')
            if e > 50 and patience > patience_target:
                args.logger.debug(f'Early Stopping')
                break
        
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        if args.robust == "none" or args.robust == "svd":
            test_ap, test_auc, test_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, best_mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=args.eval_neg_samples, mode='test', evaluation=True, seed=seed)
        elif args.robust == "proposed" or args.robust == "cosine":
            test_ap, test_auc, test_hit, _, _ = robust_link_pred_evaluation_v2(node_feats, edge_feats, best_g, best_df, model, best_mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=args.eval_neg_samples, mode='test', evaluation=True, seed=seed, threshold=threshold)

        val_aps.append(best_ap)
        val_aucs.append(best_auc)
        val_hits.append(best_hit)
        test_aps.append(test_ap)
        test_aucs.append(test_auc)
        test_hits.append(test_hit)
        args.logger.info(f'[Seed {seed}] model: {args.model}, attack: {args.attack} ({args.ptb_rate:.2f}), val_auc: {best_auc:.4f}, val_hit: {best_hit:.4f}, test_ap: {test_ap:.4f}, test_auc: {test_auc:.4f}, test_hit: {test_hit:.4f}, neg_samples: {args.eval_neg_samples}')

        if node_feats is not None:
            del node_feats
        if edge_feats is not None:
            del edge_feats
        del g, df
        
    val_ap_mean, val_ap_std = np.mean(val_aps), np.std(val_aps)
    val_auc_mean, val_auc_std = np.mean(val_aucs), np.std(val_aucs)
    val_hit_mean, val_hit_std = np.mean(val_hits), np.std(val_hits)
    test_ap_mean, test_ap_std = np.mean(test_aps), np.std(test_aps)
    test_auc_mean, test_auc_std = np.mean(test_aucs), np.std(test_aucs)
    test_hit_mean, test_hit_std = np.mean(test_hits), np.std(test_hits)
    args.logger.info(f'[Final] model: {args.model}, attack: {args.attack} ({args.ptb_rate:.2f}), seeds: {args.seeds}, val_auc: {val_auc_mean:.4f}+-{val_auc_std:.4f}, val_hit: {val_hit_mean:.4f}+-{val_hit_std:.4f}, test_ap: {test_ap_mean:.4f}+-{test_ap_std:.4f}, test_auc: {test_auc_mean:.4f}+-{test_auc_std:.4f}, test_hit: {test_hit_mean:.4f}+-{test_hit_std:.4f}, neg_samples: {args.eval_neg_samples}')


if __name__ == "__main__":
    args = parse_argument()
    evaluation(args)
