import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="WIKI")
args=parser.parse_args()

df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    ext_full_indices[dst].append(src)
    ext_full_ts[dst].append(row['time'])
    ext_full_eid[dst].append(idx)

def ext_sort(i, indices, t, eid):
    idx = np.argsort(t[i])
    indices[i] = np.array(indices[i])[idx].tolist()
    t[i] = np.array(t[i])[idx].tolist()
    eid[i] = np.array(eid[i])[idx].tolist()

for i in tqdm(range(len(ext_full_ts)), disable=False):
    ext_sort(i, ext_full_indices, ext_full_ts, ext_full_eid)

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

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

for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, np_ext_full_indices, np_ext_full_ts, np_ext_full_eid)

g = {
    'indptr': ext_full_indptr,
    'indices': np_ext_full_indices,
    'ts': np_ext_full_ts,
    'eid': np_ext_full_eid,
    'ext_full_indices': ext_full_indices,
    'ext_full_ts': ext_full_ts,
    'ext_full_eid': ext_full_eid
}

with open(f'DATA/{args.data}/ext_full.pkl', 'wb') as f:
    pickle.dump(g, f)
