import sys
import numpy as np
import scipy.sparse as sp
import argparse
import pickle

import MDAnalysis as md
import tqdm

import networkx as nx

from util import (square_distance_matrix, determine_coordination)

parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--adj', type=str, help='coordination adjacency file of sparse matrix')
parser.add_argument('--a_hat', type=str, help='a hat file of sparse matrix')

args = parser.parse_args()

tpr_filename = args.s
xtc_filename = args.f

u = md.Universe(tpr_filename, xtc_filename)

h2o = u.select_atoms('resname SOL')
ow = u.select_atoms('name OW')

num_frame = len(u.trajectory)
num_h2o = len(h2o)
n_ow = len(ow)
apm = int(num_h2o/n_ow)

o_adj_cn = []
o_a_hat = []

r_max = 12.

for i in tqdm.tqdm(range(num_frame)):
    
    ts = u.trajectory[i]
    
    box = ts.dimensions
    pos_ow_mat = ow.positions
    

    # make adjacency matrix of whole system
    adj_cn = np.zeros((n_ow, n_ow), int)

    sqr_dist_ow_mat = np.zeros((n_ow, n_ow), dtype=np.float32)
    square_distance_matrix(pos_ow_mat, box, sqr_dist_ow_mat, r_max**2, n_ow)

    for j in range(n_ow):
        sqr_dist_ow_vec = sqr_dist_ow_mat[j]

        idx_sorted_sqr_dist_ow_vec = np.argsort(sqr_dist_ow_vec, kind='mergesort')
        sorted_sqr_dist_ow_vec = np.sort(sqr_dist_ow_vec, kind='mergesort')

        for k in range(1, n_ow):
            adj_cn[j, idx_sorted_sqr_dist_ow_vec[k]] = 1
            if sorted_sqr_dist_ow_vec[k] > 3.5**2:
                break

        '''
        for k in range(j+1, n_ow):
            #x_j = x_h2os[apm*j:apm*j+apm]
            #x_k = x_h2os[apm*k:apm*k+apm]
            x_j = x_ows[j]
            x_k = x_ows[k]
            
            adj_cn[j, k] = determine_coordination(x_j, x_k, box)
            adj_cn[k, j] = adj_cn[j, k]
        '''


    a_hat = np.zeros((n_ow, n_ow))
    buf = adj_cn + np.eye(n_ow)

    for j in range(n_ow):
        for k in range(n_ow):
            if j == k:
                a_hat[j, k] = 1/(np.sum(buf[j])+1)
            elif buf[j, k] == 0:
                pass
            else:
                a_hat[j, k] = buf[j, k] / np.sqrt((np.sum(buf[j])+1) * (np.sum(buf[k])+1))



    adj_cn = sp.coo_matrix(adj_cn)
    a_hat = sp.coo_matrix(a_hat)


    o_adj_cn.append(adj_cn)
    o_a_hat.append(a_hat)


with open(args.adj, 'wb') as f:
    pickle.dump(o_adj_cn, f)

with open(args.a_hat, 'wb') as f:
    pickle.dump(o_a_hat, f)
