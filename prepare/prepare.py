import math
import argparse
import pickle

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy
import scipy.sparse as sp

import MDAnalysis.analysis.distances
import MDAnalysis as md
from tqdm import tqdm

#import boo

def pbc(ref_pos_mat, pos_mat, box):
    box = box[:3]

    pbc_pos_mat = np.copy(pos_mat)

    for i in range(3):
        mask1 = pos_mat[:,i] - ref_pos_mat[:,i] > 0.5*box[i]
        mask2 = ref_pos_mat[:,i] - pos_mat[:,i] > 0.5*box[i]

        pbc_pos_mat[mask1,i] -= box[i]
        pbc_pos_mat[mask2,i] += box[i]
        
    return pbc_pos_mat
            


def angle(ref_pos_mat, pos1_mat, pos2_mat):
    angle_vec = np.zeros(len(ref_pos_mat))

    v1_mat = pos1_mat - ref_pos_mat
    v2_mat = pos2_mat - ref_pos_mat

    norm1_vec = np.linalg.norm(v1_mat, axis=1)
    norm2_vec = np.linalg.norm(v2_mat, axis=1)

    v1_mat /= np.tile(norm1_vec, (3,1)).T
    v2_mat /= np.tile(norm2_vec, (3,1)).T

    return inner1d(v1_mat, v2_mat)


parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, 
                    help='topology file. .tpr extension.')
parser.add_argument('--f', type=str, 
                    help='trajectory file. .xtc, .trr, .gro extensions')
parser.add_argument('--o', type=str, 
                    help='output file. .npz extension.')
parser.add_argument('--a_hat', type=str, 
                    help='a hat file of sparse matrix converted by scipy.sparse module. .pickle extension.')
args = parser.parse_args()


u = md.Universe(args.s, args.f)
ow = u.select_atoms("name OW")

n_frame = len(u.trajectory)
n_ow = len(ow)
n_feature = 7

feature_mat3 = np.zeros((n_frame, n_ow, n_feature))
dist_ow_mat = np.zeros((n_ow, n_ow))

o_adj = []
o_a_hat = []


for i in tqdm(range(n_frame)):
    ts = u.trajectory[i]

    box = ts.dimensions
    
    pos_ow_mat = ow.positions        
    
    dist_ow_mat = MDAnalysis.analysis.distances.distance_array(pos_ow_mat, pos_ow_mat, box=box)

    sorted_dist_ow_mat = np.sort(dist_ow_mat, kind='mergesort', 
                                 axis=1)
    idx_sorted_dist_ow_mat = np.argsort(dist_ow_mat, kind='mergesort', 
                                        axis=1)

    # d
    for j in range(1,5+1):
        feature_mat3[i,:,j-1] += sorted_dist_ow_mat[:,j]


    # q_tet
    q_tet_vec = np.zeros(n_ow)
    q_tet_vec.fill(1)
    for j in range(1,4):
        pos_ow_i_mat = pos_ow_mat[idx_sorted_dist_ow_mat[:,j]]
        pbc_pos_ow_i_mat = pbc(pos_ow_mat, pos_ow_i_mat, box)

        for k in range(j+1, 5):
            pos_ow_j_mat = pos_ow_mat[idx_sorted_dist_ow_mat[:,k]]
            pbc_pos_ow_j_mat = pbc(pos_ow_mat, pos_ow_j_mat, box)

            cos_angle_vec = angle(pos_ow_mat,
                                  pbc_pos_ow_i_mat,
                                  pbc_pos_ow_j_mat)

            q_tet_vec += -0.375*(cos_angle_vec+1./3.)**2

    feature_mat3[i,:,5] += q_tet_vec

    
         
    # LSI
    lsi_vec = np.zeros(n_ow)
    for j in range(n_ow):            
        dist_ow_vec = dist_ow_mat[j]
        sorted_dist_ow_vec = sorted_dist_ow_mat[j]
        pos_ow_vec = pos_ow_mat[j]

        # LSI
        lsi = 0.
        lsi_dist_vec = []
        for k in range(1, n_ow):
            lsi_dist_vec.append(sorted_dist_ow_vec[k])
            if sorted_dist_ow_vec[k] > 3.7:
                break

        diff_lsi_dist_vec = []
        for k in range(len(lsi_dist_vec)-1):
            diff_lsi_dist_vec.append(lsi_dist_vec[k+1] - lsi_dist_vec[k])

        avg_diff = np.mean(diff_lsi_dist_vec)

        for k in range(len(diff_lsi_dist_vec)):
            lsi += (diff_lsi_dist_vec[k] - avg_diff)**2

        if len(diff_lsi_dist_vec) == 0:
            lsi = 0
        else:
            lsi /= len(diff_lsi_dist_vec)

        lsi_vec[j] = lsi
    feature_mat3[i,:,6] += lsi_vec


    # bond orientation order
    #l_list = [2, 4, 6, 8, 12]
    #sorted_ngbs_dist_ow_mat = np.copy(sorted_dist_ow_mat[:,1:12])
    #idx_sorted_ngbs_dist_ow_mat = np.copy(idx_sorted_dist_ow_mat[:,1:5])
    #mask = (sorted_ngbs_dist_ow_mat>3.5) 
    ###idx_sorted_ngbs_dist_ow_mat[mask] = -10
    ###print(idx_sorted_ngbs_dist_ow_mat[:10])
    
    #for j, l in enumerate(l_list):
    #    qlm_mat = boo.ngbs2qlm(pos_ow_mat,
    #                           idx_sorted_ngbs_dist_ow_mat,
	#						   l=l, periods=box[:3])
    #    cqlm_mat = boo.coarsegrain_qlm_ngbs(qlm_mat,
    #                                        idx_sorted_ngbs_dist_ow_mat)
    #    ql_mat = boo.ql(cqlm_mat)

    #    feature_mat3[i,:,7+j] += ql_mat

       


    # make adjacency matrix of whole system
    adj = np.zeros((n_ow, n_ow), int)
    a_hat = np.zeros((n_ow, n_ow))

    for j in range(n_ow):
        dist_ow_vec = dist_ow_mat[j]

        idx_sorted_dist_ow_vec = np.argsort(dist_ow_vec, kind='mergesort')
        sorted_dist_ow_vec = np.sort(dist_ow_vec, kind='mergesort')

        for k in range(1, n_ow):
            adj[j, idx_sorted_dist_ow_vec[k]] = 1
            if sorted_dist_ow_vec[k] > 3.5:
                break

    buf = adj + np.eye(n_ow)

    for j in range(n_ow):
        for k in range(n_ow):
            if j == k:
                a_hat[j, k] = 1/(np.sum(buf[j])+1)
            elif buf[j, k] == 0:
                pass
            else:
                a_hat[j, k] = buf[j, k] / np.sqrt((np.sum(buf[j])+1) * (np.sum(buf[k])+1))

    adj = sp.coo_matrix(adj)
    a_hat = sp.coo_matrix(a_hat)

    o_adj.append(adj)
    o_a_hat.append(a_hat)

print(np.shape(feature_mat3))

np.savez(args.o, feature=feature_mat3)

with open(args.a_hat, 'wb') as f:
    pickle.dump(o_a_hat, f)
