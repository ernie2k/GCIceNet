import math
import argparse

import numpy as np
import MDAnalysis as md
from tqdm import tqdm
from util import(square_distance_matrix, angle, pbc)

parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--o', type=str, help='output file')

args = parser.parse_args()

tpr_filename = args.s
xtc_filename = args.f
ofilename = args.o

u = md.Universe(tpr_filename, xtc_filename)


box = u.trajectory[0].dimensions
n_frame = len(u.trajectory)

ow = u.select_atoms("name OW")
h2o = u.select_atoms("resname SOL")

n_ow = len(ow)
n_h2o = len(h2o)

apm = n_h2o // n_ow
print("atom per molecule: {}".format(apm))

r_max = 10.


tet_mat = np.zeros((n_frame, n_ow))

for i in tqdm(range(n_frame)):
    ts = u.trajectory[i]

    box = ts.dimensions
    
    pos_ow_mat = ow.positions        
    
    sqr_dist_ow_mat = np.zeros((n_ow, n_ow), dtype=np.float32)
    square_distance_matrix(pos_ow_mat, box, sqr_dist_ow_mat, r_max**2, n_ow)
         
    for j in range(n_ow):            
        sqr_dist_ow_vec = sqr_dist_ow_mat[j]

        idx_sorted_sqr_dist_ow_vec = np.argsort(sqr_dist_ow_vec, kind='mergesort')
        sorted_sqr_dist_ow_vec = np.sort(sqr_dist_ow_vec, kind='mergesort')

        pos_ow_vec = pos_ow_mat[j]
        
        # q_tet
        q_tet = 0.
        for k in range(1,4):
            idx_i = idx_sorted_sqr_dist_ow_vec[k]
            pos_i_vec = pos_ow_mat[idx_i]
            pbc_pos_i_vec = np.zeros(3, dtype=np.float32)
            pbc_pos_i_vec = pbc(pos_i_vec, pos_ow_vec, box)

            for l in range(k+1,5):
                idx_j = idx_sorted_sqr_dist_ow_vec[l]
                pos_j_vec = pos_ow_mat[idx_j]
                pbc_pos_j_vec = np.zeros(3, dtype=np.float32)
                pbc_pos_j_vec = pbc(pos_j_vec, pos_ow_vec, box)

                cos_angle = angle(pos_ow_vec, pbc_pos_i_vec, pbc_pos_j_vec)

                q_tet += (cos_angle+1./3.)**2

        q_tet = -0.375*q_tet
        q_tet = 1 + q_tet

        tet_mat[i, j] = q_tet



np.savez(ofilename,
        tet = tet_mat,
        )


