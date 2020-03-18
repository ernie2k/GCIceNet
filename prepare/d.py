import math
import argparse

import numpy as np
import MDAnalysis as md
from tqdm import tqdm
from util import(square_distance_matrix)

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

d1_mat = np.zeros((n_frame, n_ow))
d2_mat = np.zeros((n_frame, n_ow))
d3_mat = np.zeros((n_frame, n_ow))
d4_mat = np.zeros((n_frame, n_ow))
d5_mat = np.zeros((n_frame, n_ow))

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

        # d5
        d1_mat[i, j] = np.sqrt(sorted_sqr_dist_ow_vec[1])
        d2_mat[i, j] = np.sqrt(sorted_sqr_dist_ow_vec[2])
        d3_mat[i, j] = np.sqrt(sorted_sqr_dist_ow_vec[3])
        d4_mat[i, j] = np.sqrt(sorted_sqr_dist_ow_vec[4])
        d5_mat[i, j] = np.sqrt(sorted_sqr_dist_ow_vec[5])


np.savez(ofilename,
        d1 = d1_mat,
        d2 = d2_mat,
        d3 = d3_mat,
        d4 = d4_mat,
        d5 = d5_mat,
        )


