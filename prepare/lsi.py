import math
import argparse

import numpy as np
import MDAnalysis as md
from tqdm import tqdm
from util import(square_distance_matrix, distance, square_distance, between_angle, angle, pbc)

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


lsi_mat = np.zeros((n_frame, n_ow))

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

        # LSI
        lsi = 0.
        lsi_dist_vec = []
        for k in range(1, n_ow):
            lsi_dist_vec.append(np.sqrt(sorted_sqr_dist_ow_vec[k]))
            if sorted_sqr_dist_ow_vec[k] > 3.7*3.7:
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

        lsi_mat[i, j] = lsi


np.savez(ofilename,
        lsi = lsi_mat,
        )


