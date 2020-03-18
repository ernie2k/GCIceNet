import sys
import numpy as np
import scipy
import MDAnalysis as md
import argparse

import tqdm

import time

from util import (square_distance_matrix, pbc, distance, angle, f_cut, g2, g4)



parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--o', type=str, help='output file')
parser.add_argument('--n_neighbor', type=int, default=8, help='number of neighbours')

args = parser.parse_args()

tpr_filename = args.s
xtc_filename = args.f
ofilename = args.o
n_neighbor = args.n_neighbor


u = md.Universe(tpr_filename, xtc_filename)

num_frames = len(u.trajectory)

ow = u.select_atoms('name OW')

num_ow = len(ow)

r_cut = 12.


# parameter space
eta2_vec = np.array([0.001, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4])
num_G2 = len(eta2_vec)

eta4_vec = np.array([0.0001, 0.003, 0.008, 0.015, 0.025, 0.045, 0.08])
lambda_vec = np.array([-1,1])
zeta_vec = np.array([1.,2.,4.,16.])
num_G4 = len(eta4_vec)*len(lambda_vec)*len(zeta_vec)


# Setting output
G2_mat = []
G4_mat = []




for i, ts in tqdm.tqdm(enumerate(u.trajectory), total=num_frames):

    dimension_box_vector = ts.dimensions[:3]

    position_ow_matrix = ow.positions

    square_distance_ow_matrix = np.zeros((num_ow, num_ow), dtype=np.float32)
    square_distance_matrix(position_ow_matrix,
                           dimension_box_vector,
                           square_distance_ow_matrix,
                           r_cut*2,
                           num_ow)   

    for j in range(num_ow):

        square_distance_ow_vector = square_distance_ow_matrix[j]

        sorted_square_distance_ow_vector = np.sort(square_distance_ow_vector,
                                                   kind='mergesort')
        index_sorted_square_distance_ow_vector = np.argsort(square_distance_ow_vector,
                                                            kind='mergesort')


        index_list = np.zeros(n_neighbor+1, int)
        for k in range(n_neighbor+1):
            index_list[k] = index_sorted_square_distance_ow_vector[k]



        #for k in range(n_neighbor+1):
        for k in range(1):
            position_ow_vector = position_ow_matrix[index_list[0]]

            # G2
            G2_vec = np.zeros(num_G2)

            for l in range(1, n_neighbor+1):
                position_ow_i_vector = position_ow_matrix[index_list[l]]
                position_ow_i_vector = pbc(position_ow_i_vector,
                                           position_ow_vector,
                                           dimension_box_vector)

                r = distance(position_ow_vector, position_ow_i_vector)


                for m in range(num_G2):
                    G2_vec[m] += g2(eta2_vec[m],r)*f_cut(r,r_cut)


            G2_mat.append(G2_vec)


            # G4
            G4_vec = np.zeros(num_G4)
            for l in range(1, n_neighbor):
                position_ow_i_vector = position_ow_matrix[index_list[l]]
                position_ow_i_vector = pbc(position_ow_i_vector,
                                           position_ow_vector,
                                           dimension_box_vector)

                for m in range(l+1, n_neighbor+1):
                    position_ow_j_vector = position_ow_matrix[index_list[m]]
                    position_ow_j_vector = pbc(position_ow_j_vector,
                                               position_ow_vector,
                                               dimension_box_vector)

                    r1 = distance(position_ow_vector, position_ow_i_vector)
                    r2 = distance(position_ow_vector, position_ow_j_vector)
                    r3 = distance(position_ow_i_vector, position_ow_j_vector)


                    cos_angle = angle(position_ow_vector,
                                      position_ow_i_vector,
                                      position_ow_j_vector)


                    index = 0
                    for n in range(len(eta4_vec)):
                        for o in range(len(lambda_vec)):
                            for p in range(len(zeta_vec)):
                                G4_vec[index] += g4(eta4_vec[n],lambda_vec[o],zeta_vec[p],
                                                    r1, r2, r3, cos_angle)*f_cut(r1,r_cut)*f_cut(r2,r_cut)*f_cut(r3,r_cut)
                                index += 1

            G4_mat.append(G4_vec)






np.savez(ofilename,
         G2=G2_mat,
         G4=G4_mat)



