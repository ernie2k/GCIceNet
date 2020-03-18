import math
import argparse
import pickle

import numpy as np
import scipy
import scipy.sparse as sp

import MDAnalysis as md
from tqdm import tqdm
from util import(square_distance_matrix, angle, pbc, cartesian_to_spherical)

def get_ylm_matrix(spherical_coordinate_matrix, l):
    num_vectors = len(spherical_coordinate_matrix)

    ylm_matrix = np.zeros((num_vectors, l*2+1))

    spherical_coordinate_matrix = np.array(spherical_coordinate_matrix)

    theta_vec = spherical_coordinate_matrix[:,1]
    pi_vec = spherical_coordinate_matrix[:,2]

    for j in range(0,l):
        m = j-l
        ylm_matrix[:,j] = np.sqrt(2)*(-1**m)*(scipy.special.sph_harm(m,l,pi_vec,theta_vec)).imag
    
    for j in range(l,l+1):
        ylm_matrix[:,j] = scipy.special.sph_harm(0,l,pi_vec,theta_vec).real

    for j in range(l+1,2*l+1):
        m = j-l
        ylm_matrix[:,j] = np.sqrt(2)*(-1**m)*(scipy.special.sph_harm(m,l,pi_vec,theta_vec)).real

    return ylm_matrix


def ylm_to_qlm(ylm_matrix, l):
    num_vectors = len(ylm_matrix)

    qlm_vector = np.zeros(2*l+1)

    for i in range(num_vectors):
        qlm_vector += ylm_matrix[i]

    qlm_vector /= num_vectors

    return qlm_vector

def qlm_to_qlm_average(qlm_matrix, square_distance_ow_matrix, n_neighbor, l):
    qlm_average_matrix = []

    num_ow = len(square_distance_ow_matrix)
    
    for i in range(num_ow):

        square_distance_ow_vector = square_distance_ow_matrix[i]
        
        index_sorted_square_distance_ow_vector = np.argsort(square_distance_ow_vector,
                                                            kind='mergesort')
        
        qlm_average_vector = np.zeros(2*l+1)        

        qlm_average_vector += qlm_matrix[i]
        
        for j in range(1, n_neighbor+1):
            
            index_j = index_sorted_square_distance_ow_vector[j]
            
            qlm_average_vector += qlm_matrix[index_j]
            
        qlm_average_vector /= n_neighbor+1
        
        qlm_average_matrix.append(qlm_average_vector)

    return np.array(qlm_average_matrix)
 

def qlm_average_to_ql(qlm_average_matrix, l=4):
    ql_vector = []

    num_ow = len(qlm_average_matrix)
    
    for i in range(num_ow):
        
        qlm_average_vector = qlm_average_matrix[i]
        
        ql = 0
        
        for qlm_average in qlm_average_vector:

            ql += qlm_average**2
            
        ql *= 4*np.pi/(2*l+1)
        ql = np.sqrt(ql)
        
        ql_vector.append(ql)

    return ql_vector


parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str, help='topology file')
parser.add_argument('--f', type=str, help='trajectory file')
parser.add_argument('--o', type=str, help='output file')
parser.add_argument('--adj', type=str, help='coordination adjacency file of sparse matrix')
parser.add_argument('--a_hat', type=str, help='a hat file of sparse matrix')
parser.add_argument('--n_neighbor', type=int, default=8, help='number of neighbours')

args = parser.parse_args()

tpr_filename = args.s
xtc_filename = args.f
ofilename = args.o

u = md.Universe(tpr_filename, xtc_filename)

n_frame = len(u.trajectory)

ow = u.select_atoms("name OW")

n_ow = len(ow)

r_max = 10.

d5_mat = np.zeros((n_frame, n_ow))
tet_mat = np.zeros((n_frame, n_ow))
lsi_mat = np.zeros((n_frame, n_ow))
q4_mat = np.zeros((n_frame, n_ow))
q6_mat = np.zeros((n_frame, n_ow))

o_adj = []
o_a_hat = []


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
        
        # d5
        d5_mat[i, j] = np.sqrt(sorted_sqr_dist_ow_vec[5])

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


    # q4 & q6
    q4lm_mat = []
    q6lm_mat = []

    for j in range(n_ow):

        sqr_dist_ow_vec = sqr_dist_ow_mat[j]

        sorted_sqr_dist_ow_vec = np.sort(sqr_dist_ow_vec, kind='mergesort')
        idx_sorted_sqr_dist_ow_vec = np.argsort(sqr_dist_ow_vec, kind='mergesort')

        angle_matrix = []

        pos_ow_vec = pos_ow_mat[j]
        
        sph_coord_mat = []


        for k in range(1, args.n_neighbor+1):

            idx_k = idx_sorted_sqr_dist_ow_vec[k]

            pos_ow_i_vec = pos_ow_mat[idx_k]
            pos_ow_i_vec = pbc(pos_ow_i_vec, pos_ow_vec, box)

            x, y, z = pos_ow_i_vec - pos_ow_vec

            r, theta, pi = cartesian_to_spherical(x, y, z)

            sph_coord_mat.append([r, theta, pi])

        sph_coord_mat = np.array(sph_coord_mat)

        y4lm_mat = get_ylm_matrix(sph_coord_mat, l=4)
        y6lm_mat = get_ylm_matrix(sph_coord_mat, l=6)
        q4lm_vec = ylm_to_qlm(y4lm_mat, l=4)
        q6lm_vec = ylm_to_qlm(y6lm_mat, l=6)

        q4lm_mat.append(q4lm_vec)
        q6lm_mat.append(q6lm_vec)

    q4lm_avg_mat = qlm_to_qlm_average(q4lm_mat, sqr_dist_ow_mat, args.n_neighbor, l=4)
    q6lm_avg_mat = qlm_to_qlm_average(q6lm_mat, sqr_dist_ow_mat, args.n_neighbor, l=6)

    q4_mat[i] = qlm_average_to_ql(q4lm_avg_mat, l=4)
    q6_mat[i] = qlm_average_to_ql(q6lm_avg_mat, l=6)


    # make adjacency matrix of whole system
    adj = np.zeros((n_ow, n_ow), int)
    a_hat = np.zeros((n_ow, n_ow))

    for j in range(n_ow):
        sqr_dist_ow_vec = sqr_dist_ow_mat[j]

        idx_sorted_sqr_dist_ow_vec = np.argsort(sqr_dist_ow_vec, kind='mergesort')
        sorted_sqr_dist_ow_vec = np.sort(sqr_dist_ow_vec, kind='mergesort')

        for k in range(1, n_ow):
            adj[j, idx_sorted_sqr_dist_ow_vec[k]] = 1
            if sorted_sqr_dist_ow_vec[k] > 3.5**2:
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


np.savez(ofilename,
        d5 = d5_mat,
        tet = tet_mat,
        lsi = lsi_mat,
        q4 = q4_mat,
        q6 = q6_mat,
        )

with open(args.adj, 'wb') as f:
    pickle.dump(o_adj, f)

with open(args.a_hat, 'wb') as f:
    pickle.dump(o_a_hat, f)
