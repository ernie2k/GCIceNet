import sys
import numpy as np
import scipy
import MDAnalysis as md
import argparse

import tqdm

import time

from util import (square_distance_matrix, pbc, cartesian_to_spherical)


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

r_max = 0


# l=1 to l=12
l_vector = [4, 6]
n_l = len(l_vector)

q_matrix3 = np.zeros((n_l, num_frames, num_ow))


# start
for i, ts in tqdm.tqdm(enumerate(u.trajectory), total=num_frames):

    dimension_box_vector = ts.dimensions[:3]

    position_ow_matrix = ow.positions

    square_distance_ow_matrix = np.zeros((num_ow, num_ow), dtype=np.float32)
    square_distance_matrix(position_ow_matrix,
                           dimension_box_vector,
                           square_distance_ow_matrix,
                           r_max,
                           num_ow)   

    # qlm_matrix
    qlm_matrix3 = []
    for j in range(n_l):
        qlm_matrix3.append([])
    
    for j in range(num_ow):

        square_distance_ow_vector = square_distance_ow_matrix[j]

        sorted_square_distance_ow_vector = np.sort(square_distance_ow_vector,
                                                   kind='mergesort')
        index_sorted_square_distance_ow_vector = np.argsort(square_distance_ow_vector,
                                                            kind='mergesort')

        angle_matrix = []

        position_ow_vector = position_ow_matrix[j]
        
        spherical_coordinate_matrix = []


        for k in range(1, n_neighbor+1):

            index_k = index_sorted_square_distance_ow_vector[k]

            position_ow_i_vector = position_ow_matrix[index_k]
            position_ow_i_vector = pbc(position_ow_i_vector,
                                       position_ow_vector,
                                       dimension_box_vector)

            x, y, z = position_ow_i_vector - position_ow_vector

            r, theta, pi = cartesian_to_spherical(x, y, z)

            spherical_coordinate_matrix.append([r, theta, pi])

        spherical_coordinate_matrix = np.array(spherical_coordinate_matrix)

        for k in range(n_l):
            ylm_matrix = get_ylm_matrix(spherical_coordinate_matrix, l=l_vector[k])
            qlm_vector = ylm_to_qlm(ylm_matrix, l=l_vector[k])
            qlm_matrix3[k].append(qlm_vector)
            
        

    qlm_average_matrix3 = []
    for j in range(n_l):
        qlm_average_matrix = qlm_to_qlm_average(qlm_matrix3[j], square_distance_ow_matrix, n_neighbor, l=l_vector[j])
        qlm_average_matrix3.append(qlm_average_matrix)

            

    for j in range(n_l):
        q_vector = qlm_average_to_ql(qlm_average_matrix3[j], l=l_vector[j])
        q_matrix3[j, i] = q_vector


    
np.savez(ofilename,
        q4=q_matrix3[0],
        q6=q_matrix3[1],)



















