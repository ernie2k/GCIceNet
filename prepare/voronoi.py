import math
import re

import cython
cimport cython

from libc.math cimport sqrt, abs, pi, acos

import numpy as np
cimport numpy as np
import MDAnalysis as md
from tqdm import tqdm
from util import(square_distance_matrix, distance, square_distance, between_angle, angle, pbc, pbc_mol)
import tess


def voronoi(universe, run_vol=False, run_aspher=False, skip_atom=0):
    dimension_box_vector = universe.trajectory[0].dimensions[:3]
    number_frame = len(universe.trajectory)
    
    ow_atoms = universe.select_atoms("(not name H*) and (not name M*)")
    number_ow_atoms = len(ow_atoms)
    
    volume_matrix = []
    asphericity_matrix = []
    out_dict = {}
    
    for i, ts in tqdm(enumerate(universe.trajectory), total=number_frame):
        position_ow_atoms_matrix = ow_atoms.positions
        center = tess.Container(position_ow_atoms_matrix,
                               limits=dimension_box_vector,
                               periodic=True)
        
        volume_vector = []
        asphericity_vector = []
        
        if run_vol == True:
            for j in range(skip_atom, number_ow_atoms):
                voronoi = center[j]
                volume = voronoi.volume()
                volume_vector.append(volume)
                        
        if run_aspher == True:
            for j in range(skip_atom, number_ow_atoms):
                voronoi = center[j]
                volume = voronoi.volume()
                area = voronoi.surface_area()
                asphericity = area**3/36./np.pi/volume**2
                asphericity_vector.append(asphericity)
                
        volume_matrix.append(volume_vector)
        asphericity_matrix.append(asphericity_vector)

    out_dict['volume'] = volume_matrix
    out_dict['asphericity'] = asphericity_matrix
    
    return out_dict

        

cpdef structure(universe, ofilename,  int skip_frame=1, float r_max=10.):
    
    cdef int i, j, k, l, m
    cdef int hb, hb_ow, hb_mol, hb_flag
    cdef int number_frame, number_ow, apm
    cdef int index, index_ow, index_h, start_index_ow, index_hb, index_i, index_j
    cdef int idx_1d, idx_x, idx_y, idx_z
    cdef float x, y, z, y_min, y_max, z_min, z_max, dr
    cdef float kth_distance, d5, q_tet, zeta, distance_first_no_hb, distance_last_hb
    cdef float hb_r_cut, cos_hda_angle_cut
    cdef float dist1, dist2, angle1, angle2
    cdef float hydrogen_donor_acceptor_angle, cos_angle
    #cdef long[:] index_sorted_square_distance_ow_vector
    #cdef float[:] dimension_box_vector, square_distance_ow_vector
    #cdef float[:] position_ow_vector, position_i_vector, position_o_vector, position_h_vector, position_ow_vector


    cdef float[:,:] position_h2o_vector, position_h2o_i_vector, position_h2o_j_vector
    cdef float[:,:] position_ow_matrix, position_h2o_matrix, square_distance_ow_matrix
    
    dimension_box_vector = universe.trajectory[0].dimensions
    number_frame = len(universe.trajectory)
    
    ow = universe.select_atoms("name OW")
    h2o = universe.select_atoms("resname SOL")

    number_ow = len(ow)
    number_h2o = len(h2o)

    apm = number_h2o // number_ow
    print("atom per molecule: {}".format(apm))


    cos_hda_angle_cut = sqrt(3)*0.5

    num_data = 4

    data_mat = []
    for i in range(num_data):
        data_mat.append([])
    
    for i in tqdm(range(number_frame)):
        if i % skip_frame != 0:
            continue

        ts = universe.trajectory[i]

        dimension_box_vector = ts.dimensions
        
        position_ow_matrix = ow.positions        
        position_h2o_matrix = h2o.positions
        
        square_distance_ow_matrix = np.zeros((number_ow, number_ow), dtype=np.float32)
        square_distance_matrix(position_ow_matrix,
                              dimension_box_vector,
                              square_distance_ow_matrix,
                              r_max,
                              number_ow)
             
        for j in range(number_ow):            
            square_distance_ow_vector = square_distance_ow_matrix[j]

            index_sorted_square_distance_ow_vector = np.argsort(square_distance_ow_vector, kind='mergesort') 

            sorted_square_distance_ow_vector = np.sort(square_distance_ow_vector, kind='mergesort')

            position_ow_vector = position_ow_matrix[j]

            
            # d5
            data_mat[0].append(sqrt(sorted_square_distance_ow_vector[5]))

            # q_tet
            q_tet = 0.
            for k in range(1,4):
                index_i = index_sorted_square_distance_ow_vector[k]
                position_i_vector = position_ow_matrix[index_i]
                pbc_position_i_vector = np.zeros(3, dtype=np.float32)
                pbc_position_i_vector = pbc(position_i_vector,
                                        position_ow_vector,
                                        dimension_box_vector)

                for l in range(k+1,5):
                    index_j = index_sorted_square_distance_ow_vector[l]
                    position_j_vector = position_ow_matrix[index_j]
                    pbc_position_j_vector = np.zeros(3, dtype=np.float32)
                    pbc_position_j_vector = pbc(position_j_vector,
                                                position_ow_vector,
                                                dimension_box_vector)

                    cos_angle = angle(position_ow_vector,
                                      pbc_position_i_vector,
                                      pbc_position_j_vector)

                    q_tet += (cos_angle+1./3.)**2

            q_tet = -0.375*q_tet
            q_tet = 1 + q_tet

            data_mat[1].append(q_tet)

            # LSI
            lsi = 0.
            lsi_distance_vector = []
            for k in range(1, number_ow):
                lsi_distance_vector.append(sqrt(sorted_square_distance_ow_vector[k]))
                if sorted_square_distance_ow_vector[k] > 3.7*3.7:
                    break

            diff_lsi_distance_vector = []
            for k in range(len(lsi_distance_vector)-1):
                diff_lsi_distance_vector.append(lsi_distance_vector[k+1] - lsi_distance_vector[k])

            average_diff = np.mean(diff_lsi_distance_vector)

            for k in range(len(diff_lsi_distance_vector)):
                lsi += (diff_lsi_distance_vector[k] - average_diff)**2

            if len(diff_lsi_distance_vector) == 0:
                lsi = 0
            else:
                lsi /= len(diff_lsi_distance_vector)

            data_mat[2].append(lsi)

            
            # zeta
            hb_candidate_index_vector = sorted_square_distance_ow_vector[sorted_square_distance_ow_vector < 3.5**2]
            hb_candidate_index_vector = index_sorted_square_distance_ow_vector[:len(hb_candidate_index_vector)+1]

            hb_existence_vector = np.zeros(len(hb_candidate_index_vector))

            distance_first_no_hb = 0.
            distance_last_hb = 0.
            for k in range(1, len(hb_candidate_index_vector)-1):
                donor_index = hb_candidate_index_vector[0]
                acceptor_index = hb_candidate_index_vector[k]

                donor_position_matrix = position_h2o_matrix[apm*donor_index:apm*donor_index+3]
                acceptor_position_matrix = position_h2o_matrix[apm*acceptor_index:apm*acceptor_index+3]

                acceptor_position_matrix = pbc_mol(acceptor_position_matrix,
                                                   donor_position_matrix[0],
                                                   dimension_box_vector)

                for l in range(2):
                    hydrogen_donor_acceptor_angle = angle(donor_position_matrix[0],
                                                         donor_position_matrix[l+1],
                                                         acceptor_position_matrix[0])
                    if hydrogen_donor_acceptor_angle > cos_hda_angle_cut:
                        hb_existence_vector[k] = 1
                        break

                if hb_existence_vector[k] == 0:
                    for l in range(2):
                        hydrogen_donor_acceptor_angle = angle(acceptor_position_matrix[0],
                                                             acceptor_position_matrix[l+1],
                                                             donor_position_matrix[0])
                        if hydrogen_donor_acceptor_angle > cos_hda_angle_cut:
                            hb_existence_vector[k] = 1
                            break

            for k in range(1, len(hb_existence_vector)):
                if hb_existence_vector[k] == 0:
                    distance_first_no_hb = sqrt(sorted_square_distance_ow_vector[k])
                    break
            distance_last_hb = 0
            for k in range(len(hb_existence_vector)-1, 0, -1):
                if hb_existence_vector[k] == 1:
                    distance_last_hb = sqrt(sorted_square_distance_ow_vector[k])
                    break

            zeta = distance_first_no_hb - distance_last_hb
            if zeta > r_max:
                zeta = 0.
            
            data_mat[3].append(zeta)


    data_mat = np.transpose(data_mat)

    np.savez(ofilename,
            data = data_mat,
            )


