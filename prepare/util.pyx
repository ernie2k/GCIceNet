import numpy as np

import cython
cimport cython
cimport numpy as np
import scipy

from libc.math cimport sqrt, abs, atan2, pi, cos, exp
from libc.stdlib cimport malloc, free

import MDAnalysis as md
from tqdm import tqdm


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing. 
cpdef float[:] pbc(float[:] vec, float[:] ref_vec, float[:] box_vec):
    cdef int i
    cdef float[:] o_vec

    o_vec = np.zeros(3, dtype=np.float32)
    
    for i in range(3):
        o_vec[i] = vec[i]
        
        if vec[i] - ref_vec[i] > 0.5*box_vec[i]:
            o_vec[i] -= box_vec[i]
        elif vec[i] - ref_vec[i] < -0.5*box_vec[i]:
            o_vec[i] += box_vec[i]

    return o_vec
            


cpdef float[:,:] pbc_mol(float[:,:] mat, float[:] ref_vec, float[:] box_vec):
    cdef int i, j, n_row

    cdef float[:,:] o_mat

    o_mat = np.zeros((3,3), dtype=np.float32)
    
    n_row = mat.shape[0] 
    
    for i in range(3):
        if mat[0,i] - ref_vec[i] > 0.5*box_vec[i]:
            for j in range(n_row):
                o_mat[j,i] = mat[j,i] - box_vec[i]
        elif mat[0,i] - ref_vec[i] < -0.5*box_vec[i]:
            for j in range(n_row):
                o_mat[j,i] = mat[j,i] + box_vec[i]
        else:
            for j in range(n_row):
                o_mat[j,i] = mat[j,i]

    return o_mat
            

cpdef double norm(double[:] vec):
    return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

cpdef double distance(float[:] vec1, float[:] vec2):
    return sqrt((vec2[0]-vec1[0])**2 + (vec2[1]-vec1[1])**2 + (vec2[2]-vec1[2])**2)

cpdef double square_distance(float[:] vec1, float[:] vec2):
    return (vec2[0]-vec1[0])**2 + (vec2[1]-vec1[1])**2 + (vec2[2]-vec1[2])**2

cpdef float between_angle(float[:] v1, float[:] v2):
    cdef int i
    cdef float norm1, norm2

    norm1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    norm2 = sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

    for i in range(3):
        v1[i] /= norm1
        v2[i] /= norm2

    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]


cpdef float angle(float[:] vec, float[:] vec1, float[:] vec2):
    cdef int i
    cdef float norm1, norm2
    cdef float[:] v1, v2

    v1 = np.zeros(3, dtype=np.float32)
    v2 = np.zeros(3, dtype=np.float32)

    for i in range(3):
        v1[i] = vec1[i] - vec[i]
        v2[i] = vec2[i] - vec[i]

    norm1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    norm2 = sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    #if norm1 == 0 or norm2 == 0:
    #    print(np.array(vec))
    #    print(np.array(vec1))
    #    print(np.array(vec2))
    #    exit(1)

    for i in range(3):
        v1[i] /= norm1
        v2[i] /= norm2

    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

cpdef void square_distance_matrix(float[:,:] position_matrix,
                             float[:] box_vec,
                             float[:,:] square_dist_mat,
                             float r_max,
                             int number_row):
    cdef int i, j, k
    cdef int pbc_flag
    cdef float square_dist
    cdef float[:] vec1, vec2, vec3
    
    vec3 = np.zeros(3, dtype=np.float32)

    for i in range(number_row):
        vec1 = position_matrix[i]

        pbc_flag = 0
        
        if r_max == 0:
            pbc_flag = 1
            r_max = 1000000
            
        if r_max != 0:
            for j in range(3):
                if vec1[j] < r_max or vec1[j] > box_vec[j]-r_max:
                    pbc_flag = 1
                    break
                
        for j in range(i+1, number_row):
            vec2 = position_matrix[j]

            for k in range(3):
                vec3[k] = vec2[k]
            
            if pbc_flag == 1:   
                    
                for k in range(3):                    
                    if vec2[k] - vec1[k] > 0.5*box_vec[k]:
                        vec3[k] = vec2[k] - box_vec[k]
                    elif vec2[k] - vec1[k] < -0.5*box_vec[k]:
                        vec3[k] = vec2[k] + box_vec[k]
                        
            if abs(vec3[0]-vec1[0])>r_max or abs(vec3[1]-vec1[1])>r_max or abs(vec3[2]-vec1[2])>r_max:
                square_dist = 100.
            else:
                square_dist = square_distance(vec1, vec3)           
           
            square_dist_mat[i,j] = square_dist
            square_dist_mat[j,i] = square_dist_mat[i,j]


cpdef double[:,:] distance_matrix(float[:,:] position_matrix, float[:] box_vec, double rcut):
    cdef int i, j, k, nrow, ncol
    cdef float[:] vec1, vec2, vec3
    cdef double[:,:] dist_mat

    nrow = position_matrix.shape[0]

    dist_mat = np.zeros((nrow, nrow), dtype=np.double) 

    for i in range(nrow):
        vec1 = position_matrix[i]
        
        for j in range(i+1, nrow):
            vec2 = position_matrix[j]
            vec3 = np.zeros(3, dtype=np.float32)
            vec3 = pbc(vec2, vec1, box_vec)
            
            #if abs(vec3[0]-vec1[0])>rcut or abs(vec3[1]-vec1[1])>rcut or abs(vec3[2]-vec1[2])>rcut:
            #    dist_mat[i,j] = rcut
            #else:
            dist_mat[i,j] = distance(vec1, vec3)
            
            dist_mat[j,i] = dist_mat[i,j]

    return dist_mat


cpdef int determine_coordination(float[:] x_i, float[:] x_j, float[:] box):
    cdef int i
    cdef float r_cut, ang_cut
    cdef float[:] x_k
    
    r_cut = 3.5*3.5
    
    x_k = pbc(x_j, x_i, box)
    
    if square_distance(x_i, x_k) > r_cut:
        return 0
    
    return 1



cpdef double[:] cartesian_to_spherical(double x, double y, double z):
    cdef double xsq_plus_ysq
    cdef double r, theta, pi

    xsq_plus_ysq = x**2 + y**2

    r = sqrt(xsq_plus_ysq + z**2)
    theta = atan2(z, np.sqrt(xsq_plus_ysq))
    pi = atan2(y,x)

    return np.array([r, theta, pi])


cpdef double f_cut(double r, double r_cut):
    return 0.5*(cos(pi*r/r_cut)+1)

cpdef double g2(double eta, double r):
    return exp(-eta*r*r)

cpdef double g4(double eta, double lamb, double zeta, double r1, double r2, double r3, double cos_angle):
    return 2**(1-zeta)*((1+lamb*cos_angle)**zeta)*exp(-eta*(r1*r1+r2*r2+r3*r3))

cpdef double m_g4(double eta, double zeta, double r1, double r2, double cos_angle):
    return 2**(1-zeta)*((1+cos_angle)**zeta)*exp(-eta*(((r1+r2)*0.5)**2))


cpdef double lennard_jones(float[:] vec1, float[:] vec2, double sig1, double sig2, double eps1, double eps2):
    cdef double r, p, sig, eps
    
    if sig1 == 0 or sig2 == 0:
        return 0
    else:
        r = distance(vec1, vec2)
        sig = 0.5*(sig1 + sig2)
        eps = sqrt(eps1*eps2)
        
        p = (sig/r)**6
        
        #return 4*eps*p*(p-1)
        return 4*eps*-p
    
cpdef double coulomb(float[:] vec1, float[:] vec2, double q1, double q2):
    cdef double r
    
    if q1 == 0 or q2 == 0:
        return 0
    else:
        r = distance(vec1, vec2)
        
        return 138.93546*q1*q2/r
    
cpdef find_hbjump(top, traj, hbm, hbn, hbjump):
    cdef int i, j, k, l
    #cdef long[:] hb_vec, hbm_vec, hbn_vec, hbm_sub_vec, hbn_sub_vec
    
    print("Loading File.")
    hbm_mat = np.load(hbm).astype(np.int)
    hbn_mat = np.load(hbn)
    
    print("Finding Jump states")
    u = md.Universe(top, traj)
    num_frame = len(u.trajectory)
    dt = round(u.trajectory.dt, 3)
    t_vec = np.array([i*dt for i in range(num_frame)])
    t_min = 0.1

    hbm_mat = hbm_mat.T
    # fill hbm_mat of transient hb breking -> 1
    trans_frame = int(t_min/dt)
    for i in tqdm(range(hbm_mat.shape[0]), desc="Fill transient breaking of h-bond"):
        hbm_vec = hbm_mat[i]
        trans_vec = []
        
        start_flag = 0
        for j in range(len(hbm_vec)):
            hbm = hbm_vec[j]
            
            if hbm == 0:
                if start_flag == 0:
                    start_frame = j
                elif start_flag == 1:
                    continue
                start_flag = 1
            if hbm == 1:
                if start_flag == 0:
                    continue
                elif start_flag == 1:
                    end_frame = j
                    trans_vec.append([start_frame, end_frame])
                start_flag = 0
        
        for trans in trans_vec:
            if trans[1] - trans[0] < trans_frame:
                hbm_mat[i,trans[0]:trans[1]] = 1
    
    # find time interval of hb 
    hb_time_interval_mat = []
    for i in tqdm(range(hbm_mat.shape[0]), desc="Find hb existence."):
        hbm_vec = hbm_mat[i]
        hb_time_interval_vec = []
        
        start_flag = 0
        for j in range(len(hbm_vec)):
            hbm = hbm_vec[j]
                                 
            if (j==len(hbm_vec)-1) and (hbm==1):
                end_frame = j
                hb_time_interval_vec.append([start_frame, end_frame])
            
            if hbm == 0:
                if start_flag == 0:
                    continue
                elif start_flag == 1:
                    end_frame = j
                    start_flag = 0
                    hb_time_interval_vec.append([start_frame, end_frame])
            
            if hbm == 1:
                if start_flag == 0:
                    start_frame = j
                    start_flag = 1
                elif start_flag == 1:
                    continue
                    
        hb_time_interval_mat.append(hb_time_interval_vec)
   
    #print(hb_time_interval_mat[:10]); exit(1)
            
    jump_mat = []
    
    for i, hbn_vec in tqdm(enumerate(hbn_mat), total=len(hbn_mat)):
        idx_donor, idx_h, idx_acceptor = hbn_vec
        
        hbn_family_mat = hbn_mat[(hbn_mat[:,0]==idx_donor) & (hbn_mat[:,1]==idx_h)]
        idx_hbn_family_mat = np.where((hbn_mat[:,0]==idx_donor) & (hbn_mat[:,1]==idx_h))

        hb_time_interval_vec = hb_time_interval_mat[i]
        for j, hb_time_interval in enumerate(hb_time_interval_vec):
            start_time, end_time = hb_time_interval
            
            for k in range(len(idx_hbn_family_mat[0])):
                idx_hbn_family = idx_hbn_family_mat[0][k]
                hb_time_interval_family_vec = hb_time_interval_mat[idx_hbn_family]
                
                for l, hb_time_interval_family in enumerate(hb_time_interval_family_vec):
                    family_start_time, family_end_time = hb_time_interval_family
                    
                    if (start_time < family_start_time) and (end_time > family_end_time):
                        jump_mat.append([hbn_mat[i], end_time, hbn_mat[idx_hbn_family]])
        
                           
    np.save(hbjump, jump_mat)

