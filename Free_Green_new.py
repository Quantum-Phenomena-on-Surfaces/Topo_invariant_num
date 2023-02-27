#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:13:10 2018

@author: cristina
"""

import numpy as np
import cmath as cm
import scipy.spatial.distance

# Functions
pi = np.pi
sin = np.sin
cos = np.cos
sqrt = np.sqrt
exp = np.exp

def Free_Green(N_x, N_y, layers, lomega, Damping, Fermi_k, mass_eff, DOS_o, Delta, a_interatomic):

    G = np.zeros([N_x * N_y * layers * 4, N_x * N_y * layers * 4], dtype=complex)
    
    omega = lomega + 1j * Damping

    # Non diagonal in atom
    # i = np.arange(N_y)
    # j = np.arange(N_x)
    # k = np.arange(layers)
    # J, K, I = np.meshgrid(j, k, i)
    # ii = np.reshape(I, ((N_x*N_y*layers), ))
    # jj = np.reshape(J, ((N_x*N_y*layers), ))
    # kk = np.reshape(K, ((N_x*N_y*layers), ))
    # ijk = zip(kk,jj,ii)
    # ijk = list(ijk)
    # IJK = np.array(ijk, dtype = 'double')
    
    
    i = np.arange(N_y)
    j = np.arange(N_x)
    k = np.arange(layers)
    I, K, J = np.meshgrid(i,k,j)
    ii = np.reshape(I, ((N_x*N_y*layers), ))
    jj = np.reshape(J, ((N_x*N_y*layers), ))
    kk = np.reshape(K, ((N_x*N_y*layers), ))
    ijk = zip(kk,jj,ii)
    ijk = list(ijk)
    IJK = np.array(ijk, dtype = 'double')
    
    
    rr = scipy.spatial.distance.cdist(IJK, IJK, metric='euclidean')*a_interatomic#distance between sites
    rr[np.where(rr == 0)] = 100
    
   
    t = np.arange(N_x*N_y*layers)
    T, T2 = np.meshgrid(t, t)
    t_i = np.reshape(T, (((N_x*N_y*layers) ** 2), ))
    t_j = np.reshape(T2, (((N_x*N_y*layers) ** 2), ))
    
    
    SS = sqrt(Delta**2 - omega**2)
    xi = Fermi_k / (mass_eff * SS)
    factor = - pi * DOS_o * exp(-rr/ xi) / (SS * Fermi_k * rr)
    #factor = - pi * DOS_o * 1.0 / (SS * Fermi_k * rr)
    
    ######### cambio 2 y 3
    G[t_j * 4 + 0, t_i * 4 + 0] = ( omega * sin(Fermi_k * rr[t_j,t_i]) + SS * cos(Fermi_k * rr[t_j,t_i]) )* factor[t_j,t_i]
    G[t_j * 4 + 1, t_i * 4 + 1] = ( omega * sin(Fermi_k * rr[t_j,t_i]) + SS * cos(Fermi_k * rr[t_j,t_i]) )* factor[t_j,t_i]
    G[t_j * 4 + 2, t_i * 4 + 2] = ( omega * sin(Fermi_k * rr[t_j,t_i]) - SS * cos(Fermi_k * rr[t_j,t_i]) )* factor[t_j,t_i]
    G[t_j * 4 + 3, t_i * 4 + 3] = ( omega * sin(Fermi_k * rr[t_j,t_i]) - SS * cos(Fermi_k * rr[t_j,t_i]) )* factor[t_j,t_i]
    
    #G[t_j * 4 + 0, t_i * 4 + 0] = 0.0
    #G[t_j * 4 + 1, t_i * 4 + 1] = 0.0
    #G[t_j * 4 + 2, t_i * 4 + 2] = 0.0
    #G[t_j * 4 + 3, t_i * 4 + 3] = 0.0

    G[t_j * 4 + 0, t_i * 4 + 3] = - Delta * sin(Fermi_k * rr[t_j,t_i]) * factor[t_j,t_i]
    G[t_j * 4 + 1, t_i * 4 + 2] = Delta * sin(Fermi_k * rr[t_j,t_i]) * factor[t_j,t_i]
    G[t_j * 4 + 2, t_i * 4 + 1] = Delta * sin(Fermi_k * rr[t_j,t_i]) * factor[t_j,t_i]
    G[t_j * 4 + 3, t_i * 4 + 0] = - Delta * sin(Fermi_k * rr[t_j,t_i]) * factor[t_j,t_i]

    # Diagonal in atom
    omega = lomega + 1j * Damping
    SS = sqrt(Delta**2 - omega**2)
    factor_diag = - pi * DOS_o / SS

    G[t * 4 + 0, t * 4 + 0] = omega * factor_diag
    G[t * 4 + 1, t * 4 + 1] = omega * factor_diag
    G[t * 4 + 2, t * 4 + 2] = omega * factor_diag
    G[t * 4 + 3, t * 4 + 3] = omega * factor_diag
#    G[t * 4 + 0, t * 4 + 0] = 0.0
#    G[t * 4 + 1, t * 4 + 1] = 0.0
#    G[t * 4 + 2, t * 4 + 2] = 0.0
#    G[t * 4 + 3, t * 4 + 3] = 0.0
    G[t * 4 + 0, t * 4 + 3] = -Delta * factor_diag 
    G[t * 4 + 1, t * 4 + 2] = Delta * factor_diag 
    G[t * 4 + 2, t * 4 + 1] = Delta * factor_diag 
    G[t * 4 + 3, t * 4 + 0] = -Delta * factor_diag
    
    
    return (G)

