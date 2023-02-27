# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:30:42 2021

@author: crisl
"""

import numpy as np

sin = np.sin
cos = np.cos
exp =np.exp
ui = complex(0.0, 1.0)

#Local self-energy for exchange and SOC

def Self_Energy(J, S, thetaS, phi, U, N_atoms, N_x, N_y, layers, borde, lamda):
    
     Self = np.zeros([N_y * N_x * layers, N_y * N_x * layers, 4, 4], dtype=complex)
     Self2 = np.zeros([N_y * N_x * layers * 4, N_y * N_x * layers * 4], dtype=complex)
     
     "diagonal in the atom space"
     
     for i_atom in range(N_atoms):
         
         g_i = int(N_y/2.0) * N_x + (i_atom + borde)
         #g_i = int(N_y/2.0) * N_x + (2*i_atom + borde)##### d=2a
         theta_i = thetaS[i_atom]
         phi_i = phi[i_atom]
         
         
         Self [g_i, g_i, 0, 0]= J*S*cos(theta_i)-U
         Self [g_i, g_i, 1, 1]= - J*S*cos(theta_i)-U
         Self [g_i, g_i, 2, 2]= - J*S*cos(theta_i)+U
         Self [g_i, g_i, 3, 3]= J*S*cos(theta_i)+U
         
         Self [g_i, g_i, 0, 1]= J*S*sin(theta_i)*exp(-ui*phi_i)
         Self [g_i, g_i, 1, 0]= J*S*sin(theta_i)*exp(ui*phi_i)
         Self [g_i, g_i, 2, 3]= - J*S*sin(theta_i)*exp(ui*phi_i)
         Self [g_i, g_i, 3, 2]= - J*S*sin(theta_i)*exp(-ui*phi_i)
         
        
     "Non - diagonal in the atom space"
     "Spin orbit interaction"

     #the coupling along x direction (sigma_y)
     
     
     for i_matrix in range(N_y):
         for j_matrix in range(N_x - 1):
            
             g_i = (i_matrix)*N_x + j_matrix
             g_j = (i_matrix)*N_x + (j_matrix + 1)
            
             Self[g_i, g_j, 0, 1]= lamda
             Self[g_i, g_j, 1, 0]= - lamda
             Self[g_i, g_j, 2, 3]= - lamda
             Self[g_i, g_j, 3, 2]= lamda
            
            
             Self[g_j, g_i, 0, 1]= - lamda
             Self[g_j, g_i, 1, 0]= lamda
             Self[g_j, g_i, 2, 3]= lamda
             Self[g_j, g_i, 3, 2]= -lamda
              ###cambiar signo o no????

    
    #the coupling along y direction (sigma_x)
    
     for i_matrix in range(N_y - 1):
         for j_matrix in range(N_x):
            
             g_i = (i_matrix)*N_x + j_matrix
             g_j = (i_matrix + 1)*N_x + j_matrix
            
             Self [g_i, g_j, 0, 1]= - ui * lamda
             Self [g_i, g_j, 1, 0]= - ui * lamda
             Self [g_i, g_j, 2, 3]= - ui * lamda
             Self [g_i, g_j, 3, 2]= - ui * lamda

             Self [g_j, g_i, 0, 1]= ui * lamda
             Self [g_j, g_i, 1, 0]= ui * lamda
             Self [g_j, g_i, 2, 3]= ui * lamda
             Self [g_j, g_i, 3, 2]= ui * lamda
            
     for i in range(N_x*N_y*layers):
        for j in range(N_x*N_y*layers):
            for t_i in range(4):
                for t_j in range(4):
                    Self2[(i) * 4 + t_i, (j) * 4 + t_j] = Self[i, j, t_i, t_j]
                    
                    
     return(Self2)