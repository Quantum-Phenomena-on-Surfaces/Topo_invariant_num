#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:22:48 2021

@author: cristina
"""

import numpy as np
#import matplotlib as mpl
#mpl.use('Agg') ####to run in Oberon/other clusters
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = [8, 8]
from numpy.linalg import inv

######import necessary functions
#import G0_omega_calc as Gcalc
import Shiba_Chain_3D as sc
pi=np.pi

def find_nearest(array, value):
    #####calculates the closest but not bigger value in a given array
    diff = array - value
    
    if (value>0):
        i = np.where(diff<=0)
        idx = (abs(diff[i])).argmin()
        return idx
    
    elif(value<0):
        i = np.where(diff>=0)
        idx = (abs(diff[i])).argmin()
        return i[idx][0]        

'''load parameters'''
#energy is given in eV
import json
with open("parameters_numerical.json", "r") as f:
    param = json.load(f)

lattice_param = param["lattice_param"]
k_F = param["k_F"]
K = param["K"]
J = param["J"]
theta = param["theta"]
phi = param["phi"]
DOS = param["DOS"]
s = param["s"]
delta = param["delta"]
dynes = param["Dynes"]
alpha = param["alpha"]
mass_eff = param["mass_eff"]
N_atoms = param["N_atoms"]
borde = param["border"]
ancho = param["wide"]
layers = param["layers"]

#####convert to atomic units
d = 1.0#distance between sites MUST BE 1.0!!!!!!
a_interatomic = d*lattice_param/0.529
K = K/27.2116#%potential scatt
j = J/27.2116#coupling
delta =delta/27.2116 #SC gap
Dynes = dynes/27.2116
lamda = (alpha/(2*a_interatomic*0.529))/27.2116 

state = 'FM'
N_period = 2

##########k vector
Nk = param["Nk"]
k = np.linspace(-np.pi, np.pi, Nk)/(a_interatomic)

#########omega vector
N_omega = param["N_omega"]
range_omega = param["range_omega"]
N_delta = range_omega    

#'''solve Dyson's equation'''
N_x = N_atoms + 2*borde
N_y = ancho

atom_0 = int(N_y/2)*N_x + int(N_x/2)
#

'''G_0_k calculation'''
####remove border atoms for FT
N_remove = 0
T = int(N_remove/2.0)


###solve Dyson Equation
(GG , N_x, N_y, N_omega , vv, Go, Self2, num_cores, Romega, thetaS, step_omega) = sc.Shiba_Chain3(N_atoms, layers, state,\
 N_period, lamda, borde, ancho, k_F, K, j, DOS, s, delta, N_omega, range_omega, Dynes, a_interatomic, mass_eff)



Green_0 = np.zeros([N_y * N_x * layers, N_y * N_x * layers, 4, 4, N_omega], dtype=complex)
for i_omega in range(N_omega):
    for i_atom in range(N_x*N_y*layers):
            for j_atom in range(N_x*N_y*layers):
                for t_i in range(4):
                    for t_j in range(4):
                    
                        Green_0[i_atom, j_atom, t_i, t_j, i_omega] = GG[i_atom*4 + t_i, j_atom*4 + t_j, i_omega]       

G_k_R = np.zeros([N_x - N_remove, 4, 4, Nk, N_omega], dtype=complex)
G_k_omega = np.zeros([4, 4, Nk, N_omega], dtype=complex)

for i_omega in range(N_omega):
    for i_k in range(Nk):
        for i in range(T, N_x - T -1):
            
            #center of the 2D array
            j = int(N_y/2)*N_x + i
            
            G_k_R[i,:,:,i_k, i_omega] = Green_0[atom_0, j, :, :, i_omega]*np.exp(-1j*k[i_k]*(atom_0 - j)*a_interatomic)   
            G_k_omega[:,:, i_k, i_omega] = np.sum(G_k_R[:,:,:, i_k, i_omega], axis = 0)
            
            
G11 = np.zeros([Nk, N_omega], dtype = float)
G22 = np.zeros([Nk, N_omega], dtype = float)
G33 = np.zeros([Nk, N_omega], dtype = float)
G44 = np.zeros([Nk, N_omega], dtype = float)
spect_k_omega = np.zeros([Nk, N_omega], dtype = float)

for i_k in range(Nk):
    for i_omega in range(N_omega):
      spect_k_omega[i_k, i_omega] = -1/pi*np.imag( G_k_omega[0,0, i_k, i_omega] + G_k_omega[1,1, i_k, i_omega] )  
      
      G11[i_k, i_omega] = -1/pi*np.imag( G_k_omega[0,0, i_k, i_omega])
      G22[i_k, i_omega] = -1/pi*np.imag( G_k_omega[1,1, i_k, i_omega])
      G33[i_k, i_omega] = -1/pi*np.imag( G_k_omega[2,2, i_k, i_omega])
      G44[i_k, i_omega] = -1/pi*np.imag( G_k_omega[3,3, i_k, i_omega])
     
        
min_val = np.amin(spect_k_omega)
max_val = np.amax(spect_k_omega)

plt.figure(5)
plt.imshow(spect_k_omega,  vmin=min_val, vmax=max_val, origin='lower', aspect='auto',\
          extent=[vv[0], vv[-1], k[0], k[-1]])
plt.colorbar()
plt.xlabel('omega (meV)')
plt.ylabel('k')
plt.savefig('ImGG.png', dpi = 260, bbox_inches='tight')
            

'''G_k and H_k calculation'''
#import Self_Energy_k as sk

Gk_0 = np.zeros([4, 4], dtype=complex)
H_k = np.zeros([4, 4, Nk], dtype=complex) 
omega_0 = int(N_omega/2.0)

for i_k in range(Nk):
    for i_omega in range(N_omega):
            
        #####G0(k)        
        Gk = G_k_omega[:,:,i_k,i_omega]
       
        Der = (Gk - Gk_0)/step_omega####G_k derivative
        Gk_0 = Gk#####new Gk_0                
                
        Der_inv = inv(Der)        
        ######H(k)
        if (i_omega == omega_0):
            H_k[:,:,i_k] = -Der_inv@Gk_0



'''H(k) diagonalization'''
from numpy import linalg as LA
diag = LA.eig        

E_total = np.zeros([4, Nk], dtype = complex)
psi_total = np.zeros([4, 4, Nk], dtype = complex)


for i_k in range(Nk):

    #Diagonalize                       
    H_m = np.matrix(H_k[:,:,i_k])    
    (E, psi) = diag(H_m)
    
    #E_order = np.sort(E)
    E_total[:, i_k] = E#*27211.6
    psi_total[:,:, i_k] = psi
    
    
#%%
'''Plot all E_n versus k'''
plt.figure(1)
for n in range(4):    
    plt.plot(k, np.real(E_total[n,:]*27211.6), 'b.', ms = 1.0, label = 'H(k)')

plt.savefig('bands.png', dpi = 260, bbox_inches='tight')

##################save data
np.savetxt('G11_k.txt', G11)
np.savetxt('G22_k.txt', G22)
np.savetxt('G33_k.txt', G33)
np.savetxt('G44_k.txt', G44)
            
np.savetxt('bands.txt', np.real(E_total))
np.savetxt('k_vector.txt', k)
np.savetxt('vv.txt', vv)
            


'''New winding number and topo inv'''
i1 = find_nearest(k, -k_F)
i2 = find_nearest(k, k_F)
NNk = i2 - i1 + 1

d_x_old = 0.0###initialized
d_y_old = 0.0
step_k = k[1] - k[0] 
winding_int = np.zeros(Nk, dtype = float)

d_x_k = np.zeros(Nk, dtype = float)
d_y_k = np.zeros(Nk, dtype = float)

A_k = np.zeros([Nk], dtype = complex)
NW = 0.0###initialized

'''for small k_F'''
for i_k in range(Nk):
    
    #i_j = i_k - i1    
    H_ki = H_k[:,:,i_k]
    
    AA = (H_ki[0,0] +  H_ki[0,2]) * (H_ki[1,1] +  H_ki[1,3]) \
    - (H_ki[0,1] +  H_ki[0,3]) * (H_ki[1,0] +  H_ki[1,2])
         
    A_k[i_k] = AA
         
    d_x = np.real(AA)
    d_y = -np.imag(AA)  
    
    ###normalize    
    d_m=np.sqrt(d_x**2+d_y**2)
    d_x = d_x/d_m
    d_y = d_y/d_m
    
    d_x_k[i_k] = d_x
    d_y_k[i_k] = d_y
        
    Der_x = (d_x-d_x_old)/step_k#dx derivative
    Der_y = (d_y-d_y_old)/step_k#dy derivative
    d_x_old = d_x
    d_y_old = d_y
        
    if (i_k == 0):
        NW = 0.5*step_k*(d_x*Der_y-d_y*Der_x)
        #pfa1 = d_x
        #print('pfa -pi/a', pfa1)
        
    elif (i_k < Nk):
        NW = NW+step_k*(d_x*Der_y-d_y*Der_x)
        
    #        if(i_k == int(Nk/2)):
    #            #pfa2 = d_x
    #            #print('pfa 0.0', pfa2)
            
    else:            
        NW = NW + 0.5*step_k*(d_x*Der_y-d_y*Der_x) 
        
        
    winding_int[i_k] = NW
            
            
plt.figure(2) 
plt.plot(k, winding_int/(2*pi), '.', label = r'$w(k)$')
plt.plot(k, d_x_k, '.', label = r'$d_x$')
plt.plot(k, d_y_k, '.', label = r'$d_y$')
plt.grid(True, color='0.95')
plt.show()
plt.title('Winding number')
plt.xlabel('k')
#plt.ylabel('Winding number')
plt.legend()    
plt.savefig('dx_dy.png', dpi = 260, bbox_inches='tight')

#####winding vector

theta = np.linspace(2*np.pi, 0, Nk)
x1 = np.cos(theta)
x2 = np.sin(theta)

norm = plt.Normalize()
color = plt.cm.cool(np.linspace(0,1.0,len(d_x_k)))
plt.rcParams['image.cmap'] = 'cool'           

plt.figure(3)
plt.title('Winding vector')
plt.plot(x1, x2, '--k', linewidth = 0.8)
plt.axhline(y=0.0, color='k', linestyle='--', linewidth = 0.8)
plt.axvline(x=0.0, color='k', linestyle='--', linewidth = 0.8)
plt.scatter(d_x_k[i1:i2], d_y_k[i1:i2], facecolors='none', edgecolors = color)
#plt.plot(d_x_k[0], d_y_k[0], 'go', label = 'First')
#plt.plot(d_x_k[-1], d_y_k[-1], 'ro', label = 'Last')
plt.savefig('winding.png', dpi = 260, bbox_inches='tight')

plt.xlabel(r'Real $z$')
plt.ylabel(r'Imag $z$')


print('Winding number', NW/(2*pi))
            
######save data
np.savetxt('d_x_k.txt', d_x_k)           
np.savetxt('d_y_k.txt', d_y_k)           
np.savetxt('winding.txt', winding_int)           
np.savetxt('k_vector.txt', k)
            
            
            
            
            
            



