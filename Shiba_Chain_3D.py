# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:53:30 2021

@author: crisl
"""

#everything in atomic units
import numpy as np
#from joblib import Parallel, delayed
import multiprocessing as mp

def Shiba_Chain3(N_atoms, layers, state, N_period, lamda, borde, ancho, k_f, U, j, DOS,\
                 s, delta, N_omega, range_omega, Dynes, lattice_param, mass_eff):
    
    #array size
    pi=np.pi
    N_x = N_atoms + 2*borde
    N_y = ancho
    
    
    "Magnetic impurities parameters and spin state"
    S = s

    if (state == 'FM'):
        thetaS = np.zeros(N_atoms, dtype = 'float')
    elif (state == 'AF'):
        thetaS = np.zeros(N_atoms, dtype = 'float')
        for i in range(int(len(thetaS)/2)):
            thetaS[2*i+1] = pi
        if N_atoms % 2 == 0:
            thetaS[-1] = pi        
        
    N_periodicty = N_period   
    if (state == 'helix'):        
        theta = np.linspace(0.0, 2*np.pi, N_periodicty+1, dtype = 'float')
        theta = theta[0:N_periodicty]####remove 2pi 
        thetaS = np.zeros(N_atoms, dtype = 'float')
        t = []
        
        if(N_periodicty < N_atoms):
    
            while (len(t)<N_atoms):
                t = np.concatenate((t,theta))
        
            thetaS[0:N_atoms] = t[0:N_atoms]
        
        else:
            
            thetaS[0:N_atoms] = theta[0:N_atoms]
            
    phi = np.zeros(N_atoms)
    
    
    "Material data Bi2Pd"
    Damping = Dynes #Dynes damping
    Delta = delta
    DOS_o = DOS #Normal phase DOS
    Fermi_k = k_f
    #mass_eff=1 #SC Band effective mass
    a_interatomic=lattice_param#BiPd
    J = j ##coupling with magnetic atom

    "spin-orbit coupling"
    #lamda = (alpha/(2*a_interatomic*0.529))/27.2116

    "we define the omega vector"
    "from -N_delta/2 to N_delta/2"
    N_delta = range_omega
    
    Romega = np.zeros([N_omega])
    Romega=np.array(Romega, np.longdouble)
    step_omega=N_delta*Delta/(N_omega-1)

    for i_omega in range(N_omega):
        Romega[i_omega] = (-N_delta/2.*Delta+(i_omega)*step_omega)
         
    Romega = np.array([-1.10247101e-06, -5.51235503e-07, -3.38813179e-21,  5.51235503e-07,1.10247101e-06])
    Romega = np.array(Romega)
    vv=Romega*27211.6
    
    "We calculate the Green's functions and solve Dyson eq"    
     
    import Self_Energy_3D as SL
    Self2 = SL.Self_Energy(J, S, thetaS, phi, U, N_atoms, N_x, N_y, layers, borde, lamda)
    
    
    "Parallel calculation"
    num_cores = mp.cpu_count()
    #pool = mp.Pool(num_cores)
    
    #Free Green's function
    #import Free_Green_new as FG
    #G0 = [pool.apply(FG.Free_Green, args=((N_x, N_y, omega,
                                            #Damping, Fermi_k, mass_eff, DOS_o, Delta, a_interatomic))) for omega in Romega]

    #Solve Dyson eq  
    #import Dyson as Dy
    #GG = [pool.apply(Dy.Dyson_eq, args=((Go , Self2 , N_x, N_y))) for Go in G0]

    #pool.close() 
    
    GG = np.zeros([4 * N_y * N_x*layers , 4 * N_y * N_x*layers, N_omega], dtype=complex)
    Go = np.zeros([4 * N_y * N_x*layers , 4 * N_y * N_x*layers, N_omega], dtype=complex)
    
    N_omega = 5
    for i_omega in range(N_omega):
        
        omega = Romega[i_omega]
    
    
        #BCS Green's function
        import Free_Green_new as FG
        G0 = FG.Free_Green(N_x, N_y, layers, omega, Damping, Fermi_k, mass_eff, DOS_o, Delta, a_interatomic)
        Go[:,:,i_omega] = G0
        
        #import Free_Gree_loop as FL
        #(Go2, go) = FL.Free_Green(N_x, N_y, omega, Damping, Fermi_k, mass_eff, DOS_o, Delta, a_interatomic)
        
        #Solve Dyson's equation
        import Dyson as Dy
        gg = Dy.Dyson_eq(G0 , Self2 , N_x, N_y, layers)
        
        GG[:,:, i_omega] = gg
        
        
        
    return(GG , N_x, N_y, N_omega , vv, Go, Self2, num_cores, Romega, thetaS, step_omega)
    
    
    
    
    
    
    
    
    
    
    