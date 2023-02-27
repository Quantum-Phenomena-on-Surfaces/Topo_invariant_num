# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:12:22 2021

@author: crisl
"""

import numpy as np
from numpy.linalg import inv

#we solve Dyson's equation
def Dyson_eq(Go , Self , N_x, N_y, layers):
    
    
    Id = np.identity(4 * N_y * N_x * layers)
   
    matrx_inv = inv(Id - np.dot(Go, Self))##### + o -?????
    gg = np.dot(matrx_inv , Go)      
    
    
    return(gg)
