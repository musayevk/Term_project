import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import special as sp
import numpy as np
import ISA

def r_inv(kin,t,phi,ct,mu):
    """computes the radius of investigation for Yeh Agarwal procedure. Original formula includes rw 
    (see Feitosa paper) which should be removed from the r_inv formula. Correct one is without rw. 
    t: hours, rw: ft, k: mD, ct: 1/psi, mu: cP"""
    return 0.02436*(kin*t/phi/ct/mu)**0.5


def numerical_diff(ki,ki_prev,ki_next,ri,ri_prev,ri_next):
    ''' given the points i, i-1, i+1, returns the numerical differentiation for point i.'''
    return ((ki-ki_prev)/(ki-ki_prev)*(ki_next-ki)+(ki_next-ki)/(ri_next-ri)*(ri-ri_prev))/(ri_next-ri_prev)

def yeh_agarwal(time, delP_prime,qB,phi, ct,mu,h):
    """ Yeh-Agarwal procedure for computing permeability distribution from given pressure data."""
    #containers for data gathering
    ri_con,kin_con,kr_con=[],[],[]

    for i in range(len(time)):
        #compute the k instantaneous from Feitosa et al.(1994)
        kin=ISA.kin(qB,mu,delP_prime[i],h)
        #compute the radius of investigation from Yeh-Agarwal formula
        ri_=r_inv(kin,time[i],phi,ct,mu)
        #gather data
        ri_con.append(ri_)
        kin_con.append(kin)
        
    for i in range(len(time)):
        try:
            #differentiate numerically
            dkin=numerical_diff(kin_con[i],kin_con[i-1],kin_con[i+1],ri_con[i],ri_con[i-1],ri_con[i+1])
        except:
            #set dkin to zero if error happens
            dkin=0
        #compute the permeability value using Yeh Agarwal formula (eq.12 in Feitosa et al.(1994))
        kr=ri_con[i]/2*dkin+kin_con[i]
        #gather data
        kr_con.append(kr)
    
    return kr_con, ri_con


def modified_yeh_agarwal(time, delP_prime,qB,phi,ct,mu,h):
    """Modified Yeh-Agarwal procedure for computing permeability distribution from given pressure data.
    
    inputs:time (hr), delP_prime (pressure derivative with respect to lnt),qB (sandface rate, rb/Day), phi (porosity),
    ct (total compressibility), mu (viscosity),h (payzone thickness, ft)
    
    outputs: type list containers, kr_con (permeability), ri_con (radius of investigation)
    """
    #containers for data gathering
    ri_con,kin_con,kr_con=[],[],[]

    for i in range(len(time)):
        #compute the k instantaneous from Feitosa et al.(1994)
        kin=ISA.kin(qB,mu,delP_prime[i],h)
        #compute the radius of investigation from Yeh-Agarwal formula
        ri_=r_inv(kin,time[i],phi,ct,mu)
        #gather data
        ri_con.append(ri_)
        kin_con.append(kin)
        
    for i in range(len(time)):
        try:
            #differentiate numerically. use 1/kin for differentiation as given in eq. 15 from Feitosa et al. (1994)
            dkin=numerical_diff(1/kin_con[i],1/kin_con[i-1],1/kin_con[i+1],ri_con[i],ri_con[i-1],ri_con[i+1])
        except:
            dkin=0
        #compute the reciprocal permeability value using Modified Yeh Agarwal formula (eq.15 in Feitosa et al.(1994))
        reciprocal_kr=ri_con[i]/2*dkin+1/kin_con[i]
        #gather data
        kr_con.append(1/reciprocal_kr)
    
    return kr_con, ri_con