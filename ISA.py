#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import special as sp
import numpy as np


# ![image.png](attachment:image.png)

# In[405]:


def kin(qB,mu,delP_prime,h):
    return 70.6*qB*mu/h/delP_prime



def pseudo_time(k,t,phi,ct,mu,rw):
    return 2.637e-4*k*t/(phi*ct*mu*rw**2)

def ri(rw,td_pseudo):
    return 2*rw*td_pseudo**0.5


def ri_(k_hat,tn,phi,mu,ct):
    return 2*(0.006328/24*k_hat*tn/phi/mu/ct)**0.5

def whittaker(z):
    #correction for formula in page 36 of Feitosa thesis is given. term -2sqrt(pi)/z should be +2sqrt(pi)/z
    if z<=2: #approximation in eq.C-1 from Oliver's paper is applicable
        summation=0
        for i in range(9):
            summation+=math.gamma(i+1/2)/math.factorial(i)/math.factorial(i+1)*z**i*\
            (sp.digamma(i+1)+sp.digamma(i+2)-sp.digamma(i+1/2)-math.log(z))  
        summation+=2*math.pi**0.5/z
        
        return z*math.exp(-z/2)/2/math.pi*summation
    else:
        return z**0.5*math.exp(-z/2)*(1+1/4/z-3/32/z**2+15/128/z**3-525/2048/z**4)


def kernel(rD, tD):
    #kernel function is validated with the graph in page 8 in Oliver's paper
    return math.pi**0.5*rD/tD*math.exp(-rD**2/2/tD)*whittaker(rD**2/tD)

def omega(zD):
    """
    omega function is a function that is equal to integral (kernel function*drD), see Feitosa thesis page 98.
    zD=rD/sqrt(tD)
    """
    return math.pi**0.5*zD*math.exp(-zD**2/2)*whittaker(zD**2)

def trapezoidal_rule(f, a, b, n=1000):
    """
    Compute the approximate definite integral of f(x) from a to b using
    the Trapezoidal Rule with n subintervals.
    """
    h = (b - a) / n
    x = [a + i * h for i in range(n+1)]
    fx = [f(x_i) for x_i in x]
    integral = (h / 2) * (fx[0] + 2*sum(fx[1:-1]) + fx[-1])
    return integral

def ISA( phi, ct, rw, qB, mu, h, n, time, delP_prime, ki_container=[0], radius_inv=[0], verbose=True):
    """ ISA is the inverse solution algorithm provided in Feitosa thesis page 98
    
    inputs: phi         : porosity
            ct          : total compressibility (1/psi)
            rw          : well radius (ft)
            qB          : sandface rate (rb/D)
            mu          : viscosity (cP)
            h           : pay zone height (ft)
            time        : list of time values where pressure data recorded.
                          Starts with [0,...] for alg. to work correct.
            delP_prime  : list of pressure derivative with respect to lnt. 
                          Again starts with [0,...].
            ki_container: type list, permeability container
            radius_inv  : type list, radius container
    
    Returns permeability at last time step, list of permeability, list of corresponding radius.
    """
    if n<2: #base case 
        #solve for base case kn which is k_hat (instantaneous)
        kn=kin(qB,mu,delP_prime[n],h)
        #compute pseudo time 
        tD_hat=pseudo_time(kn,time[n],phi,ct,mu,rw)
        #compute radius of investigation 
        r_inv=ri(rw,tD_hat)
        #gather data
        radius_inv.append(r_inv)
        ki_container.append(kn)
        return kn, ki_container, radius_inv
    else:
        #compute k_instantaneous for the current n
        k_hat=kin(qB,mu,delP_prime[n],h)
        tD_hat=pseudo_time(k_hat,time[n],phi,ct,mu,rw) #pseudo time
        #recurse to find previous permeability (n-1)
        k_prev, ki_container, radius_inv=ISA( phi, ct, rw, qB, mu, h, n-1, time, delP_prime, ki_container, radius_inv, verbose)
        summation=0
        for i in range(2,n):
            #find integral ranges Zi-1,D and Z0D, take the tD_hat at n for rjD/sqrt(tD_hat)
            k_hat_i_prev=kin(qB,mu,delP_prime[i-1],h)
            tD_hat_i_prev=pseudo_time(k_hat_i_prev,time[i-1],phi,ct,mu,rw) #pseudo time
            r_inv=ri(rw,tD_hat_i_prev)
            rjD=r_inv/rw
            #compute Upper and lower bounds of integral. Use tD_hat. !!!Do NOT use tD_hat_prev!!!
            Z_upper=rjD/tD_hat**0.5
            Z_lower=min(1/tD_hat**0.5,0.12)
            #compute integral using trapezoid rul
            omega_integral=trapezoidal_rule(omega, Z_lower, Z_upper)
            #compute delta(1/ki) equals to =1/ki-1/k(i-1)
            del_reciprocal_ki=1/ki_container[i]-1/ki_container[i-1]
            #compute sum, given in the numerator of ISA algorithm formula
            summation+=del_reciprocal_ki*omega_integral
        #compute integral bounds Zn-1,D and Z0D (this is for n-1)
        k_hat_prev=kin(qB,mu,delP_prime[n-1],h)
        tD_hat_prev=pseudo_time(k_hat_prev,time[n-1],phi,ct,mu,rw) #pseudo time
        r_inv_prev=ri(rw,tD_hat_prev)
        rjD=r_inv_prev/rw
        #integral bounds, again use tD_hat. !!!do NOT use tD_hat_prev!!!
        Z_upper=rjD/tD_hat**0.5
        Z_lower=min(1/tD_hat**0.5,0.12)
        #compute integral in denominator of ISA algorithm.
        omega_integral=trapezoidal_rule(omega, Z_lower, Z_upper)
        #compute reciprocal_kn (1/kn).
        reciprocal_kn=(1/k_hat-1/k_prev+summation)/(1-omega_integral)+1/k_prev
        #convert reciprocal kn into kn 
        kn=1/reciprocal_kn
        #compute radius of investigation for (tn,rn).
        r_inv=ri(rw,tD_hat)
        #gather the data
        ki_container.append(kn)
        radius_inv.append(r_inv)
    if verbose:
        print('solved for n:',n,'kn:',kn, 'radius:',r_inv)
    return kn, ki_container, radius_inv



def trapezoidal_rule_adapted_for_kernel(f, a, b, tD, n=1000):
    """
    Compute the approximate definite integral of f(x) from a to b using
    the Trapezoidal Rule with n subintervals.
    """
    h = (b - a) / n
    x = [a + i * h for i in range(n+1)]
    fx = [f(x_i, tD) for x_i in x]
    integral = (h / 2) * (fx[0] + 2*sum(fx[1:-1]) + fx[-1])
    return integral

def ISA_adapted_for_kernel( qB, mu, h, n, time, delP_prime, ki_container=[0], radius_inv=[0], verbose=True):
    """ ISA is the inverse solution algorithm provided in Feitosa thesis page 98"""
    if n<2: #base case 
        #solve for base case kn which is k_hat (instantaneous)
        k_hat=kin(qB,mu,delP_prime[n],h)
        #compute pseudo time
        tD_hat=pseudo_time(k_hat,time[n],phi,ct,mu,rw) #pseudo time
        r_inv=ri(rw,tD_hat)
        #gather data
        ki_container.append(k_hat)
        radius_inv.append(r_inv)
        return k_hat, ki_container, radius_inv
    else:
        #compute k_instantaneous for the current n
        k_hat=kin(qB,mu,delP_prime[n],h)
        tD_hat=pseudo_time(k_hat,time[n],phi,ct,mu,rw) #pseudo time
        #recurse to find previous permeability (n-1)
        k_prev, ki_container, radius_inv=ISA_adapted_for_kernel(qB,mu,h,n-1,time,delP_prime,ki_container,radius_inv)
        summation=0
        for i in range(2,n):
            #find integral ranges ri-1,D and r0D, use tD_hat for n
            k_hat_i_prev=kin(qB,mu,delP_prime[i-1],h)
            tD_hat_i_prev=pseudo_time(k_hat_i_prev,time[i-1],phi,ct,mu,rw) #pseudo time
            r_inv=ri(rw,tD_hat_i_prev)
            rjD=r_inv/rw
            #note Kernel integral is computed using tD_hat for n. !!!Do NOT use tD_hat_prev!!!
            kernel_integral=trapezoidal_rule_adapted_for_kernel(kernel, 1, rjD, tD_hat)
            reciprocal_ki=1/ki_container[i]-1/ki_container[i-1]
            summation+=reciprocal_ki*kernel_integral
        #find integral ranges Zn-1,D and Z0D (this is for n-1)
        k_hat_prev=kin(qB,mu,delP_prime[n-1],h)
        tD_hat_prev=pseudo_time(k_hat_prev,time[n-1],phi,ct,mu,rw) #pseudo time
        r_inv_prev=ri(rw,tD_hat_prev)
        rjD=r_inv_prev/rw
        #note Kernel integral is computed using tD_hat for n. !!!Do NOT use tD_hat_prev!!!
        kernel_integral=trapezoidal_rule_adapted_for_kernel(kernel, 1, rjD, tD_hat)
        reciprocal_kn=(1/k_hat-1/k_prev+summation)/(1-kernel_integral)+1/k_prev
        #compute kn from reciprocal_kn
        kn=1/reciprocal_kn
        #compute radius of investigation for kn
        r_inv=ri(rw,tD_hat)
        #gather data
        ki_container.append(kn)
        radius_inv.append(r_inv)
    if verbose:
        print('solved for n:',n,'kn:',kn, 'r_inv:',r_inv)
    return kn, ki_container, radius_inv