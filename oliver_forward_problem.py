import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import special as sp
import numpy as np


#kernel function
def kernel(rD, tD):
    #kernel function is validated with the graph in page 8 in Oliver's paper
    return 0.5*math.pi**0.5*rD/tD*math.exp(-rD**2/2/tD)*whittaker(rD**2/tD)

def whittaker(z):
    if z<=2: #approximation in eq.C-1 from Oliver's paper is applicable
        summation=0
        for i in range(9):
            summation+=math.gamma(i+1/2)/math.factorial(i)/math.factorial(i+1)*z**i*\
            (sp.digamma(i+1)+sp.digamma(i+2)-sp.digamma(i+1/2)-math.log(z))  
        summation+=2*math.pi**0.5/z
        
        return z*math.exp(-z/2)/2/math.pi*summation
    else:
        return z**0.5*math.exp(-z/2)*(1+1/4/z-3/32/z**2+15/128/z**3-525/2048/z**4)

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


def _td(kref,t,phi,ct,mu,rw):
    """ takes: kref (md), time (hours), phi (porosity), ct (compressibility, 1/psi), mu (viscosity, cp)
        rw (well radius, ft)
        
        returns: the dimensionless time
    """
    return 2.637e-4*kref*t/(phi*ct*mu*rw**2)

def _rd(r, rw):
    return r/rw

def _kd(k,kref):
    return k/kref

def convert_delP_prime(Pd_prime, kref, h, qB, mu):
    return Pd_prime*141.2*qB*mu/kref/h

def oliver(r_cont, k_cont, time, phi,ct,mu,rw, qB, h):
    """ inputs: r_cont: list of radiuses
                k_cont: list of permeability values corresponding to radius values
                time: list of the time in hours
                phi: porosity
                ct: compressibility (1/psi)
                mu: viscosity (cp)
                rw: well radius (ft)
                qB: flow rate (rb/D)
                h: payzone (ft)
                
        outputs: time and t*d(delP)/dt (Bourdet derivative) values
    """
    #set the first semilog plot permeability as kref 
    kref=k_cont[0]
    #create containers for time and pressure derivative 
    delP_prime_cont, t_cont=[], []
    #iterate over each time to obtain the pressure derivative response
    for t in time:
	#compute dimensionless time
        td=_td(kref,t,phi,ct,mu,rw)
        #check if td is larger than 100
        if td>100:
            Pd_prime=0
            #integral bounds for initial rD1
            a=1
            b=_rd(r_cont[0], rw)
            #start iteration from 1 so that the integral bounds are correct as (|1|--k1--|rd1|--k2--|rd2|--k3--|rd3|)
            for i in range(1, len(r_cont)):
                #compute the Pd_prime using the Oliver's formula (correct version is given in Feitosa thesis page36)
                Pd_prime+=1/_kd(k_cont[i-1],kref)*trapezoidal_rule_adapted_for_kernel(kernel, a, b, td)
                #change the integral bounds (|1|--k1--|rd1|--k2--|rd2|--k3--|rd3|...rn-1D--|kn-1|--3*sqrt(td))
                a=b
                b=_rd(r_cont[i],rw)
            #last integral section, bounded by rn-1 to 3*sqrt(tD)
            Pd_prime+=1/_kd(k_cont[-1],kref)*trapezoidal_rule_adapted_for_kernel(kernel, a, 3*td**0.5, td)
            #gather the computed data    
            t_cont.append(t)
            #before gathering convert the td*dPd/dtd to t*d(delP)/dt (delP=Pi-P(rw,t))
            delP_prime_cont.append(convert_delP_prime(Pd_prime, kref, h, qB, mu))
            
    return t_cont, delP_prime_cont