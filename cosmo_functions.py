import numpy as np
import scipy as sp
from scipy import integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import math

c=299797. #km/s

def e_z(z,omega_m,omega_k,omega_lambda):
		return (1.0/math.sqrt(omega_m*((1.+z)**3.)+ omega_k*((1.+z)**2.) + omega_lambda))

def mu_model(z,H0,Dm0,Dlambda0):
    e_z_int, e_z_int_err = integrate.quad(lambda x: e_z(x,Dm0,0.,Dlambda0),0.,z) #Universo plano
    return 25. + 5.*sp.log10( (c*(1.+z)/H0) * e_z_int)

def M_SN(H0,M_c):
    return M_c - 25 +5*sp.log10(H0)

def mu_data(m,M_c,H0,alpha,s,beta,Av):
    M=M_SN(H0,M_c)
    return m - M + alpha*(s-1) - beta*(Av)

def error_data(sigma_z,z,sigma_m,alpha,sigma_s,beta,sigma_Av,sigma_int):
    error=sp.array([0.]*len(z))
    new_sigma_z=sp.array([0.]*len(z))
    for i in range(0,len(error)):
        new_sigma_z[i] = sigma_z[i] * (5.0*(1.0+z[i]))/(z[i]*(1.0 + z[i]/2.0)*sp.log(10.0))
        error[i] = sigma_m[i]**2 + (alpha*sigma_s[i])**2 + (beta*sigma_Av[i])**2 + new_sigma_z[i]**2 + sigma_int**2
        error[i] = sp.sqrt(error[i])
    return error

def m_model(z,s,Av,M_c,alpha,beta,Dm0,Dlambda0):
    e_z_int, e_z_int_err = integrate.quad(lambda x: e_z(x,Dm0,0.,Dlambda0),0.,z) #Universo plano
    integral = 5*sp.log10(c * (1+z) * e_z_int)
    return beta*Av - alpha*(s-1) + M_c + integral

def QuitarNan(A): #De la tabla general quita los valores que tienen NaN
    B=[None]*len(A)
    B=sp.array(B)
    for i in range (0,len(A)):
        for j in range (0,len(A)):
            B[j]=A[j][sp.where(sp.isnan(A[i])==False)]
    return B

def log_likelihood(theta, mSN, sigma_m, s, sigma_s, Av, sigma_Av, z, sigma_z):
    alpha, beta, Dm0, Dlambda0, M_c, sigma_int  = theta #Universo plano
    #Calculamos el error total
    new_sigma_z = sp.array([0.]*len(sigma_z)); sigma2 = sp.array([0.]*len(sigma_z)); likelihood=sp.array([0.]*len(sigma_z)); m=sp.array([0.]*len(mSN))
    #Calculamos la integral elemento a elemento
    for i in range(0,len(m)):
        new_sigma_z[i] = sigma_z[i] * ( 5.0 * (1.0+z[i]) ) / ( z[i]*(1.0 + z[i]/2.0) * sp.log(10.0) )
        sigma2[i] = sigma_m[i]**2 + (alpha*sigma_s[i])**2 + (beta*sigma_Av[i])**2 + new_sigma_z[i]**2 + sigma_int**2
        likelihood[i]= ( ((mSN[i]-m_model(z[i],s[i],Av[i],M_c,alpha,beta,Dm0,Dlambda0))**2) / sigma2[i]  +  sp.log(sigma2[i]*2*np.pi) )
    return -0.5 * sp.sum(likelihood)

def log_prior(theta):
    alpha, beta, Dm0, Dlambda0, M_c, sigma_int  = theta #Universo plano
    if alpha > 0.0 and beta > 0.0 and 0.< Dm0 <1. and 0.< Dlambda0 <1. and M_c < 0.0 and sigma_int > 0.0:
        return 0.0
    return -np.inf

def log_prob(theta, mSN, sigma_m, s, sigma_s, Av, sigma_Av, z, sigma_z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, mSN, sigma_m, s, sigma_s, Av, sigma_Av, z, sigma_z)
