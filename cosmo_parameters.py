import numpy as np
import scipy as sp
from scipy import integrate
from scipy.optimize import minimize
from cosmo_functions import *
import matplotlib.pyplot as plt
import emcee
import math

#Extraemos data
print 'Extrayendo informacion ...\n'
archivo = open("listjavi.txt", "r")
lineas = archivo.readlines()

#Listas con magnitudes, z,stress, color variability
mSN= []; sigma_m=[]; Z=  []; sigma_Z=[]; S= []; sigma_S=[]; AV= []; sigma_Av=[]

for line in lineas:
    ind=lineas.index(line)
    if ind!=0:
        aux=line.split( )
        Z.append(float(aux[1])); sigma_Z.append(float(aux[2])); mSN.append(float(aux[3])); sigma_m.append(float(aux[4]))
        S.append(float(aux[5])); sigma_S.append(float(aux[6])); AV.append(float(aux[7])); sigma_Av.append(float(aux[8]))

#Pasamos a scipy arrays
mSN= sp.array(mSN); sigma_m= sp.array(sigma_m); Z= sp.array(Z); sigma_Z= sp.array(sigma_Z); S= sp.array(S); sigma_S = sp.array(sigma_S); AV= sp.array(AV); sigma_Av = sp.array(sigma_Av)

#Quitamos los nan
print 'Quitando NaN ...\n'
Data=sp.array([Z, sigma_Z, mSN, sigma_m, S, sigma_S, AV, sigma_Av])
Z, sigma_Z, mSN, sigma_m, S, sigma_S, AV, sigma_Av = QuitarNan(Data)

#Parametros: Adivinanzas originales
alpha_ini= 0.5
beta_ini= 3.0 #3.1 2.5
Dm0_ini= 0.3 #Universo Plano Dm0+Dlambda0=1
Dlambda0_ini=0.7
M_c_ini= -3.4 #-3.7 Ir cambiando/ M cursiva
#M_ini=-19.5 #Magnitud absoluta SN
sigma_ini= 0.5

print 'Minimizacion con optimize minimize...\n'
nll = lambda *args: -log_likelihood(*args)
bnds = ((0., None), (0., None),(0.,1.),(0.,1.),(None,0.),(0.,None)) #Limites para parametros
result = minimize(nll, [alpha_ini, beta_ini, Dm0_ini, Dlambda0_ini, M_c_ini, sigma_ini], args=(mSN, sigma_m, S, sigma_S, AV, sigma_Av, Z, sigma_Z),method='L-BFGS-B',bounds=bnds)

print 'Resultados Minimizacion: ...\n'
alpha_min, beta_min, Dm0_min, Dlambda0_min, M_c_min, sigma_min = result["x"]
print result["x"]

H0 = 68 #Definir constante de hubble

mu_data_min = mu_data(mSN,M_c_min,H0,alpha_min,S,beta_min,AV)
err_data_min = error_data(sigma_Z,Z,sigma_m,alpha_min,sigma_S,beta_min,sigma_Av,sigma_min)
Z_modelmin=sp.linspace(min(Z),max(Z),100)

mu_model_min =sp.array([0.]*len(Z_modelmin)); dif_min=sp.array([0.]*len(Z))
for i in range(0,len(Z_modelmin)):
    mu_model_min[i] = mu_model(Z_modelmin[i],H0,Dm0_min,Dlambda0_min)
for i in range(0,len(Z)):
    dif_min[i] = mu_data_min[i]-mu_model(Z[i],H0,Dm0_min,Dlambda0_min)

#Ploteamos modulo de distancia y diferencia teorico-data
plt.figure()
plt.subplot(211)
plt.plot(Z_modelmin, mu_model_min)
plt.errorbar(Z,mu_data_min,yerr=err_data_min,fmt='v')
plt.ylabel(r'$\mu$')
plt.subplot(212)
plt.errorbar(Z,dif_min,yerr=err_data_min,fmt='v')
plt.xlabel(r'$z$')
plt.ylabel(r'$\mu_{data} - \mu_{model}$')
plt.savefig('mod_dist_min_17.png')

#Minimizacon con MCMC
print 'Minimizacion usando MCMC ...\n'

#Preparamos la caminata aleatoria
ndim, nwalkers = 6, 500 #5 parametros #6 con DE

print 'pos ...\n'
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
print 'emcee ...\n'
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(mSN, sigma_m, S, sigma_S, AV, sigma_Av, Z, sigma_Z))
print 'mcmc ...\n'
sampler.run_mcmc(pos, 1000)
print 'sampler chain ...\n'
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
print 'corner ...\n'
import corner

fig = corner.corner(samples, labels=[r"$\alpha$", r"$\beta$", r"$\Omega_{m}$", r"$\Omega_{\Lambda}$", r"$M_c$", r"$\sigma$"],
                      truths=[alpha_min, beta_min, Dm0_min, Dlambda0_min, M_c_min, sigma_min]) #Universo Plano
fig.savefig("corner_17.png")

#Obtenemos los resultados obtenidos con MCMC
samples[:, 2] = np.exp(samples[:, 2])
alpha_mcmc, beta_mcmc, Dm0_mcmc, Dlambda0_mcmc, M_c_mcmc, sigma_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
#Ploteamos y guardamos los graficos del modulo de distancia vs z
mu_data_mcmc=mu_data(mSN,M_c_mcmc[0],H0,alpha_mcmc[0],S,beta_mcmc[0],AV)
err_data_mcmc=error_data(sigma_Z,Z,sigma_m,alpha_mcmc[0],sigma_S,beta_mcmc[0],sigma_Av,sigma_mcmc[0])
Z_model_mcmc=sp.linspace(min(Z),max(Z),100)

mu_model_mcmc =sp.array([0.]*len(Z_model_mcmc)); dif_mcmc=sp.array([0.]*len(Z))
for i in range(0,len(Z_model_mcmc)):
    mu_model_mcmc[i] = mu_model(Z_model_mcmc[i],H0,Dm0_mcmc[0],Dlambda0_mcmc[0])
for i in range(0,len(Z)):
    dif_mcmc[i] = mu_data_mcmc[i]-mu_model(Z[i],H0,Dm0_mcmc[0],Dlambda0_mcmc[0])

plt.figure()
plt.subplot(211)
plt.plot(Z_model_mcmc, mu_model_mcmc)
plt.errorbar(Z,mu_data_mcmc,yerr=err_data_mcmc,fmt='^')
plt.ylabel(r'$\mu$')
plt.subplot(212)
plt.errorbar(Z,dif_mcmc,yerr=err_data_mcmc,fmt='^')
plt.xlabel(r'$z$')
plt.ylabel(r'$\mu_{data} - \mu_{model}$')
plt.savefig('mod_dist_mcmc_17.png')
