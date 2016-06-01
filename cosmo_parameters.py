import numpy as np
import scipy as sp
from scipy import integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import math

#Dm0+Dde0=1 Flat Universe
c=299797. #km/s
#MSN=-19.5 #Magnitud absoluta de SN
H0=68 #Constante de Hubble

def e_z(z,omega_m,omega_k,omega_lambda):
		return (1.0/math.sqrt(omega_m*((1.+z)**3.)+ omega_k*((1.+z)**2.) + omega_lambda))

def mu_model(z,H0,Dm0):
    e_z_int, e_z_int_err = integrate.quad(lambda x: e_z(x,Dm0,0.,1. - Dm0),0.,z) #Universo plano
    return 25. + 5.*sp.log10( (c*(1.+z)/H0) * e_z_int)

def mu_data(m,M,alpha,s,beta,Av):
    return m - M +alpha*(s-1) - beta*(Av)

def error_data(sigma_z,z,sigma_m,alpha,sigma_s,beta,sigma_Av,sigma_int):
    error=sp.array([0.]*len(z))
    new_sigma_z=sp.array([0.]*len(z))
    for i in range(0,len(error)):
        new_sigma_z[i] = sigma_z[i] * (5.0*(1.0+z[i]))/(z[i]*(1.0 + z[i]/2.0)*sp.log(10.0))
        error[i] = sigma_m[i]**2 + (alpha*sigma_s[i])**2 + (beta*sigma_Av[i])**2 + new_sigma_z[i]**2 + sigma_int**2
        error[i] = sp.sqrt(error[i])
    return error

def m_model(z,s,Av,M,alpha,beta,Dm0,Dde0):
    integral = 5*sp.log10(c * (1+z) * integrate.quad(lambda x: 1/sp.sqrt( Dm0*(1+x)**3 + Dde0), 0, z)[0])
    return beta*Av - alpha*(s-1) + M + integral

def Hubble(M,M1):
    return 10**((M+25.0-M1)/5.)

def QuitarNan(A): #De la tabla general quita los valores que tienen NaN
    B=[None]*len(A)
    B=sp.array(B)
    for i in range (0,len(A)):
        for j in range (0,len(A)):
            B[j]=A[j][sp.where(sp.isnan(A[i])==False)]
    return B

#Extraemos data
print 'Extrayendo informacion ...\n'

archivo = open("listjavi.txt", "r")
lineas = archivo.readlines()

mSN= [] #List with the SN magnitudes
sigma_m=[]
Z=  [] #List with the SN redshifts
sigma_Z=[]
S= [] #List with the SN stres
sigma_S=[]
AV= [] #List with the SN color variability
sigma_Av=[]

for line in lineas:
    ind=lineas.index(line)
    if ind!=0:
        aux=line.split( )
        Z.append(float(aux[1])) #El 0 es el nombre de la SN
        sigma_Z.append(float(aux[2]))
        mSN.append(float(aux[3]))
        sigma_m.append(float(aux[4]))
        S.append(float(aux[5]))
        sigma_S.append(float(aux[6]))
        AV.append(float(aux[7]))
        sigma_Av.append(float(aux[8]))


#Pasamos a scipy arrays
mSN= sp.array(mSN)#List with the SN magnitudes
sigma_m= sp.array(sigma_m)
Z= sp.array(Z) #List with the SN redshifts
sigma_Z= sp.array(sigma_Z)
S= sp.array(S) #List with the SN stres
sigma_S = sp.array(sigma_S)
AV= sp.array(AV) #List with the SN colors
sigma_Av = sp.array(sigma_Av)

#Quitamos los nan
print 'Quitando NaN ...\n'

Data=[Z, sigma_Z, mSN, sigma_m, S, sigma_S, AV, sigma_Av]
Data=sp.array(Data)
AUX=QuitarNan(Data)
Z=AUX[0]
sigma_Z=AUX[1]
mSN=AUX[2]
sigma_m=AUX[3]
S=AUX[4]
sigma_S=AUX[5]
AV=AUX[6]
sigma_Av=AUX[7]
print 'Analizando '+str(len(Z))+' SNs ...\n'

#Parametros: Adivinanzas originales

alpha_ini= 0.5
beta_ini= 3.0 #3.1 2.5
Dm0_ini= 0.28 #Universo Plano Dm0+Dlambda0=1
#M_ini= -3.4 #-3.7 Ir cambiando/ M cursiva
M_ini=-19.5 #Magnitud absoluta SN
sigma_ini= 0.5


def log_likelihood(theta, mSN, sigma_m, s, sigma_s, Av, sigma_Av, z, sigma_z):
    alpha, beta, Dm0, M, sigma_int  = theta
    #Calculamos el error total
    new_sigma_z = sp.array([0.]*len(sigma_z))
    sigma2 = sp.array([0.]*len(sigma_z))
    likelihood=sp.array([0.]*len(sigma_z))
    #Calculamos magnitud modelo, creando un array de 0s
    m=[0.]*len(mSN)
    m=sp.array(m)
    #Calculamos la integral elemento a elemento
    for i in range(0,len(m)):
        integral = 5.0*sp.log10((c * (1.0+z[i])/H0) * integrate.quad(lambda x: 1.0/sp.sqrt( Dm0*((1.0+x)**3 - 1.0) + 1.0 ), 0.0, z[i])[0])
        m[i] = beta*Av[i] - alpha*(s[i]-1.0) + M + 25 + integral
        new_sigma_z[i] = sigma_z[i] * ( 5.0 * (1.0+z[i]) ) / ( z[i]*(1.0 + z[i]/2.0) * sp.log(10.0) )
        sigma2[i] = sigma_m[i]**2 + (alpha*sigma_s[i])**2 + (beta*sigma_Av[i])**2 + new_sigma_z[i]**2 + sigma_int**2
        likelihood[i]= ( ((mSN[i]-m[i])**2) / sigma2[i]  +  sp.log(sigma2[i]*2*np.pi) )

    return -0.5 * sp.sum(likelihood)

print 'Minimizacion ...\n'

nll = lambda *args: -log_likelihood(*args)
bnds = ((0., None), (0., None),(0.,1.),(None,0.),(0.,None))
result = minimize(nll, [alpha_ini, beta_ini, Dm0_ini, M_ini, sigma_ini], args=(mSN, sigma_m, S, sigma_S, AV, sigma_Av, Z, sigma_Z),method='L-BFGS-B',bounds=bnds)

print 'Resultados Minimizacion: ...\n'

alpha_min, beta_min, Dm0_min, M_min, sigma_min = result["x"]

print result["x"]

mu_data_min = mu_data(mSN,M_min,alpha_min,S,beta_min,AV)
err_data_min = error_data(sigma_Z,Z,sigma_m,alpha_min,sigma_S,beta_min,sigma_Av,sigma_min)
Z_modelmin=sp.sort(Z)

mu_model_min =sp.array([0.]*len(Z_modelmin))
for i in range(0,len(Z_modelmin)):
    mu_model_min[i] = mu_model(Z_modelmin[i],H0,Dm0_min)

#Ploteamos modulo de distancia y diferencia teorico-data
plt.figure()
plt.subplot(211)
plt.plot(Z_modelmin, mu_model_min)
plt.errorbar(Z,mu_data_min,yerr=err_data_min,fmt='v')
plt.ylabel(r'$\mu$')

dif_min=sp.array([0.]*len(Z))
for i in range(0,len(Z)):
    for j in range(0,len(Z_modelmin)):
        if Z[i]==Z_modelmin[j]:
            dif_min[i]=mu_data_min[j]-mu_model_min[i]

plt.subplot(212)
plt.errorbar(Z,dif_min,yerr=err_data_min,fmt='v')
plt.xlabel(r'$z$')
plt.ylabel(r'$\mu_{data} - \mu_{model}$')
plt.savefig('mod_dist_min_9.png')

#Minimizacon con MCMC
print 'Minimizacion usando MCMC ...\n'

def log_prior(theta):
    alpha, beta, Dm0, M, sigma_int  = theta
    if alpha > 0.0 and beta > 0.0 and 0.0 < Dm0 < 1.0 and M < 0.0 and sigma_int > 0.0:
        return 0.0
    return -np.inf

def log_prob(theta, mSN, sigma_m, s, sigma_s, Av, sigma_Av, z, sigma_z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, mSN, sigma_m, s, sigma_s, Av, sigma_Av, z, sigma_z)

#Preparamos la caminata aleatoria
ndim, nwalkers = 5, 50 #5 parametros

print 'pos ...\n'
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

print 'emcee ...\n'
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(mSN, sigma_m, S, sigma_S, AV, sigma_Av, Z, sigma_Z))
print 'mcmc ...\n'
sampler.run_mcmc(pos, 100)

print 'sampler chain ...\n'
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

print 'corner ...\n'
import corner

fig = corner.corner(samples, labels=[r"$\alpha$", r"$\beta$", r"$\Omega_{m}$", r"$M$", r"$\sigma$"],
                      truths=[alpha_min, beta_min, Dm0_min, M_min, sigma_min])
fig.savefig("corner_9.png")

#Obtenemos los resultados obtenidos con MCMC
samples[:, 2] = np.exp(samples[:, 2])
alpha_mcmc, beta_mcmc, Dm0_mcmc, M_mcmc, sigma_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

#Ploteamos y guardamos los graficos del modulo de distancia vs z
mu_data_mcmc=mu_data(mSN,M_mcmc[0],alpha_mcmc[0],S,beta_mcmc[0],AV)
err_data_mcmc=error_data(sigma_Z,Z,sigma_m,alpha_mcmc[0],sigma_S,beta_mcmc[0],sigma_Av,sigma_mcmc[0])
Z_model_mcmc=sp.sort(Z)

mu_model_mcmc =sp.array([0.]*len(Z_model_mcmc))
for i in range(0,len(mu_model_mcmc)):
    mu_model_mcmc[i] = mu_model(Z_model_mcmc[i],H0,Dm0_mcmc[0])

plt.figure()
plt.subplot(211)
plt.plot(Z_model_mcmc, mu_model_mcmc)
plt.errorbar(Z,mu_data_mcmc,yerr=err_data_mcmc,fmt='^')
plt.ylabel(r'$\mu$')

dif_mcmc=sp.array([0.]*len(Z))
for i in range(0,len(Z)):
    for j in range(0,len(Z_model_mcmc)):
        if Z[i]==Z_model_mcmc[j]:
            dif_mcmc[i]=mu_data_mcmc[j]-mu_model_mcmc[i]

plt.subplot(212)
plt.errorbar(Z,dif_mcmc,yerr=err_data_mcmc,fmt='^')
plt.xlabel(r'$z$')
plt.ylabel(r'$\mu_{data} - \mu_{model}$')
plt.savefig('mod_dist_mcmc_9.png')
