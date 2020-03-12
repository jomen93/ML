    # -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_1 import N

"""
Author : Johan Méndez
"""

######################### Exercise 1 ############################################
## MVE

# Data from exercise 1
 
data ={
   "Nombres":["Denis","Guadalupe","Alex","Alex", "Cris",
               "Juan","Juan","Denis","Alex","Cris","Rene","Guadalupe","Guadalupe"],
   "Estatura (m)":[1.72,1.82,1.8,1.7,1.73,1.8,1.8,1.5,1.52,1.62,1.67,1.65,1.75],
   "Peso (kg)":[75.3,81.6,86.1,77.1,78.2,74.8,74.3,50.5,45.3,61.2,68.0,58.9,68.0],
   "Genero":["M","M","M","M","M","M","M","F","F","F","F","F","F"],
   }

df = pd.DataFrame(data, columns = ["Nombres","Estatura (m)","Peso (kg)","Genero"])
data = df.to_numpy()


NM = float(sum(data[:,-1:] == "M")) # Number of class H data
NF = float(sum(data[:,-1:] == "F")) # Number of class F data
NT  = np.shape(data)[0]             # Total number of data

mu_Est = np.mean(data[:,1:2]) ; std_Est = np.std(data[:,1:2]) 
mu_Pes = np.mean(data[:,2:3]) ; std_Pes = np.std(data[:,2:3])


Mas = data[data[:,-1]=="M"][:,:-1]
Fem = data[data[:,-1]=="F"][:,:-1]
NamM = data[data[:,-1]=="M"][:,:1]
NamF = data[data[:,-1]=="F"][:,:1]
# Maximum likelihood estimator

# Normal distribution
mu_EM = np.mean(Mas[::,-2]); std_EM = np.std(Mas[::,-2])
mu_EF = np.mean(Fem[::,-2]); std_EF = np.std(Fem[::,-2])
mu_PM = np.mean(Mas[::,-1]); std_PM = np.std(Mas[::,-1])
mu_PF = np.mean(Fem[::,-1]); std_PF = np.std(Fem[::,-1])

# Categorical Distribution
 
q_M = np.asarray(np.unique(NamM,return_counts=True))
q_F = np.asarray(np.unique(NamF,return_counts=True))

qM_prov = {q_M[0][i]:q_M[1][i] for i in range(len(q_M[0]))}
qF_prov = {q_F[0][i]:q_F[1][i] for i in range(len(q_F[0]))}


# Redefinition of variables to make easy the algoritmh

mu = np.array([mu_EM,mu_PM,mu_EF,mu_PF])
std = np.array([std_EM,std_PM,std_EF,std_PF]) 

## Sample to predict

X_p =  [["Rene",1.68,65.],
             ["Guadalupe",1.75,80.],
             ["Denis",1.80,79.],
             ["Alex",1.90,85.],
             ["Cris",1.65,70]]


#"Estimator function"

def MVE(x):
   InCMdata = x[0] in qM_prov; InCFdata = x[0] in qF_prov
   CM = 0; CF = 0
   
   if InCMdata == True:
      CM = (NM/NT)*np.prod(N(x[1:],mu[:2],std[:2]))*qM_prov[str(x[0])]/NM
   if InCFdata == True:
      CF = (NF/NT)*np.prod(N(x[1:],mu[2:4],std[2:4]))*qF_prov[str(x[0])]/NF
   clases = {CM:"Masculino",CF:"Femenino"}
   return x,clases[max(CM,CF)],CM,CF
   
print(" ")
print("Reesults EMV")
print(" ")
print("parameters")
print(" mu_EM = {0:.3f}".format(mu_EM))
print(" mu_EF = {0:.3f}".format(mu_EF))
print(" mu_PM = {0:.3f}".format(mu_PM))
print(" mu_PF = {0:.3f}".format(mu_PF))
print(" ")
# Prediction fo data
print("predictions")
for i in range(len(X_p)):
   print(MVE(X_p[i]))

print(" ")



### MAP

# parameters definition
alpha = 1 
muoEM = 1.7;  muoEF = 1.5;  sigmaoEM_2 = 0.3;  sigmaoEF_2 = 0.1
muoPM = 85.5; muoPF = 70.3; sigmaoPM_2 = 17.0; sigmaoPF_2 = 85.0

mu_EM2 = (sigmaoEM_2*sum(Mas[::,-2])+(std_EM**2)*muoEM)/(sigmaoEM_2*NM + std_EM**2)
mu_EF2 = (sigmaoEF_2*sum(Fem[::,-2])+(std_EF**2)*muoEF)/(sigmaoEF_2*NF + std_EF**2)
mu_PM2 = (sigmaoPM_2*sum(Mas[::,-1])+(std_EM**2)*muoPM)/(sigmaoPM_2*NM + std_PM**2)
mu_PF2 = (sigmaoPF_2*sum(Fem[::,-1])+(std_EF**2)*muoPF)/(sigmaoPF_2*NF + std_PF**2)

mu2 = np.array([mu_EM2,mu_PM2,mu_EF2,mu_PF2])

def MAP(x):
   InCMdata = x[0] in qM_prov; InCFdata = x[0] in qF_prov
   CM = 0; CF = 0
   
   if InCMdata == True:
      CM = (NM/NT)*np.prod(N(x[1:],mu2[:2],std[:2]))*qM_prov[str(x[0])]/NM
   if InCFdata == True:
      CF = (NF/NT)*np.prod(N(x[1:],mu2[2:4],std[2:4]))*qF_prov[str(x[0])]/NF
   clases = {CM:"Masculino",CF:"Femenino"}
   return x,clases[max(CM,CF)],CM,CF

print(" ")
print("Results MAP")
print(" ")
print("parameters")
print(" mu_EM = {0:.3f}".format(mu_EM2))
print(" mu_EF = {0:.3f}".format(mu_EF2))
print(" mu_PM = {0:.3f}".format(mu_PM2))
print(" mu_PF = {0:.3f}".format(mu_PF2))
print(" ")
print("predictions")
for i in range(len(X_p)):
   print(MAP(X_p[i]))
print(" ")




