
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:02:58 2020

@author: geros
"""
import numpy as np
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
import math 
import pylab as pp
from pymaxent import reconstruct
import time
from pymaxent import reconstruct,maxent_reconstruct_c0,maxent_reconstruct_c1,temp
start = time.time()


'''Αρχικοποίση των μεταβλητών'''

tspan=(0, 100)
t=np.linspace(tspan[0],tspan[1],51)

initial_m=[]
for i in range(4):
    def distr(L,i=i):
        return (L**i)*3*L**2*np.exp(-L**3)
   
    m, err=integrate.quad(distr, 0, np.inf)
    print('m(',i,')=',m)
    initial_m.append(m)
mu=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0,lambdas=maxent_reconstruct_c0(mu=mu,bnds=[0,2])
temp = lambdas    
  
''' Επίλυση του συστήματος διαφορικών εξισώσεων με την Maximum Entropy'''

def moments(t,y):
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3]
    Lmean=m1/m0
    σ=np.abs(m2-Lmean**2)**(1/2)
    Lmin=Lmean-3*σ
    Lmax=Lmean+4*σ
    bnds=[0,Lmax]
    

    sol, lambdas= maxent_reconstruct_c1(mu=y ,bnds=bnds)

    print('time is', t)
    
 
        
    dm0dt=-m0**2/2
    
    def moment1(L,λ):

        return(((L**3+λ**3)**(1/3)/2-L)*sol(L)*sol(λ))
    dm1dt, err1=integrate.dblquad(moment1, bnds[0], bnds[1], bnds[0], bnds[1])
    
    def moment2(L,λ):

        return(((L**3+λ**3)**(2/3)/2-L**2)*sol(L)*sol(λ))
    dm2dt, err2=integrate.dblquad(moment2, bnds[0], bnds[1], bnds[0], bnds[1])
    
    dm3dt=0
    
    return(dm0dt,dm1dt,dm2dt,dm3dt)


'''Χρήση της BDF, step by step'''


r=integrate.solve_ivp(moments ,tspan, initial_m, method='BDF',t_eval=t, jac=None, rtol=10**(-3))


'''Χρήση γνωστής αναλυτικής σχέσης για τις ροπές'''


def Am0(t):
    return(initial_m[0]*(2/(2+t)))
def Am1(t):
    return(initial_m[1]*(2/(2+t))**(2/3))
def Am2(t):
    return(initial_m[2]*(2/(2+t))**(1/3))


pp.figure(0)

pp.plot(r.t,r.y[0,:]/initial_m[0],'.b',t,Am0(t)/initial_m[0],'-b')
pp.plot(r.t,r.y[1,:]/initial_m[1],'.r',t,Am1(t)/initial_m[1],'-r')
pp.plot(r.t,r.y[2,:]/initial_m[2],'.g',t,Am2(t)/initial_m[2],'-g')
pp.plot(r.t,r.y[3,:]/initial_m[3],'.y',[0,100],[1,1],'-y')

pp.xlabel('t(sec)',{"fontsize":16})
pp.ylabel('mκ(t)/mκ(0)',{"fontsize":16})
pp.title('Evolution of moments')
pp.yscale('log')
pp.legend(('Maximum Entropy','Analytical Solution'),loc=0)

pp.show()
                               


'''Κατασκευή γραφημάτων της κατανομής για συγκεκριμένες χρονικές στιγμές'''

'''t=0'''
L=np.linspace(0,2)

Lmean=r.y[1,0]/r.y[0,0]
σ=np.abs(r.y[2,0]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[0,4]




pp.figure(1)
mu0=[initial_m[0],initial_m[1],initial_m[2],initial_m[3]]
sol0, lambdas0=maxent_reconstruct_c0(mu=mu0,bnds=bnds)
pp.plot(L,sol0(L))
pp.title('t = 0')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})





'''Travelling wave'''
L=np.linspace(0, 8)

'''t=20'''

Lmean=r.y[1,10]/r.y[0,10]
σ=np.abs(r.y[2,10]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(2)

mu20=[r.y[0,10],r.y[1,10],r.y[2,10],r.y[3,10]]
sol20, lambdas20=reconstruct(mu=mu20,bnds=bnds)
pp.plot(L,sol20(L))
pp.title('t = 20')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''t=40'''

Lmean=r.y[1,20]/r.y[0,20]
σ=np.abs(r.y[2,20]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(3)
mu40=[r.y[0,20],r.y[1,20],r.y[2,20],r.y[3,20]]
sol40, lambdas40=reconstruct(mu=mu40,bnds=bnds)
pp.plot(L,sol40(L))
pp.title('t = 40')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''t=60'''

Lmean=r.y[1,30]/r.y[0,30]
σ=np.abs(r.y[2,30]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[0,Lmax]



pp.figure(4)
mu60=[r.y[0,30],r.y[1,30],r.y[2,30],r.y[3,30]]
sol60, lambdas60=reconstruct(mu=mu60,bnds=bnds)
pp.plot(L,sol60(L))
pp.title('t = 60')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})


'''t=80'''

Lmean=r.y[1,40]/r.y[0,40]
σ=np.abs(r.y[2,40]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[0,Lmax]



pp.figure(5)

mu80=[r.y[0,40],r.y[1,40],r.y[2,40],r.y[3,40]]
sol80, lambdas80=reconstruct(mu=mu80,bnds=bnds)
pp.plot(L,sol80(L))
pp.title('t = 80')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''t=100'''

Lmean=r.y[1,50]/r.y[0,50]
σ=np.abs(r.y[2,50]-Lmean**2)**(1/2)
Lmin=0
Lmax=Lmean+4*σ
bnds=[0,Lmax]


pp.figure(6)
L=np.linspace(0,8,51)
mu100=[r.y[0,50],r.y[1,50],r.y[2,50],r.y[3,50]]
sol100, lambdas100=reconstruct(mu=mu100,bnds=bnds)
pp.plot(L,sol100(L))
pp.title('t = 100')
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n #/mLμL',{"fontsize":16})



'''Ολικό γράφημα του travelling wave '''

pp.figure(6)
pp.plot(L,sol20(L),L,sol40(L),L,sol60(L),L,sol80(L),L,sol100(L))
pp.xlabel('L (μm)',{"fontsize":16})
pp.ylabel('n(#/μm.mL)',{"fontsize":16})
pp.legend(('20 sec','40 sec','60 sec','80 sec','100 sec'),loc=0)
pp.show 
end=time.time()

print('Συνολικός χρόνος κώδικα', end-start)




