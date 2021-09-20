# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:55:12 2021

@author: geros
"""


import pandas as pd
import numpy as np
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
import math 
import pylab as pp
from pymaxent import reconstruct
import time
from pymaxent import reconstruct,maxent_reconstruct_c0,maxent_reconstruct_c1,temp
import time 
import scipy.optimize as optimize
from scipy import integrate, interpolate
from scipy import optimize

start=time.time()
'''Αρχικοποίηση μεταβλητών'''
'''Data import'''

excel = pd.read_excel (r'C:\Users\geros\OneDrive\Υπολογιστής\Διπλωματική\Data_michalis\Data_Michalis.xlsx')
dt = pd.DataFrame(excel, columns= ['Sheet1'])

values=np.array(excel.values)


'''Χρ΄όνος σε sec'''

t=[]
for i in range (0,51):
    trial=values[i,0]
    t.append(trial)
t=np.array(t)

'''Συγκέντρωση'''

C=[]
for i in range (0,51):
    trial=values[i,1]
    C.append(trial)
C=np.array(C)

'''Γράφημα συγκέντρωσης'''

pp.figure(0)
pp.plot(t[0:51],C)
pp.xlabel('t(sec)',{"fontsize":16})
pp.ylabel('C',{"fontsize":16})
pp.legend(('From Theory'),loc=0)

'''Ροπές'''

m0=[]
for i in range (0,51):
    trial=values[i,3]
    m0.append(trial)
m0=np.array(m0)


m1=[]
for i in range (0,51):
    trial=values[i,4]
    m1.append(trial)
m1=np.array(m1)


m2=[]
for i in range (0,51):
    trial=values[i,5]
    m2.append(trial)
m2=np.array(m2)



m3=[]
for i in range (0,51):
    trial=values[i,6]
    m3.append(trial)
m3=np.array(m3)

'''Κατανομές για διαφορους χρόνους'''
'''t=0'''

L=[]
for i in range (0,1001):
    trial=values[i,11]
    L.append(trial)
L=np.array(L)


n0=[]
for i in range (0,1001):
    trial=values[i,12]
    n0.append(trial)
n0=np.array(n0)


'''t=0.5'''

n1=[]
for i in range (0,1001):
    trial=values[i,13]
    n1.append(trial)
n1=np.array(n1)


'''t=1'''
n2=[]
for i in range (0,1001):
    trial=values[i,14]
    n2.append(trial)
n2=np.array(n2)

'''t=2'''

n3=[]
for i in range (0,1001):
    trial=values[i,15]
    n3.append(trial)
n3=np.array(n3)

'''t=5'''

n4=[]
for i in range (0,1001):
    trial=values[i,16]
    n4.append(trial)
n4=np.array(n4)



pp.figure(2)
pp.plot(L,n0,L,n1,L,n2,L,n3,L,n4)
pp.xlabel('L(m)',{"fontsize":16})
pp.ylabel('n(#/m.m3)',{"fontsize":16})
pp.legend(('t=0','t=0.5','t=1','t=2','t=5'),loc=0)


'''Γραφήματα εξέλιξης ροπών'''


pp.figure(1)
pp.plot(t,m0,'r',t,m1,'b',t,m2,'y',t,m3,'g')
pp.xlabel('t(sec)',{"fontsize":16})
pp.ylabel('Moments',{"fontsize":16})
pp.legend(('m0', 'm1','m2','m3'),loc=0)


mu=[m0[0],m1[0],m2[0],m3[0]]
sol0,lambdas=maxent_reconstruct_c0(mu=mu,bnds=[0,2])
temp = lambdas    





'''Μοντελοποίηση για εύρεση kg, αρχικά μόνο growth'''

x_data = t[0:51]
y_data = C

def f(t, y, k): 
    """define the ODE system in terms of 
        dependent variable y,
        independent variable t, and
        optinal parmaeters, in this case a single variable k """
    rc = 2500
    kv=1
    s = 1
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3] 
    C = y[4] 
    dm0dt = 0
    dm1dt = k[0]*m0*(C/s-1)
    dm2dt = 2*k[0]*m1*(C/s-1)
    dm3dt =3*k[0]*m2*(C/s-1)
    dCdt = -kv*rc*3*k[0]*(C/s-1)*m2
    return (dm0dt,dm1dt,dm2dt,dm3dt,dCdt)

def my_ls_func(x,teta):
    """definition of function for LS fit
        x gives evaluation points,
        teta is an array of parameters to be varied for fit"""
    # create an alias to f which passes the optional params    
    f2= lambda t,y: f(t, y, teta) # calculate ode solution, retuen values for each entry of "x"
    r = integrate.solve_ivp(f2,[x_data[0],x_data[50]],y0,method='BDF',jac=None,t_eval=x_data,rtol=10**-4) 
    #in this case, we only need one of the dependent variable values
    return r.y[4,:]

def f_resid(p):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    return (y_data-my_ls_func(x_data,p))

guess = [10**(-4)]
y0 = [m0[0],m1[0], m2[0], m3[0], y_data[0]]
(kg, kvg) = optimize.leastsq(f_resid, guess)
print ( "Growth parameter: kg=", kg) 
xeval=np.linspace(min(x_data), max(x_data),51) 

gls = interpolate.UnivariateSpline(xeval, my_ls_func(xeval,kg), k=3, s=0)

#pick a few more points for a very smooth curve, then plot 
#   data and curve fit
xeval=np.linspace(min(x_data), max(x_data),200)
#Plot of the data as red dots and fit as blue line
pp.figure(3)
pp.plot(x_data, y_data,'.r',xeval,gls(xeval),'-b')
pp.xlabel('t (s)',{"fontsize":16})
pp.ylabel("C (kg/kg-solv)",{"fontsize":16})
pp.legend(('data','fit'),loc=0)
pp.show()

             

'''Μοντελοποίση με χρήση του kg'''


def moments(t,y,kg):
    m0 = y[0]
    m1 = y[1]
    m2 = y[2]
    m3 = y[3]
    C = y[4]
    rc = 2500
    kv=1
    s = 1

    
    Lmax=2
    bnds=[0,Lmax]
    
    sol, lambdas=reconstruct(mu=[m0,m1,m2,m3],bnds=bnds)
    
    '''Μηδενική ροπή'''
    
    '''Breakage'''
    k=0
    def moment0(L):
        return(L**(k+3)*sol(L))
    
    
    
    temp0, err0=integrate.quad(moment0, 0, Lmax)  
    
    dm0dt_B=(6/(k+3)-1)*temp0
    
        
    
    
    '''Aggregation'''
    
    dm0dt_A=-m0**2/2
    
       
    

    
    dm0dt=dm0dt_A+dm0dt_B
    
    
        
    '''Πρώτη ροπή'''
    
    '''Breakage'''
    k=1
    
    def moment1(L):
        return(L**(k+3)*sol(L))
    
    
    
    temp1, err1=integrate.quad(moment1, 0, Lmax) 
    
    dm1dt_B=(6/(k+3)-1)*temp1  
    
    '''Growth'''
    
    
    
    dm1dt_G=kg*m0*(C/s-1)
    
    '''Aggregation'''
    
    def moment1_A(L,λ):
        return(((L**3+λ**3)**(1/3)/2-L)*sol(L)*sol(λ))
    
    
    dm1dt_A, err1_A=integrate.dblquad(moment1_A, 0, 2, 0, 2)    
    

    
    dm1dt=dm1dt_G+dm1dt_A+dm1dt_B    
    
    '''Δεύτερη ροπή'''
    
    '''Breakage'''
    
    k=2
    def moment2(L):
        return(L**(k+3)*sol(L))
    
    
    
    temp2, err2=integrate.quad(moment2, 0, Lmax) 
    
    dm2dt_B=(6/(k+3)-1)*temp2      
    
    '''Growth'''
    dm2dt_G = 2*kg*m1*(C/s-1)
    
    '''Aggregation'''
    
    def moment2_A(L,λ):
        return(((L**3+λ**3)**(2/3)/2-L**2)*sol(L)*sol(λ))
    
    dm2dt_A, err2_A=integrate.dblquad(moment2_A, 0, 2, 0, 2)   
    
    dm2dt=dm2dt_G+dm2dt_A+dm2dt_B    
    
    '''Τρίτη ροπή'''
    
    '''Breakage'''
    k=3
    
    def moment3(L):
        return(L**(k+3)*sol(L))
    
    
    
    temp3, err3=integrate.quad(moment3, 0, Lmax) 
    
    dm3dt_B=(6/(k+3)-1)*temp3   
    '''Growth'''
    
    dm3dt_G =3*kg*m2*(C/s-1)   
    
    
    
    dm3dt=dm3dt_B+dm3dt_G    
      

    
    '''Συγκέντρωση'''
    
    dCdt = -kv*rc*3*kg*(C/s-1)*m2
    
    print('Χρονική στιγμ΄ή',t)
    return(dm0dt,dm1dt,dm2dt,dm3dt,dCdt)   




def my_ls_func(x,teta):
    """definition of function for LS fit
        x gives evaluation points,
        teta is an array of parameters to be varied for fit"""
    # create an alias to f which passes the optional params    
    f2= lambda t,y: moments(t, y, teta) # calculate ode solution, retuen values for each entry of "x"
    r = integrate.solve_ivp(f2,[x_data[0],x_data[50]],y0,method='BDF',jac=None,t_eval=x_data,rtol=10**-3) 
    #in this case, we only need one of the dependent variable values
    return r.y[4,:]

def f_resid(p):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    return (y_data-my_ls_func(x_data,p))


guess = [10**(-4)]
y0 = [m0[0],m1[0], m2[0], m3[0], y_data[0]]
(kg, kvg) = optimize.leastsq(f_resid, guess)
print ( "Growth parameter: kg=", kg, "μm/min") 
xeval=np.linspace(min(x_data), max(x_data),51) 

gls = interpolate.UnivariateSpline(xeval, my_ls_func(xeval,kg), k=3, s=0)

#pick a few more points for a very smooth curve, then plot 
#   data and curve fit
xeval=np.linspace(min(x_data), max(x_data),200)
#Plot of the data as red dots and fit as blue line
pp.plot(x_data, y_data,'.r',xeval,gls(xeval),'-b')
pp.xlabel('t (min)',{"fontsize":16})
pp.ylabel("C (kg)",{"fontsize":16})
pp.legend(('data','fit'),loc=0)
pp.show()


r1 = integrate.solve_ivp(moments,[x_data[0],x_data[50]],y0,method='BDF',jac=None,t_eval=x_data,rtol=10**-3,args=(kg,))   
    
pp.figure(4)


pp.plot(r1.t,r1.y[0,:],'.b',t,m0,'-b')
pp.plot(r1.t,r1.y[1,:],'.r',t,m1,'-r')
pp.plot(r1.t,r1.y[2,:],'.g',t,m2,'-g')
pp.plot(r1.t,r1.y[3,:],'.y',t,m3,'-y')

pp.xlabel('t(sec)',{"fontsize":16})
pp.ylabel('mκ(t)',{"fontsize":16})
pp.title('Evolution of moments')
pp.legend(('Maximum Entropy','PBE solution'))
pp.show()



'''Ανακατασκευή κατανομής'''

'''t=0'''

bnds=[0,4]

mu=[r1.y[0,0],r1.y[1,0],r1.y[2,0],r1.y[3,0]]
sol0, lambdas0=reconstruct(mu=mu,bnds=bnds)
pp.figure(3)
pp.plot(L,sol0(L),'r.',L,n0,'b-')
pp.xlabel('L (m)',{"fontsize":16})
pp.ylabel('n(#/m.m3)',{"fontsize":16})
pp.legend(('Maximum Entropy','PBE solution'),loc=0)
pp.title('t=0 sec')
 
'''t=0.5'''

bnds=[0,4]

mu=[r1.y[0,5],r1.y[1,5],r1.y[2,5],r1.y[3,5]]
sol1, lambdas1=reconstruct(mu=mu,bnds=bnds)
pp.figure(4)
pp.plot(L,sol1(L),'r.',L,n1,'b-')
pp.xlabel('L (m)',{"fontsize":16})
pp.ylabel('n(#/m.m3)',{"fontsize":16})
pp.legend(('Maximum Entropy','PBE solution'),loc=0)
pp.title('t=0.5 sec')
             
'''t=1'''

bnds=[0,4]

mu=[r1.y[0,10],r1.y[1,10],r1.y[2,10],r1.y[3,10]]
sol2, lambdas2=reconstruct(mu=mu,bnds=bnds)
pp.figure(5)
pp.plot(L,sol2(L),'r.',L,n2,'b-')
pp.xlabel('L (m)',{"fontsize":16})
pp.ylabel('n(#/m.m3)',{"fontsize":16})
pp.legend(('Maximum Entropy','PBE solution'),loc=0)
pp.title('t=1 sec')

             
'''t=2'''

bnds=[0,4]

mu=[r1.y[0,20],r1.y[1,20],r1.y[2,20],r1.y[3,20]]
sol3, lambdas3=reconstruct(mu=mu,bnds=bnds)
pp.figure(6)
pp.plot(L,sol3(L),'r.',L,n3,'b-')
pp.xlabel('L (m)',{"fontsize":16})
pp.ylabel('n(#/m.m3)',{"fontsize":16})
pp.legend(('Maximum Entropy','PBE solution'),loc=0)
pp.title('t=2 sec')

'''t=5'''

bnds=[0,4]

mu=[r1.y[0,50],r1.y[1,50],r1.y[2,50],r1.y[3,50]]
sol4, lambdas4=reconstruct(mu=mu,bnds=bnds)
pp.figure(7)
pp.plot(L,sol4(L),'r.',L,n4,'b-')
pp.xlabel('L (m)',{"fontsize":16})
pp.ylabel('n(#/m.m3)',{"fontsize":16})
pp.legend(('Maximum Entropy','PBE solution'),loc=0)
pp.title('t=5 sec')


end=time.time()

print('Συνολικός χρόνος κώδικα', end-start)













    
