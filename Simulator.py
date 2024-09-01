import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import scipy as sp
from scipy import stats as st
import sympy as sym
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.integrate import solve_ivp
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['figure.dpi'] = 300

#Main
st.set_page_config(layout="wide")
st.title("Glucose simulator")

#Slidebar
st.sidebar.subheader('1, Select parameter values')

length=2880
t_span = [0,length]
D=20000
t_eval = np.linspace(*t_span,length) # time for sampling
Input=pd.DataFrame([[0, 0],
                    [360, 0], [362, (1/2)*D], [364, 0],
                    [720, 0], [722, (3/2)*(1/2)*D], [724, 0],
                    [1080, 0], [1082, 2*(1/2)*D], [1084, 0],
                    [1440, 0],
                    [1440+360, 0], [1440+390, (1/30)*D], [1440+420, 0],
                    [1440+720, 0], [1440+750, (3/2)*(1/30)*D], [1440+780, 0],
                    [1440+1080, 0], [1440+1110, 2*(1/30)*D], [1440+1140, 0],
                    [1440+1440,0]],columns=['time', 'glucose'])
f1= interpolate.interp1d(Input["time"], Input["glucose"],kind='linear') #G補完 

#parameter
VG = st.sidebar.slider('VG', 1.49, 1.88, 0.02)
k1=0.065
k2=0.079
VI=0.05
m1=0.190
m2=0.484
m4=0.194
m5=0.0304
m6=0.6471
kmax=0.0558
kmin=0.0080
kabs=0.057
kgri=0.0558
f=0.90
a=0.00013 
b=0.82
c=0.00236
d=0.010
kp1=2.70
kp2=0.0021
kp3=0.009
kp4=0.0618
ki=0.0079
Fcns=1
Vmo=2.50
Vmx=0.047
Kmo=225.59
p2U=0.0331
K=2.30
alpha=0.050
beta=0.11
gamma=0.5
ke1=0.0005
ke2=339
BW=78

#initial
#Y,Ipo,Il,Ip,I1,Id,X,Qsto1,Qsto2,Qgut,Gp,Gt
HEb=0.6 #simulationには使わない
Yb=0
Sb=(m6-HEb)/m5 #simulationには使わない
Ipob=Sb/gamma
Ipb=0.4*Sb*(1-HEb)/m4
m3b=HEb*m1/(1-HEb)  #simulationには使わない
Ilb=(Sb-m4*Ipb)/m3b
I1b=Ipb/VI
Idb=Ipb/VI
Ib=Ipb/VI
Xb=0
Qsto1b=0
Qsto2b=0
Qgutb=0

def func(x, Gtb,Gpb):
    eq1=kp1-kp2*Gpb-kp3*Ib-kp4*Ipob-(Fcns+k1*Gpb-k2*Gtb)
    eq2=kp1-kp2*Gpb-kp3*Ib-kp4*Ipob-(Fcns+(Vmo*Gtb/(Kmo+Gtb)))
    return np.array([eq1,eq2])

x=np.arange(2)
y=np.array([0,0])
popt, pcov = curve_fit(func,x, y,p0=[100,100], maxfev=1000000)
Gpb=popt[1]
Gtb=popt[0]
h=Gpb/VG
init   = [Yb,Ipob,Ilb,Ipb,I1b,Idb,Xb,Qsto1b,Qsto2b,Qgutb,Gpb,Gtb]

def glu(t,XYAR, VG,k1,k2,VI,m1,m2,m4,m5,m6,kmax,kmin,kabs,kgri,f,b,c,
        kp1,kp2,kp3,kp4,ki,Fcns,Vmo,Vmx,Kmo,p2U,K,alpha,beta,gamma,ke1,ke2,h):
    Y,Ipo,Il,Ip,I1,Id,X,Qsto1,Qsto2,Qgut,Gp,Gt= XYAR
    G=Gp/VG
    #insulin secretion
    S=gamma*Ipo
    
    #insulin dynamics
    HE=-m5*S+m6
    m3=m1*HE/(1-HE)
    dIl=-(m1+m3)*Il+m2*Ip+S
    dIp=-(m2+m4)*Ip+m1*Il
    I=Ip/VI

    #insulin dynamics2
    dI1=-ki*(I1-I)
    dId=-ki*(Id-I1)
    dX=-p2U*X+p2U*(I-Ib)

    #Glucose appearabce
    Qsto=Qsto1+Qsto2
    alpha1=5/(2*D*(1-b))
    beta1=5/(2*D*c)
    kempt=kmin+((kmax-kmin)/2)*(np.tanh(alpha1*(Qsto-b*D))-np.tanh(beta1*(Qsto-c*D))+2)
    dQsto1=-kgri*Qsto1+f1(t)
    dQsto2=-kempt*Qsto2+kgri*Qsto1
    dQgut=-kabs*Qgut+kempt*Qsto2
    Ra=f*kabs*Qgut/BW
    
    #Glucose renal excretion
    E=np.max([ke1*(Gp-ke2),0])
    
    #glucose subsystem
    EGP=kp1-kp2*Gp-kp3*Id-kp4*Ipo
    Uii=Fcns
    Vm=Vmo+Vmx*X
    Kmx=0
    Km=Kmo+Kmx*X
    Uid=Vm*Gt/(Km+Gt)
    dGp=EGP+Ra-Uii-E-k1*Gp+k2*Gt
    dGt=-Uid+k1*Gp-k2*Gt

    #insulin secretion
    dY=alpha*np.max([beta*(G-h),-Sb])-alpha*Y
    Spo=Y+K*np.max([dGp/VG,0])+Sb
    dIpo=Spo-gamma*Ipo

    return [dY,dIpo,dIl,dIp,dI1,dId,dX,dQsto1,dQsto2,dQgut,dGp,dGt]
sol = solve_ivp(glu,t_span,init,method='LSODA',t_eval=t_eval,
                args=(VG,k1,k2,VI,m1,m2,m4,m5,m6,kmax,kmin,kabs,kgri,f,b,c,
                      kp1,kp2,kp3,kp4,ki,Fcns,Vmo,Vmx,Kmo,p2U,K,alpha,beta,gamma,ke1,ke2,h),
                rtol = 10**(-13),atol = 10**(-16))
Y,Ipo,Il,Ip,I1,Id,X,Qsto1,Qsto2,Qgut,Gp,Gt= sol.y

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(np.arange(length),Gp/VG,color="black")
ax.set_title("Plasma glucose")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(4,3))
#plt.plot(np.arange(1440),Qgut)
plt.plot(np.arange(length),Ip/VI,color="black")
ax.set_title("plasma insulin")
st.pyplot(fig)
dff=pd.DataFrame(sm.tsa.stattools.acf((Gp/VG)[0::5],nlags=30,fft=False))

st.write("Mean") 
st.write(np.mean(Gp/VG)) 
st.write('Std')
st.write(np.std(Gp/VG)) 
st.write('AC_Var')
st.write(dff.iloc[1:].var()[0]) 