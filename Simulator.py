import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import scipy as sp
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

st.set_page_config(layout="wide")

#Slidebar
st.sidebar.subheader('Select parameter values')

st.markdown("""
# Glucose Simulator Web Application

This web application is designed to simulate blood glucose dynamics and calculate key metrics related to glucose dynamics. It provides an interactive interface for users to adjust various parameters and observe their effects on glucose and insulin levels over time.

## Features

1. **Blood Glucose Simulation**: The app simulates blood glucose levels over a 48-hour period, incorporating meal inputs and insulin responses.

2. **Parameter Adjustment**: Users can modify key parameters using sliders in the sidebar:
   - m4: peripheral insulin clearance
   - Vmx: insulin sensitivity
   - K, alpha, beta: parameters related to insulin secretion capacity

3. **Visual Output**: The application generates three graphs:
   - Blood glucose levels over time
   - Blood insulin levels over time
   - Administered glucose (meal inputs) over time

4. **Key Metrics**: After simulation, the app calculates and displays:
   - Mean blood glucose (Mean)
   - Standard deviation (Std) of blood glucose
   - AC_Var: a metric calculated from the autocorrelation of blood glucose

## Underlying model

The blood glucose dynamics simulation is based on the glucose-insulin meal simulation model reported in the following research paper:

Dalla Man, Chiara, Robert A. Rizza, and Claudio Cobelli. "Meal simulation model of the glucose-insulin system." *IEEE Transactions on biomedical engineering* 54.10 (2007): 1740-1749.

This model provides a comprehensive representation of glucose-insulin interactions, including meal absorption, insulin secretion and action, and glucose utilization and production.

## Parameter interpretation

- m4 (peripheral insulin clearance): Affects how quickly insulin is removed from the bloodstream
- Vmx (insulin sensitivity): Represents how effectively cells respond to insulin
- K, alpha, beta: These parameters influence the insulin secretion capacity, simulating the pancreas's ability to produce insulin in response to glucose

## Reference values

For reference values of these parameters in normal glucose tolerance and diabetic populations, as well as detailed mathematical formulations of the model, please refer to the original research paper cited above.

## Usage

1. Adjust the parameters using the sliders in the sidebar
2. Click the "Simulate" button to run the simulation with the selected parameters
3. Observe the resulting graphs and calculated metrics
4. Experiment with different parameter combinations to understand their effects on glucose dynamics
""")

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
VG = 1.88
k1=0.065
k2=0.079
VI=0.05
m1=0.190
m2=0.484
m4=st.sidebar.slider('m4', 0.1940, 0.2690, 0.1940,step= 0.0001)
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
Vmx=st.sidebar.slider('Vmx', 0.0470, 0.0340, 0.047,step= 0.0001)
Kmo=225.59
p2U=0.0331
K=st.sidebar.slider('K', 0.990, 2.30, 2.30,step= 0.0001)
alpha=st.sidebar.slider('alpha', 0.013, 0.05, 0.013,step= 0.0001)
beta=st.sidebar.slider('beta', 0.05, 0.11, 0.11,step= 0.0001)
gamma=0.5
ke1=0.0005
ke2=339
BW=78

#initial
#Y,Ipo,Il,Ip,I1,Id,X,Qsto1,Qsto2,Qgut,Gp,Gt
HEb=0.6 #
Yb=0
Sb=(m6-HEb)/m5 #
Ipob=Sb/gamma
Ipb=0.4*Sb*(1-HEb)/m4
m3b=HEb*m1/(1-HEb)  #
Ilb=(Sb-m4*Ipb)/m3b
I1b=Ipb/VI
Idb=Ipb/VI
Ib=Ipb/VI
Xb=0
Qsto1b=0
Qsto2b=0
Qgutb=0

if st.sidebar.button('Simulate'):
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
    ax.set_title("Blood glucose")
    ax.set_xlim([-288,2880+288])
    ax.set_xticks([0, 720,1440,2160,2880])
    ax.set_xticklabels(["0", "12", "24",'36','48'])
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('mg/dL')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(4,3))
    #plt.plot(np.arange(1440),Qgut)
    plt.plot(np.arange(length),Ip/VI,color="black")
    ax.set_title("Blood insulin")
    ax.set_xlim([-288,2880+288])
    ax.set_xticks([0, 720,1440,2160,2880])
    ax.set_xticklabels(["0", "12", "24",'36','48'])
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('pmol/L')
    st.pyplot(fig)


    fig, ax = plt.subplots(figsize=(4,3))
    plt.plot(np.arange(length),f1(np.arange(length))/1000,color="black")
    ax.set_title("Administered glucose")
    ax.set_xlim([-288,2880+288])
    ax.set_xticks([0, 720,1440,2160,2880])
    ax.set_xticklabels(["0", "12", "24",'36','48'])
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('g')
    st.pyplot(fig)

    dff=pd.DataFrame(sm.tsa.stattools.acf((Gp/VG)[0::5],nlags=30,fft=False))
    st.write("Mean") 
    st.write(np.mean(Gp/VG)) 
    st.write('Std')
    st.write(np.std(Gp/VG)) 
    st.write('AC_Var')
    st.write(dff.iloc[1:].var()[0]) 

else:
    st.sidebar.write("")