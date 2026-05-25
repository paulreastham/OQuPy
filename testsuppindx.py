# %%
import oqupy
import oqupy.operators as op
import scipy.integrate as spi
import numpy as np
import sys

from oqupy.tti_tempo import TTITempo
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

pt_parameters = {'epsrel':10**(-7),
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':0.2} #0.25

omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
dt=pt_parameters['dt']

# spectral density (without cutoff)
def j(w):
    return 2*alpha*w

# %%

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(op.sigma("z")/2.0, correlations)
parameters=oqupy.TempoParameters(dt=dt,epsrel=epsrel,dkmax=500)

tf=15
num_steps=int(tf/dt)
# %%
ptt=oqupy.PtTempo(bath=bath,
            start_time=0,
            end_time=tf,
            parameters=parameters,
            extra_dim=False)

ptt.compute()
#%% 
pttempotest=ptt.get_process_tensor()
print('MPO dimensions ',pttempotest.get_mpo_tensor(2).shape)

#%% 

ptt2=oqupy.PtTempo(bath=bath,
            start_time=0,
            end_time=tf,
            parameters=parameters,
            extra_dim=True)
ptt2.compute()
pttempotest2=ptt2.get_process_tensor()
#%%
print('MPO dimensions ',pttempotest2.get_mpo_tensor(2).shape)

for step in range(len(pttempotest2)):
    old=pttempotest2.get_mpo_tensor(step)
    pttempotest2.set_mpo_tensor(step,old[:,:,:-1,:-1])



#%%

inisplit=1.0
# choose initial state to be thermal equilibrium given hamiltonian inisplit*(sigmax/2)
if (temperature>0):
    z=2*np.cosh(inisplit/(2*temperature))
    pupx=np.exp(-inisplit/(2*temperature))/z
    pdnx=np.exp(+inisplit/(2*temperature))/z
else:
    pupx=0
    pdnx=1

rhoini=oqupy.operators.spin_dm('x+')*pupx+oqupy.operators.spin_dm('x-')*pdnx

system=oqupy.System(inisplit*op.sigma('x')/2)

dynamicspttest=oqupy.compute_dynamics(
    process_tensor=pttempotest,        
    system=system,
    initial_state=rhoini,
    start_time=0)

dynamicspttest2=oqupy.compute_dynamics(
    process_tensor=pttempotest2,        
    system=system,
    initial_state=rhoini,
    start_time=0)

t,sx=dynamicspttest.expectations(op.sigma('x'),real=True)
# renormalize traces 
# tracefromtest=dynamicspttest2.states[10].trace()
#dynamicspttest2._states=dynamicspttest2._states/tracefromtest

t2,sx2=dynamicspttest2.expectations(op.sigma('x'),real=True)

fig,ax=plt.subplots(1)
ax.plot(t,sx,'o',label='Standard')
ax.plot(t2,sx2,label='Fancy caps')
#ax.set_ylim(-1,1)
ax.legend()
plt.show()


# %%
