# %%
import oqupy
import oqupy.operators as op
import scipy.integrate as spi
import numpy as np
import sys

from oqupy.tti_tempo import TTITempo
import matplotlib.pyplot as plt

import tensornetwork as tn

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
#ax.plot(t2,sx2,label='Fancy caps')
#ax.set_ylim(-1,1)
ax.legend()
plt.show()


# %%
# let's try the other way

padcoupling=np.pad(op.sigma("z")/2.0,((0,1),(0,1)),constant_values=0.0)
padbath=oqupy.Bath(padcoupling, correlations)

# %%

ptt3=oqupy.PtTempo(bath=padbath,
            start_time=0,
            end_time=tf,
            parameters=parameters,
            extra_dim=False)
ptt3.compute()
pttempotest3=ptt3.get_process_tensor()
# %%
hampad=np.pad(inisplit*op.sigma('x')/2,((0,1),(0,1)),constant_values=0.0)
system3=oqupy.System(hampad)
rhoinipad=np.pad(rhoini,((0,1),(0,1)),constant_values=0.0)


dynamicspttest3=oqupy.compute_dynamics(
    process_tensor=pttempotest3,        
    system=system3,
    initial_state=rhoinipad,
    start_time=0)


# %%
padsigx=np.pad(op.sigma('x'),((0,1),(0,1)),constant_values=0.0)
t3,sx3=dynamicspttest3.expectations(padsigx,real=True)

plt.plot(t3,sx3)
# %%

# construct an expanded process tensor with an extra null state
# as a way of making PTTEMPO trace caps work 

def expand_op(op):
    return np.pad(op,((0,1),(0,1)),constant_values=0.0))

def compute_pt_extradim(coupling_operator,correlations,
                        start_time,
                        end_time,
                        parameters):
    coupling_pad=expand_op(coupling)
    padbath=oqupy.bath(coupling_pad,correlations)
    ptt=oqupy.PtTempo(bath=padbath,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters)
    ptt.compute() 
    return ptt

# take the process tensor object for such an expanded Hilbert space and compute the caps
def compute_caps_augdof(pt):
    self=pt
    length = len(self)
    caps = [np.array([1.0], dtype=NpDtype)]
    last_cap = tn.Node(caps[-1])
    slicing_cap=np.zeros(self._hs_dim**2)
    slicing_cap[-1]=1.0#/self._hs_dim

    for step in reversed(range(length)):
        trace_square = tn.Node(slicing_cap)
        trace_in = tn.Node(self._trace_in)
        trace_out = tn.Node(self._trace_out)
        ten = tn.Node(self._mpo_tensors[step])

        if len(ten.shape) == 3:
            ten[1] ^ last_cap[0]
            ten[2] ^ trace_square[0]
            new_cap = ten @ last_cap @ trace_square
        else:
            ten[1] ^ last_cap[0]
            ten[2] ^ trace_in[0]
            ten[3] ^ trace_out[0]
            new_cap = ten @ last_cap @ trace_in @ trace_out
        caps.insert(0, new_cap.get_tensor())
        last_cap = new_cap
        
    return caps



# %%
