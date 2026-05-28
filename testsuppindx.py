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
parameters=oqupy.TempoParameters(dt=dt,epsrel=epsrel,dkmax=500)

tf=15
num_steps=int(tf/dt)
# %%
# %%

# construct an expanded process tensor with an extra null state
# as a way of making PTTEMPO trace caps work 

def expand_op(op):
    return np.pad(op,((0,1),(0,1)),constant_values=0.0)

def compute_pt_extradim(coupling_operator,
                        correlations,
                        start_time,
                        end_time,
                        parameters):
    hsd=coupling_operator.shape[0]
    hsdp=hsd+1
    coupling_pad=expand_op(coupling_operator)
    padbath=oqupy.Bath(coupling_pad,correlations)
    ptt=oqupy.PtTempo(bath=padbath,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters)
    ptt.compute()
    #ptt._backend_instance.update_process_tensor()
    #hsdim=coupling_operator.shape[0]+1
    newtrcap=np.zeros((hsdp,hsdp))
    newtrcap[hsdp-1,hsdp-1]=1
    newtrcap=newtrcap.flatten()
    newtrcap[-1]=1.0
    extpt=ptt.get_process_tensor()
    extpt.compute_caps(trace_square_override=newtrcap)
    normbath=oqupy.Bath(coupling_operator,correlations)
    result=oqupy.SimpleProcessTensor(
        hilbert_space_dimension=hsd,
        dt=extpt.dt,
        transform_in=extpt.transform_in,
        transform_out=extpt.transform_out,
        name=extpt.name,
        description=extpt.description)
    for step in range(len(extpt)):
        result.set_cap_tensor(step,extpt.get_cap_tensor(step))
        mpotensor=extpt.get_mpo_tensor(step,transformed=False)
        a,b,*_=mpotensor.shape
        if mpotensor.ndim==4:
            mpotensor=mpotensor.reshape(a,b,hsdp,hsdp,hsdp,hsdp)
            mpotensor=mpotensor[:,:,:-1,:-1,:-1,:-1]
            mpotensor=mpotensor.reshape(a,b,hsd*hsd,hsd*hsd)
        elif mpotensor.ndim==3:
            mpotensor=mpotensor.reshape(a,b,hsdp,hsdp)
            mpotensor=mpotensor[:,:,:-1,:-1]
            mpotensor=mpotensor.reshape(a,b,hsd*hsd)
        result.set_mpo_tensor(step,mpotensor)      
    result.set_cap_tensor(len(extpt),extpt.get_cap_tensor(len(extpt)))    
    return result,extpt

# %%
testpt,extpt=compute_pt_extradim(op.sigma("z")/2.0,correlations=correlations,
                         start_time=0.0,end_time=tf,parameters=parameters)


splitting=1.0
rhoini=op.spin_dm('x-')
rhoiniex=expand_op(rhoini)
system=oqupy.System(splitting*op.sigma('x')/2)
systemex=oqupy.System(expand_op(splitting*op.sigma('x')/2))

dynamics=oqupy.compute_dynamics(
    process_tensor=testpt,        
    system=system,
    initial_state=rhoini,
    start_time=0)


# %%
t,sx=dynamics.expectations(op.sigma('x'),real=True)
plt.plot(t,sx)
plt.show()
