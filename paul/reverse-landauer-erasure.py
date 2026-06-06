# %%
"""
Example of some of the capabilities of the code in the integrationtcd branch.

- Heat calculations using TTI-TEMPO, PT-TEMPO, and TEMPO and
- Time-dependent system-environment coupling in PT-TEMPO and TEMPO
- Analytical expressions for correlation functions with Ohmic spectral densities.

To avoid svd instabilities, I had to use OQuPy version that used INTEGRATE_EPSREL
and increase 'INTEGRATE_EPSREL' defined in oqupy/config.py to 2**(-34) 

"""

import sys
sys.path.insert(0,'/Users/easthamp/OQuPy')

import oqupy
import oqupy.operators as op
import scipy.integrate as spi
import numpy as np

from oqupy.tti_tempo import TTITempo
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
session = WolframLanguageSession('/Applications/Wolfram.app/Contents/MacOS/WolframKernel')
# %%

lbladqoft=session.evaluate("""Get["thermodynamics.wls"]""")

pt_parameters = {'epsrel':10**(-7),
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.1,
                 'dt':0.2} #0.25

omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
dt=pt_parameters['dt']
# we will consider dynamics up to tf
# tfswitch is the end time of the smooth switching protocol
# %%
tf=400
tfswitch=20
# spectral density
def j(w):
    return 2*alpha*w

# TTI-TEMPO process tensor for usual dynamics
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(op.sigma("z")/2.0, correlations)
parameters=oqupy.TempoParameters(dt=dt,epsrel=epsrel,dkmax=500)
tti=TTITempo(bath,start_time=0.0,parameters=parameters)
ttipt=tti.get_process_tensor()
# Corresponding object with heat markers
u=0.01
correlationscf=oqupy.counting_bath_correlations.CustomCountingSD_analytical(j_function=j,cutoff=omega_cutoff,u=u,
                                                 cutoff_type='exponential',temperature=temperature)
bathcf = oqupy.Bath(op.sigma("z")/2.0, correlationscf)
ttiheats=TTITempo(bathcf,start_time=0.0,parameters=parameters)
ttiheatpt=ttiheats.get_process_tensor()
# hamiltonian with a linear ramp from inisplit to 0 over time tf
# initial state in thermal equilibrium with bath
# %%
num_steps=int(tf/dt)
inisplit=1.0
# choose initial state to be thermal equilibrium given hamiltonian inisplit*(sigmax/2)
z=2*np.cosh(inisplit/(2*temperature))
pupx=np.exp(-inisplit/(2*temperature))/z
pdnx=np.exp(+inisplit/(2*temperature))/z
rhoini=oqupy.operators.spin_dm('x+')*pupx+oqupy.operators.spin_dm('x-')*pdnx
system=oqupy.System(inisplit*op.sigma('x')/2)
# parameterizedsystem with time-dependent splitting spl

# %%
def hamiltonian(spl):
     return op.sigma('x')*spl/2

system = oqupy.ParameterizedSystem(hamiltonian)

halfsteptimes=list((dt/4)+np.arange(num_steps*2)*dt/2)

# %%
tdsplit=session.evaluate(wl.Global.optimaltdlensol(temperature,inisplit,tf,omega_cutoff,halfsteptimes))
# session.terminate()
# %%

#tdsplit=np.ones(num_steps*2) # simple test
tdsplit=np.array(tdsplit)
tdsplit=tdsplit.reshape((num_steps*2,1))

dynamics=oqupy.state_gradient(system,rhoini,[],[ttipt],tdsplit,start_time=0.0,num_steps=num_steps,dynamics_only=True)

dynamicscf = oqupy.state_gradient(system,rhoini,[],[ttiheatpt],tdsplit,start_time=0.0,num_steps=num_steps,dynamics_only=True)

t,s_x=dynamics.expectations(op.sigma('x'),real=True)
heats=dynamicscf.states.trace(axis1=1,axis2=2).imag/u
# %%
fig,axs=plt.subplots(3)

axs[0].plot(t,heats)
axs[0].plot(t,lbladqoft)
axs[0].set_xlabel(r'$t$')
axs[0].set_ylabel(r'$\langle Q \rangle$')
axs[1].plot(t,s_x)
axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$\langle \sigma_x \rangle$')
axs[1].set_ylim(-1.0,0.0)
axs[2].plot(halfsteptimes,tdsplit)
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$\omega_q(t)$')
plt.legend()
plt.show()


# %%
# smooth switching function
lamconst=2
def smswitch(t):
    if t<=tfswitch:
        return (0.1+0.9*(t**lamconst/(t**(lamconst)+(tfswitch-t)**(lamconst))))
    elif t>=(tf-tfswitch):
        return 0.1+0.9*((tf-t)**lamconst/((tf-t)**lamconst+(tf-t-tfswitch)**(lamconst)))
    else:
        return 1.0
smv=np.vectorize(smswitch)
num_steps=int(tf/dt)
alpha_tramp=smv(np.arange(num_steps)*dt)

# %%

ptt=oqupy.PtTempo(bath=bath,
        start_time=0,
        end_time=tf,
        parameters=parameters,
        alpha_t=alpha_tramp)

ptswitch=ptt.get_process_tensor()

ptth=oqupy.PtTempo(bath=bathcf,
                    start_time=0,
                    end_time=tf,
                    parameters=parameters,
                    alpha_t=alpha_tramp,
                    extra_dim=True)

heatptswitch=ptth.get_process_tensor()

# %%

dynamicssw=oqupy.state_gradient(system,rhoini,[],[ptswitch],tdsplit,start_time=0.0,num_steps=num_steps,dynamics_only=True)

dynamicsswcf = oqupy.state_gradient(system,rhoini,[],[heatptswitch],tdsplit,start_time=0.0,num_steps=num_steps,dynamics_only=True)


tsw,sxsw=dynamicssw.expectations(op.sigma('x'),real=True)
heatssw=dynamicsswcf.states.trace(axis1=1,axis2=2).imag/u

def vneumannentropy(rho):
    probs=np.linalg.eigvalsh(rho)
    return -np.sum(probs*(np.log(probs)))

# %%
# landauer limit
ll=temperature*(vneumannentropy(dynamicssw.states[-1])-vneumannentropy(rhoini))
print('Landauer limit ',ll)
print('Achieved Q ',-heatssw[-1])
print('Percentage of Landauer achieved ',-100*heatssw[-1]/ll)
print('Percentage of Landauer achieved (Lindblad)',-100*lbladqoft[-1]/ll)

# %%
fig,axs=plt.subplots(1)
axs=[axs]

axs[0].plot(t,heatssw,label='Switched coupling')
axs[0].plot(t,heats,label='Constant coupling')
axs[0].plot(t,lbladqoft,label='Lindblad')
axs[0].set_xlabel(r'$t$')
axs[0].set_ylabel(r'$\langle Q \rangle$')
axs[0].axhline(-ll,ls='--',label='L. limit')
axs[0].legend()
plt.show()

fig,axs=plt.subplots(3)

axs[0].plot(t,heatssw,label='Switched coupling')
axs[0].plot(t,heats,label='Constant coupling')
axs[0].plot(t,lbladqoft,label='Lindblad')
axs[0].set_xlabel(r'$t$')
axs[0].set_ylabel(r'$\langle Q \rangle$')
axs[0].axhline(-ll,ls='--',label='L. limit')

axs[1].plot(t,sxsw)
axs[1].plot(t,s_x)
axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$\langle \sigma_x \rangle$')
axs[1].set_ylim(-1.0,0.0)
axs[2].plot(halfsteptimes,tdsplit)
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$\omega_q(t)$')

plt.show()

# %%
# %%

# %%
