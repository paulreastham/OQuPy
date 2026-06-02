"""
Example of some of the capabilities of the code in the integrationtcd branch.

- Heat calculations using TTI-TEMPO, PT-TEMPO, and TEMPO and
- Time-dependent system-environment coupling in PT-TEMPO and TEMPO
- Analytical expressions for correlation functions with Ohmic spectral densities.

"""

import sys
sys.path.insert(0,'..')

import oqupy
import oqupy.operators as op
import scipy.integrate as spi
import numpy as np

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

# we will consider dynamics up to tf
# tfswitch is the end time of the smooth switching protocol

tf=50
tfswitch=20

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

# %%
u=0.01
correlationscf=oqupy.counting_bath_correlations.CustomCountingSD_analytical(j_function=j,cutoff=omega_cutoff,u=u,
                                                 cutoff_type='exponential',temperature=temperature)

bathcf = oqupy.Bath(op.sigma("z")/2.0, correlationscf)

# hamiltonian with a constant qubit splitting and 
# initial state in thermal equilibrium with bath
# %%
inisplit=1.0
# choose initial state to be thermal equilibrium given hamiltonian inisplit*(sigmax/2)
z=2*np.cosh(inisplit/(2*temperature))
pupx=np.exp(-inisplit/(2*temperature))/z
pdnx=np.exp(+inisplit/(2*temperature))/z

rhoini=oqupy.operators.spin_dm('x+')*pupx+oqupy.operators.spin_dm('x-')*pdnx

system=oqupy.System(inisplit*op.sigma('x')/2)

# %%
# smooth switching function
lamconst=2
def smswitch(t):
    if t<=tfswitch:
        return (0.1+0.9*(t**lamconst/(t**(lamconst)+(tfswitch-t)**(lamconst))))
    else:
        return 1.0
smv=np.vectorize(smswitch)
num_steps=int(tf/dt)
alpha_tramp=smv(np.arange(num_steps)*dt)

# %%
pttempotest=oqupy.pt_tempo_compute(bath=bath,
                             start_time=0,
                             end_time=50,
                             parameters=parameters)
pttempotestswitch=oqupy.pt_tempo_compute(bath=bath,
                             start_time=0,
                             end_time=50,
                             parameters=parameters,
                             alpha_t=alpha_tramp)


tti=TTITempo(bath,start_time=0.0,parameters=parameters)
ttitest=tti.get_process_tensor()

dynamicspttempo=oqupy.compute_dynamics(
    process_tensor=pttempotest,        
    system=system,
    initial_state=rhoini,
    start_time=0)

dynamicspttemposwitch=oqupy.compute_dynamics(
    process_tensor=pttempotestswitch,        
    system=system,
    initial_state=rhoini,
    start_time=0)

dynamicsttitempo=oqupy.compute_dynamics(
    process_tensor=ttitest,        
    system=system,
    initial_state=rhoini,
    start_time=0,
    num_steps=num_steps)


t,sx=dynamicspttempo.expectations(op.sigma('x'),real=True)
t2,sx2=dynamicspttemposwitch.expectations(op.sigma('x'),real=True)
t3,sx3=dynamicsttitempo.expectations(op.sigma('x'),real=True)

poldp=spi.quad(lambda x: correlations.spectral_density(x)/(4.0*(x+inisplit)**2),0,30)[0]
polpred=-np.exp(-2*poldp)

fig,ax=plt.subplots(1)
ax.plot(t,sx,'o',label='PT-TEMPO, time-indep coupling')
ax.plot(t2,sx2,label='PT-TEMPO, switched coupling')
ax.plot(t3,sx3,label='TTI-TEMPO, time-indep coupling')
ax.axhline(polpred,label='Polaron result')
ax.legend()
plt.show()

# %%
# Heat dynamics
# To get correct heats from PT-TEMPO at intermediate times
# need the extra_dim=True to get the correct caps.

pttempotestheat=oqupy.pt_tempo_compute(bath=bathcf,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             extra_dim=True)

pttempotestheatswitch=oqupy.pt_tempo_compute(bath=bathcf,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             alpha_t=alpha_tramp,
                             extra_dim=True)

ttiheats=TTITempo(bathcf,start_time=0.0,parameters=parameters)
ttiheattest=ttiheats.get_process_tensor()

heatdynamicstti=oqupy.compute_dynamics(
    process_tensor=ttiheattest,        
    system=system,
    initial_state=rhoini,
    start_time=0,
    num_steps=num_steps)

heatdynamicspttest=oqupy.compute_dynamics(
    process_tensor=pttempotestheat,        
    system=system,
    initial_state=rhoini,
    start_time=0)
heatdynamicspttestswitch=oqupy.compute_dynamics(
    process_tensor=pttempotestheatswitch,        
    system=system,
    initial_state=rhoini,
    start_time=0)
# %%
# Extract heats and plot

heatspt=heatdynamicspttest.states.trace(axis1=1,axis2=2).imag/u
heatsswitch=heatdynamicspttestswitch.states.trace(axis1=1,axis2=2).imag/u
heatstti=heatdynamicstti.states.trace(axis1=1,axis2=2).imag/u

fig,ax=plt.subplots(1)
polht=spi.quad(lambda x: correlations.spectral_density(x)*x/(4.0*(x+inisplit)**2),0,30)[0] # heat (environment energy) in polaron state
ax.plot(heatdynamicspttest._times,heatspt,label='PT-TEMPO, time-indep coupling')
ax.plot(heatdynamicspttestswitch._times,heatsswitch,label='PT-TEMPO, switched coupling')
ax.plot(heatdynamicstti._times,heatstti,'x',label='TTI-TEMPO, time-indep coupling')
ax.axhline(polht,label='Polaron result')
ax.set_ylim(0,0.05)
ax.legend()
plt.show()

# %%
runtempo=True
if runtempo:
    temporesramp=oqupy.tempo_compute(system=system,
                                bath=bath,
                                initial_state=rhoini,
                                start_time=0,
                                end_time=10, #tf,
                                alpha_t=alpha_tramp,
                                parameters=parameters)
    t4,sx4=temporesramp.expectations(op.sigma('x'),real=True)

    fig,ax=plt.subplots(1)
    ax.plot(t2,sx2,'o',label='PT-TEMPO, switched coupling')
    ax.plot(t4,sx4,label='TEMPO, switched coupling')
    ax.legend()
    plt.show()

# %%
runtempoheats=True
if runtempoheats:
    temporesheatramp=oqupy.tempo_compute(system=system,bath=bathcf,
                                initial_state=rhoini,
                                start_time=0,
                                end_time=10, #tf,
                                parameters=parameters,
                                alpha_t=alpha_tramp)

    tempoheatsramp=temporesheatramp.states.trace(axis1=1,axis2=2).imag/u

    fig,ax=plt.subplots(1)
    ax.plot(heatdynamicspttestswitch._times,heatsswitch,label='PT-TEMPO, switched coupling')
    ax.plot(temporesheatramp._times,tempoheatsramp,label='TEMPO, switched coupling')
    ax.set_ylim(0,0.05)
    ax.legend()
    plt.show()

# %%
