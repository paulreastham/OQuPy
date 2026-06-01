# %%
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

# time-dependent system-environment coupling
# compute the dynamics of a qubit with constant splitting, 
# for (i)time-independent system-environment coupling, and (ii)a smooth switch on
# of the system-environment coupling


# %%
# set up a linear ramp starting in thermal equilibrium at splitting inisplit
inisplit=1.0
# choose initial state to be thermal equilibrium given hamiltonian inisplit*(sigmax/2)
z=2*np.cosh(inisplit/(2*temperature))
pupx=np.exp(-inisplit/(2*temperature))/z
pdnx=np.exp(+inisplit/(2*temperature))/z

rhoini=oqupy.operators.spin_dm('x+')*pupx+oqupy.operators.spin_dm('x-')*pdnx


rhoini=oqupy.operators.spin_dm('x+')*pupx+oqupy.operators.spin_dm('x-')*pdnx

system=oqupy.System(inisplit*op.sigma('x')/2)

# %%
# smooth switching function
lamconst=2
def smswitch(t):
    if t<=tf:
        return (0.1+0.9*(t**lamconst/(t**(lamconst)+(tf-t)**(lamconst))))
    else:
        return 1.0
smv=np.vectorize(smswitch)
alpha_tramp=smv(np.arange(num_steps)*dt)

# %%
pttempotest=oqupy.pt_tempo_compute(bath=bath,
                             start_time=0,
                             end_time=50,
                             parameters=parameters,
                             alpha_t=np.ones(num_steps))
pttempotestswitch=oqupy.pt_tempo_compute(bath=bath,
                             start_time=0,
                             end_time=50,
                             parameters=parameters,
                             alpha_t=alpha_tramp)

dynamicspttest=oqupy.compute_dynamics(
    process_tensor=pttempotest,        
    system=system,
    initial_state=rhoini,
    start_time=0)

dynamicspttestswitch=oqupy.compute_dynamics(
    process_tensor=pttempotestswitch,        
    system=system,
    initial_state=rhoini,
    start_time=0)

t,sx=dynamicspttest.expectations(op.sigma('x'),real=True)
t2,sx2=dynamicspttestswitch.expectations(op.sigma('x'),real=True)

poldp=spi.quad(lambda x: correlations.spectral_density(x)/(4.0*(x+inisplit)**2),0,30)[0]
polpred=-np.exp(-2*poldp)

fig,ax=plt.subplots(1)
ax.plot(t,sx,'o',label='Constant coupling')
ax.plot(t2,sx2,label='Smooth switch')
ax.axhline(polpred,label='Polaron result')
ax.legend()
plt.show()

# %%
# Comparison of the heat transfers at the final time 
# in the two cases above

pttempotestheat=oqupy.pt_tempo_compute(bath=bathcf,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             alpha_t=np.ones(num_steps))

pttempotestheatswitch=oqupy.pt_tempo_compute(bath=bathcf,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             alpha_t=alpha_tramp)

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
finalheatnoswit=heatdynamicspttest._states[-1].trace().imag/u
finalheatswit=heatdynamicspttestswitch._states[-1].trace().imag/u

polht=spi.quad(lambda x: correlations.spectral_density(x)*x/(4.0*(x+1.0)**2),0,30)[0]

print("Final heats: constant coupling = ", finalheatnoswit, " smooth switch = ", finalheatswit)
print("Polaron = ",polht)


heats=heatdynamicspttest.states.trace(axis1=1,axis2=2).imag/u
heatsramp=heatdynamicspttestswitch.states.trace(axis1=1,axis2=2).imag/u

fig,ax=plt.subplots(1)
polht=spi.quad(lambda x: correlations.spectral_density(x)*x/(4.0*(x+1.0)**2),0,30)[0] # heat transferred in polaron
ax.plot(heatdynamicspttest._times,heats,label='Instant switch')
ax.plot(heatdynamicspttestswitch._times,heatsramp,label='Smooth switch')
ax.axhline(polht)
#ax.axhline(2*polht)
ax.set_ylim(0,0.05)
ax.legend()
plt.show()

