# %%
import oqupy
import oqupy.operators as op
import scipy.integrate as spi
import numpy as np

from oqupy.pt_tempo import PtTempoCounting
from oqupy.pt_tempo import pt_tempo_counting_compute
from oqupy.tti_tempo import TTITempo
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize,Bounds

original_heatmarkers=False # set to True if running in PTTempoTimeDepCoupling

if original_heatmarkers:
    # from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
    # from oqupy.process_tensor import TTInvariantProcessTensor
    from oqupy.tti_tempo import TTITempoCounting


pt_parameters = {'epsrel':10**(-7),
                 'alpha':0.1,
                 'omega_cutoff':1,                 
                 'temp':0.131,
                 'dt':0.2} #0.25

omega_cutoff = pt_parameters['omega_cutoff']
alpha = pt_parameters['alpha']
temperature = pt_parameters['temp']
epsrel = pt_parameters['epsrel']
# dt = 1./omega_cutoff/np.sqrt(3)
dt=pt_parameters['dt']

tcut=27

Rho_0=oqupy.operators.spin_dm('x+')

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
correlationscf=oqupy.bath_correlations.CustomCountingSD_analytical(j_function=j,cutoff=omega_cutoff,u=u,
                                                 cutoff_type='exponential',temperature=temperature)

bathcf = oqupy.Bath(op.sigma("z")/2.0, correlationscf)

# %%
# generate results for a linear ramp using Eoin's TTI-heat markers code
# create TTI process tensors 

pt_var=TTITempo(bath,start_time=0.0,parameters=parameters)
process_tensor_var = pt_var.get_process_tensor()

if original_heatmarkers:
    pt_varcf=TTITempoCounting(bathcf,start_time=0.0,parameters=parameters)
else: # new version never converges
    pt_varcf=TTITempo(bathcf,start_time=0.0,parameters=parameters)

process_tensor_varcf = pt_varcf.get_process_tensor()

# %%
# set up a linear ramp starting in thermal equilibrium at splitting inisplit
inisplit=1.0
# choose initial state to be thermal equilibrium given hamiltonian inisplit*(sigmax/2)
z=2*np.cosh(inisplit/(2*temperature))
pupx=np.exp(-inisplit/(2*temperature))/z
pdnx=np.exp(+inisplit/(2*temperature))/z

rhoini=oqupy.operators.spin_dm('x+')*pupx+oqupy.operators.spin_dm('x-')*pdnx

for t_prot in [300]:

   fig,axs=plt.subplots(3)

   num_steps=int(t_prot/dt)

   def hx(t):
      return inisplit*(1-t/t_prot)
   
   def hamiltonian_t(t):
      return op.sigma('x')*hx(t)/2

   system = oqupy.TimeDependentSystem(hamiltonian_t)

   # compute dynamics

   dynamics=oqupy.compute_dynamics(
   process_tensor=[process_tensor_var],        
   system=system,
   initial_state=rhoini,
   start_time=0,
   num_steps=num_steps)
   t, s_x = dynamics.expectations(op.sigma('x'), real=True)

   # compute heats

   dynamicscf = oqupy.compute_dynamics(
   process_tensor=[process_tensor_varcf],        
   system=system,
   initial_state=rhoini,
   start_time=0,
   num_steps=num_steps)

   heats=dynamicscf.states.trace(axis1=1,axis2=2).imag/u

   axs[0].plot(t,heats,label=t_prot)
   axs[0].set_xlabel(r'$t$')
   axs[0].set_ylabel(r'$\langle Q \rangle$')
   axs[1].plot(t,s_x)
   axs[1].set_xlabel(r'$t$')
   axs[1].set_ylabel(r'$\langle \sigma_x \rangle$')
   axs[2].plot(t,[hx(tme) for tme in t])
   plt.legend()
   plt.show()


# %%
def vneumannentropy(rho):
    probs=np.linalg.eigvalsh(rho)
    return -np.sum(probs*(np.log(probs)))

# %%
# landauer limit
ll=temperature*(vneumannentropy(dynamics.states[-1])-vneumannentropy(rhoini))
print('Landauer limit ',ll)
print('Achieved Q ',-heats[-1])
print('Percentage of Landauer achieved ',-100*heats[-1]/ll)
# dynamics using a switched coupling using a PT
# with a constant Hamiltonian
# in comparison to a constant coupling
# %%
tf=50
num_steps=int(tf/dt)
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

fig,ax=plt.subplots(1)
ax.plot(t,sx,'o')
ax.plot(t2,sx2)

# %%
# Let's look at the heats at the end of this process. 

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
print(finalheatnoswit,finalheatswit)

# %%
# Tempo version of the dynamics with the switch
if True:
    temporesnoramp=oqupy.tempo_compute(system=system,bath=bath,
                                initial_state=rhoini,
                                start_time=0,
                                end_time=50,
                                parameters=parameters,
                                alpha_t=np.ones(num_steps))
    
    # %%
    
    temporesramp=oqupy.tempo_compute(system=system,
                                bath=bath,
                                initial_state=rhoini,
                                start_time=0,
                                end_time=50,
                                alpha_t=alpha_tramp,
                                parameters=parameters)
    
    # %%
    t,sx=temporesnoramp.expectations(op.sigma('x'))
    t2,sx2=temporesramp.expectations(op.sigma('x'))
    poldp=spi.quad(lambda x: correlations.spectral_density(x)/(4.0*(x**2+1.0)),0,30)[0] # heat transferred in polaron
    polpred=-np.exp(-2*poldp)
    
    fig,ax=plt.subplots(1)
    ax.plot(t,sx,'o',label='Constant coupling')
    ax.plot(t2,sx2,label='Ramp')
    ax.legend()
    ax.axhline(polpred) 
    ax.set_ylim(-1,-0.9)
    ax2=ax.twinx()
    ax2.set_ylim(0,1)
    ax2.plot(t,alpha_tramp,label='Coupling function')
    ax2.legend()

# %%
# tempo version of the heats with the switch (slow!)

temporesheatnoramp=oqupy.tempo_compute(system=system,bath=bathcf,
                             initial_state=rhoini,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             alpha_t=np.ones(num_steps))

# %%

temporesheatramp=oqupy.tempo_compute(system=system,bath=bathcf,
                             initial_state=rhoini,
                             start_time=0,
                             end_time=tf,
                             parameters=parameters,
                             alpha_t=alpha_tramp)

# %%

heats=temporesheatnoramp.states.trace(axis1=1,axis2=2).imag/u
heatsramp=temporesheatramp.states.trace(axis1=1,axis2=2).imag/u

fig,ax=plt.subplots(1)
polht=spi.quad(lambda x: correlations.spectral_density(x)*x/(4.0*(x**2+1.0)),0,10)[0] # heat transferred in polaron
ax.plot(temporesheatnoramp._times,heats,label='Instant switch')
ax.plot(temporesheatramp._times,heatsramp,label='Smooth switch')
ax.axhline(polht)
#ax.axhline(2*polht)
ax.set_ylim(0,0.05)
ax.legend()
plt.show()



# %%
