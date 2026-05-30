# %% 
import oqupy.backends.node_array as na

# norm of an mps defined the usual way
def normmps(mps):
    ccmps=mps.copy()
    for i in range(len(ccmps.nodes)):
        ccmps.nodes[i].tensor=ccmps.nodes[i].tensor.conj()
    mps.contract(ccmps)
    return complex(mps.nodes[0].tensor)

# norm of our process tensor defined by applying
# trace caps
def normmpscaps(pt,trace_cap):
    tensors=pt._mpo_tensors
    state=np.array([1.0])
    cap=trace_cap.flatten()
    for i in range(len(tensors)):
        tensor=tensors[i]
        state=np.einsum('i,ijk,k',state,tensor,cap)
    return state


