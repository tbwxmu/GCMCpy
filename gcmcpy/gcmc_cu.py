import cupy as cp
import numpy as np
from .gcmc_h import *
from .gcmc_move_add import *
from .gcmc_move_del import *
from .gcmc_move_trn import *
from .gcmc_move_rot import *
# from ..values import *

__all__=["setup_rng_states",
"runGCMC_cuda"]


# Kernel function to setup random number generator states
@cp.fuse()
def setup_rng_states(states, seed):
    global_threadIdx = cp.blockIdx.x * cp.blockDim.x + cp.threadIdx.x
    cp.random.seed(seed)  # Initialize RNG state
    states[global_threadIdx] = cp.random.get_random_state()  # Save RNG state in shared memory

# Define the kernel code as a string
# setup_rng_states_kernel = cp.RawKernel(r'''
# extern "C" __global__
# void setup_rng_states(curandState *states, unsigned long long seed) {
#     int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
#     curand_init(seed, global_threadIdx, 0, &states[global_threadIdx]);
# }
# ''', 'setup_rng_states')
# Assuming we have `curandState` available and we want to allocate memory for the states

# def setup_rng_states(states, seed, num_blocks, threads_per_block):
#     # Launch the kernel
#     setup_rng_states_kernel((num_blocks,), (threads_per_block,), (states, seed))
# # Example usage
# num_blocks = 128
# threads_per_block = 32
# total_threads = num_blocks * threads_per_block
# # Allocate memory for RNG states (equivalent to curandState array in CUDA)
# rng_states = cp.cuda.memory.alloc(total_threads * cp.dtype('uint32').itemsize)  # Example allocation
# # Set up RNG states
# setup_rng_states(rng_states, 12345, num_blocks, threads_per_block)

# The main GCMC function
def runGCMC_cuda(info, fragmentInfo, residueInfo, atomInfo, grid, ff, moveArray):
    # Ginfo = cp.array(info)
    # GfragmentInfo = cp.array(fragmentInfo)
    # GresidueInfo = cp.array(residueInfo)
    # GatomInfo = cp.array(atomInfo)#gcmc.atomInfo
    
    Ggrid = cp.array(grid)
    Gff = cp.array(ff)
    #gmoveArray
    
    # Determine max configuration based on confBias
    # frag.dtype.names get all the name
    maxConf = max(frag['confBias'] for frag in fragmentInfo)
    
    # Allocate device memory for temporary fragment and info
    GTempFrag = cp.zeros_like(GfragmentInfo[:maxConf])
    GTempInfo = cp.zeros(maxConf, dtype=cp.float32)

    # Allocate and initialize temporary atom info on the host
    # TempInfo = np.zeros(maxConf, dtype=object)
    TempInfo = cp.zeros(maxConf, dtype=object)
    for i in range(maxConf):
        TempInfo[i] = Atom(type=0)

    # GTempInfo[:] = cp.array(TempInfo)  # Copy to device

    # GTempInfo = np.zeros(maxConf, dtype=Atom_dtype)

    # Setup random number generator states
    # d_rng_states = cp.zeros((maxConf, cp.numThreadsPerBlock), dtype=cp.uint32)
    # setup_rng_states(d_rng_states, info.seed)
    d_rng_states=cp.random.get_random_state()

    # Calculate step threshold for progress printing (optional)
    step_threshold = info.mcsteps // 20

    # Monte Carlo loop, NOTE useful code frag type and move types into numbers with steps, should reuse thos codes
    for stepi in range(info.mcsteps):
        moveFragType = moveArray[stepi] // 4
        moveMoveType = moveArray[stepi] % 4

        # Perform move based on the moveMoveType
        if moveMoveType == 0:  # Insert
            accepted = move_add(info, Ginfo, fragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, TempInfo, GTempInfo, d_rng_states)
            if accepted:
                print('added or insert fragment')#also use this add to init GCMC frag around protein
        elif moveMoveType == 1:  # Delete
            accepted = move_del(info, Ginfo, fragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, TempInfo, GTempInfo, d_rng_states)
        elif moveMoveType == 2:  # Translate
            accepted = move_trn(info, Ginfo, fragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, TempInfo, GTempInfo, d_rng_states)
        elif moveMoveType == 3:  # Rotate
            accepted = move_rot(info, Ginfo, fragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, TempInfo, GTempInfo, d_rng_states)

        # Progress indicator (optional)
        if step_threshold != 0 and stepi % step_threshold == 0:
            print(".", end="", flush=True)

    print("\n")

    # Synchronize device
    cp.cuda.Device().synchronize()

    # Copy data back from device to host
    fragmentInfo = cp.asnumpy(GfragmentInfo)
    residueInfo = cp.asnumpy(GresidueInfo)
    atomInfo = cp.asnumpy(GatomInfo)

    # Free device memory
    del Ginfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, GTempFrag, GTempInfo, d_rng_states

    # Return the updated host arrays
    return fragmentInfo, residueInfo, atomInfo
