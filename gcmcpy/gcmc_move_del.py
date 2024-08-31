import cupy as cp
import numpy as np
from .gcmc_energy import *
from .gcmc_h import *
# from .gcmc_cu import *

__all__ = ['Gmove_del', 'GupdateDel', 'move_del']
# Function 1: Gmove_del
def Gmove_del(Ginfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, GTempInfo, d_rng_states):
    # Shared memory simulation
    SharedInfo = Ginfo[0]
    SharedFragmentInfo = GfragmentInfo[moveFragType]

    threadId = cp.cuda.threadIdx.x + cp.cuda.blockIdx.x * cp.cuda.blockDim.x
    rng_states = d_rng_states[threadId]#TODO check this rng_states

    tid = cp.cuda.threadIdx.x

    if tid == 0:
        SharedInfo = Ginfo[0]

    if tid == 1:
        SharedFragmentInfo = GfragmentInfo[moveFragType]
        SharedFragmentInfo['startRes'] = GTempInfo[cp.cuda.blockIdx.x]['type'] + GfragmentInfo[moveFragType]['startRes']
        GTempInfo[cp.cuda.blockIdx.x]['position'][:] = GresidueInfo[SharedFragmentInfo['startRes']]['position'][:]

    cp.cuda.syncthreads()

    for i in range(tid, SharedFragmentInfo['num_atoms'], cp.cuda.blockDim.x):
        atom_index = GresidueInfo[SharedFragmentInfo['startRes']]['atomStart'] + i
        SharedFragmentInfo['atoms'][i]['position'][:] = GatomInfo[atom_index]['position'][:]

    cp.cuda.syncthreads()

    if tid == 0:
        GTempFrag[cp.cuda.blockIdx.x] = SharedFragmentInfo

    calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo[cp.cuda.blockIdx.x])

# Function 2: GupdateDel
def GupdateDel(GfragmentInfo, GresidueInfo, GatomInfo, GTempFrag, GTempInfo, moveFragType, totalNum, conf_index):
    tid = cp.cuda.threadIdx.x

    if tid == 0:
        GfragmentInfo[moveFragType]['totalNum'] = totalNum

    if totalNum == 0:
        return

    if tid == 1:
        bidStartRes = GfragmentInfo[moveFragType]['startRes'] + GTempInfo[conf_index]['type']
        bidAtomNum = GresidueInfo[bidStartRes]['atomNum']
        bidStartAtom = GresidueInfo[bidStartRes]['atomStart']
        bidEndRes = GfragmentInfo[moveFragType]['startRes'] + totalNum
        bidEndAtom = GresidueInfo[bidEndRes]['atomStart']

    cp.cuda.syncthreads()

    if tid == 0:
        GresidueInfo[bidStartRes]['position'][:] = GresidueInfo[bidEndRes]['position'][:]

    for i in range(tid, bidAtomNum, cp.cuda.blockDim.x):
        GatomInfo[bidStartAtom + i]['position'][:] = GatomInfo[bidEndAtom + i]['position'][:]

# Function 3: move_del
def move_del(info, Ginfo, fragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, TempInfo, GTempInfo, d_rng_states):
    nBlock = 1

    if nBlock == 0:
        return False

    nums = np.arange(fragmentInfo[moveFragType]['totalNum'])
    np.random.shuffle(nums)

    for i in range(nBlock):
        TempInfo[i]['type'] = nums[i]

    cp.copyto(GTempInfo[:nBlock], cp.array(TempInfo[:nBlock]))

    Gmove_del(Ginfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, GTempInfo, d_rng_states)

    cp.copyto(TempInfo[:nBlock], GTempInfo[:nBlock])

    conf_p = np.zeros(nBlock)
    conf_index = 0
    energy_max = TempInfo[0]['charge']
    sum_p = 0.0

    for i in range(1, nBlock):
        if TempInfo[i]['charge'] > energy_max:
            energy_max = TempInfo[i]['charge']

    for i in range(nBlock):
        conf_p[i] = np.exp(beta * (TempInfo[i]['charge'] - energy_max))
        sum_p += conf_p[i]

    if sum_p != 0:
        conf_p_sum = 0
        for i in range(nBlock):
            conf_p_sum += conf_p[i] / sum_p
            conf_p[i] = conf_p_sum

        ran = np.random.rand()
        for i in range(nBlock):
            conf_index = i
            if ran < conf_p[i]:
                break

    period = info['cryst']
    nbar = period[0] * period[1] * period[2] * fragmentInfo[moveFragType]['conc'] * MOLES_TO_MOLECULES
    B = beta * fragmentInfo[moveFragType]['muex'] + np.log(nbar)

    diff = -TempInfo[conf_index]['charge']
    conf_p_value = np.exp(beta * (TempInfo[conf_index]['charge'] - energy_max)) / sum_p
    fn = 1 / (nBlock * conf_p_value)
    n = fragmentInfo[moveFragType]['totalNum']
    p = min(1, n / fn * np.exp(-B - beta * diff))
    ran = np.random.rand()

    if ran < p:
        fragmentInfo[moveFragType]['totalNum'] -= 1
        GupdateDel(GfragmentInfo, GresidueInfo, GatomInfo, GTempFrag, GTempInfo, moveFragType, fragmentInfo[moveFragType]['totalNum'], conf_index)
        return True
    else:
        return False

# # Placeholder for the calcEnergy function
# def calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo):
#     # Implement the energy calculation logic here
#     pass
