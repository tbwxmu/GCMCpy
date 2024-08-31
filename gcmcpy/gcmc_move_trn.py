import cupy as cp
import numpy as np
import random
from collections import defaultdict
from .gcmc_energy import *
from .gcmc_h import *
# from .gcmc_cu import *



__all__ = ['Gmove_trn', 'GupdateTrn', 'move_trn']

# Function 1: Gmove_trn
def Gmove_trn(Ginfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, GTempInfo, d_rng_states):
    # Shared memory simulation
    SharedInfo = Ginfo[0]
    SharedFragmentInfo = GfragmentInfo[moveFragType]

    energyTemp = cp.zeros(1)
    randomR = cp.zeros(3)
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

    calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo[cp.cuda.blockIdx.x])
    cp.cuda.syncthreads()

    if tid == 3:
        energyTemp = GTempInfo[cp.cuda.blockIdx.x]['charge']

    if tid < 3:
        randomR[tid] = cp.random.uniform(0, 1) * SharedInfo['grid_dx']
        if GTempInfo[cp.cuda.blockIdx.x]['position'][tid] + randomR[tid] > SharedInfo['cryst'][tid] + SharedInfo['startxyz'][tid]:
            randomR[tid] -= SharedInfo['cryst'][tid]
        if GTempInfo[cp.cuda.blockIdx.x]['position'][tid] + randomR[tid] < SharedInfo['startxyz'][tid]:
            randomR[tid] += SharedInfo['cryst'][tid]
        GTempInfo[cp.cuda.blockIdx.x]['position'][tid] += randomR[tid]

    cp.cuda.syncthreads()

    for i in range(tid, SharedFragmentInfo['num_atoms'], cp.cuda.blockDim.x):
        SharedFragmentInfo['atoms'][i]['position'][0] += randomR[0]
        SharedFragmentInfo['atoms'][i]['position'][1] += randomR[1]
        SharedFragmentInfo['atoms'][i]['position'][2] += randomR[2]

    cp.cuda.syncthreads()

    calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo[cp.cuda.blockIdx.x])
    cp.cuda.syncthreads()

    if tid == 3:
        GTempInfo[cp.cuda.blockIdx.x]['charge'] -= energyTemp

    if tid == 0:
        GTempFrag[cp.cuda.blockIdx.x] = SharedFragmentInfo

# Function 2: GupdateTrn
def GupdateTrn(GfragmentInfo, GresidueInfo, GatomInfo, GTempFrag, GTempInfo, moveFragType):
    tid = cp.cuda.threadIdx.x
    bid = cp.cuda.blockIdx.x

    bidStartRes = cp.zeros(1, dtype=cp.int32)
    bidStartAtom = cp.zeros(1, dtype=cp.int32)
    bidAtomNum = cp.zeros(1, dtype=cp.int32)

    if GTempInfo[bid]['type'] == -1:
        return

    if tid == 1:
        bidStartRes = GfragmentInfo[moveFragType]['startRes'] + GTempInfo[bid]['type']

    cp.cuda.syncthreads()

    if tid == 0:
        bidAtomNum = GresidueInfo[bidStartRes]['atomNum']
        bidStartAtom = GresidueInfo[bidStartRes]['atomStart']
        GresidueInfo[bidStartRes]['position'][:] = GTempInfo[bid]['position'][:]

    cp.cuda.syncthreads()

    for i in range(tid, bidAtomNum, cp.cuda.blockDim.x):
        GatomInfo[bidStartAtom + i]['position'][0] = GTempFrag[bid]['atoms'][i]['position'][0]
        GatomInfo[bidStartAtom + i]['position'][1] = GTempFrag[bid]['atoms'][i]['position'][1]
        GatomInfo[bidStartAtom + i]['position'][2] = GTempFrag[bid]['atoms'][i]['position'][2]

# Function 3: move_trn
def move_trn(infoHost, infoDevice, fragmentInfoHost, fragmentInfoDevice, residueInfoDevice, atomInfoDevice, gridDevice, ffDevice, moveFragType, tempFragDevice, tempInfoHost, tempInfoDevice, rngStatesDevice):
    numBlocks = min(fragmentInfoHost[moveFragType]['confBias'], fragmentInfoHost[moveFragType]['totalNum'])

    if numBlocks == 0:
        return False

    nums = np.arange(fragmentInfoHost[moveFragType]['totalNum'])
    np.random.shuffle(nums)

    for i in range(numBlocks):
        tempInfoHost[i]['type'] = nums[i]

    cp.copyto(tempInfoDevice[:numBlocks], cp.array(tempInfoHost[:numBlocks]))

    Gmove_trn(infoDevice, fragmentInfoDevice, residueInfoDevice, atomInfoDevice, gridDevice, ffDevice, moveFragType, tempFragDevice, tempInfoDevice, rngStatesDevice)

    cp.copyto(tempInfoHost[:numBlocks], tempInfoDevice[:numBlocks])

    period = infoHost['cryst']

    confIndexUnused = set(range(numBlocks))
    confProbabilities = np.zeros(numBlocks)
    confIndexUsed = set()
    needUpdate = False

    while confIndexUnused:
        it = confIndexUnused.pop()
        confIndexUsed.clear()
        confIndexUsed.add(it)

        energyMin = tempInfoHost[it]['charge']
        sumP = 0

        for iit in list(confIndexUnused):
            if distanceP(tempInfoHost[it]['position'], tempInfoHost[iit]['position'], period) <= infoHost['cutoff']:
                if tempInfoHost[iit]['charge'] < energyMin:
                    energyMin = tempInfoHost[iit]['charge']

                confIndexUsed.add(iit)
                confIndexUnused.remove(iit)

        for iit in confIndexUsed:
            confProbabilities[iit] = np.exp(-beta * (tempInfoHost[iit]['charge'] - energyMin))
            sumP += confProbabilities[iit]

        if sumP == 0:
            confIndex = it
        else:
            confPSum = 0
            for iit in confIndexUsed:
                confPSum += confProbabilities[iit] / sumP
                confProbabilities[iit] = confPSum

            ran = random.random()
            confIndex = next(iit for iit in confIndexUsed if ran < confProbabilities[iit])

        energyNew = tempInfoHost[confIndex]['charge']
        confProbabilities[confIndex] = np.exp(-beta * (energyNew - energyMin)) / sumP

        fnTmp = 1 / (len(confIndexUsed) * confProbabilities[confIndex])
        diff = energyNew
        p = min(1, fnTmp * np.exp(-beta * diff))

        ran = random.random()

        tempInfoHostType = tempInfoHost[confIndex]['type']

        for iit in confIndexUsed:
            tempInfoHost[iit]['type'] = -1

        if ran < p:
            for iit in list(confIndexUnused):
                if distanceP(tempInfoHost[confIndex]['position'], tempInfoHost[iit]['position'], period) <= infoHost['cutoff']:
                    tempInfoHost[iit]['type'] = -1
                    confIndexUnused.remove(iit)

            tempInfoHost[confIndex]['type'] = tempInfoHostType
            needUpdate = True

    if needUpdate:
        cp.copyto(tempInfoDevice[:numBlocks], cp.array(tempInfoHost[:numBlocks]))
        GupdateTrn(fragmentInfoDevice, residueInfoDevice, atomInfoDevice, tempFragDevice, tempInfoDevice, moveFragType)

    return needUpdate

# # Placeholder for the calcEnergy function
# def calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo):
#     # Implement the energy calculation logic here
#     pass

# # Placeholder for the distanceP function
# def distanceP(position1, position2, period):
#     # Implement the distance calculation function here
#     return np.linalg.norm(position1 - position2)
