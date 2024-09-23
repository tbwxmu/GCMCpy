import cupy as cp
import numpy as np
from .gcmc_energy import *
from .gcmc_h import *

__all__=["Gmove_add",
"GupdateAdd",
"move_add",
]

# Kernel function for move_add
# @cp.fuse(kernel=True)
@cp.fuse()
def Gmove_add(Ginfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, GTempInfo, rng_states):
    SharedInfo = Ginfo[0]  # Copy global information to shared memory
    SharedFragmentInfo = GfragmentInfo[moveFragType]  # Copy fragment information
    SharedFragmentInfo.startRes = -1

    threadId = cp.numThreadsPerBlock * cp.blockIdx.x + cp.threadIdx.x
    rng_state = rng_states[threadId]

    # Generate random fragment
    randomFragment(SharedInfo, SharedFragmentInfo, GTempInfo[cp.blockIdx.x], Ggrid, rng_state)

    # Copy shared fragment information back to global memory
    # if cp.threadIdx.x == 0:
    #     GTempFrag[cp.blockIdx.x] = SharedFragmentInfo

    # Calculate energy
    calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo[cp.blockIdx.x])

# Kernel function for updateAdd
@cp.fuse()
def GupdateAdd(GfragmentInfo, GresidueInfo, GatomInfo, GTempFrag, GTempInfo, moveFragType, totalNum):
    bidStartRes = GfragmentInfo[moveFragType].startRes + GTempInfo[cp.blockIdx.x].type
    #if not GCMC just add frags in the box grid, we can use parallel style like the above
    bidAtomNum = GresidueInfo[bidStartRes].atomNum
    bidStartAtom = GresidueInfo[bidStartRes].atomStart

    GresidueInfo[bidStartRes].position[0] = GTempInfo[cp.blockIdx.x].position[0]
    GresidueInfo[bidStartRes].position[1] = GTempInfo[cp.blockIdx.x].position[1]
    GresidueInfo[bidStartRes].position[2] = GTempInfo[cp.blockIdx.x].position[2]

    for i in range(cp.threadIdx.x, bidAtomNum, cp.numThreadsPerBlock):
        GatomInfo[bidStartAtom + i].position = GTempFrag[cp.blockIdx.x].atoms[i].position
        GatomInfo[bidStartAtom + i].type = GTempFrag[cp.blockIdx.x].atoms[i].type
        GatomInfo[bidStartAtom + i].charge = GTempFrag[cp.blockIdx.x].atoms[i].charge

# Main move_add function
def move_add(infoHost, Ginfo, fragmentInfoHost, fragmentInfoDevice, residueInfoDevice, atomInfoDevice,
              gridDevice, ffDevice, moveFragType, tempFragDevice, tempInfoHost, tempInfoDevice, rngStatesDevice):
    #moveFragType 0~8 based n types of frag
    if fragmentInfoHost[moveFragType].totalNum == fragmentInfoHost[moveFragType].maxNum:
        return False

    numBlocks = fragmentInfoHost[moveFragType].confBias

    # Launch the move_add kernel
    # Gmove_add(numBlocks, cp.numThreadsPerBlock)(infoDevice, fragmentInfoDevice, residueInfoDevice, atomInfoDevice, gridDevice, ffDevice, moveFragType, tempFragDevice, tempInfoDevice, rngStatesDevice)
    Gmove_add(Ginfo, fragmentInfoDevice, residueInfoDevice, atomInfoDevice, gridDevice, ffDevice, moveFragType,
               tempFragDevice, tempInfoDevice, rngStatesDevice)
    
    # Copy tempInfo back to host
    cp.cuda.runtime.memcpy(tempInfoHost, tempInfoDevice, size=tempInfoHost.nbytes, kind=cp.cuda.runtime.memcpyDeviceToHost)

    period = infoHost.cryst
    numBars = period[0] * period[1] * period[2] * fragmentInfoHost[moveFragType].conc * MOLES_TO_MOLECULES
    B = beta * fragmentInfoHost[moveFragType].muex + np.log(numBars)

    confIndexUnused = set(range(numBlocks))
    confProbabilities = np.zeros(numBlocks)
    confIndexUsed = set()

    needUpdate = False

    while confIndexUnused:
        it = confIndexUnused.pop()
        confIndexUsed.clear()
        confIndexUsed.add(it)

        sumP = 0.0
        energyMin = tempInfoHost[it].charge

        for iit in confIndexUnused.copy():
            if distanceP(tempInfoHost[it].position, tempInfoHost[iit].position, period) <= infoHost.cutoff:
                if tempInfoHost[iit].charge < energyMin:
                    energyMin = tempInfoHost[iit].charge
                confIndexUsed.add(iit)
                confIndexUnused.remove(iit)

        sumP = sum(cp.exp(-beta * (tempInfoHost[iit].charge - energyMin)) for iit in confIndexUsed)
        if sumP == 0:
            confIndex = it
        else:
            confPSum = 0
            for iit in confIndexUsed:
                confProbabilities[iit] = np.exp(-beta * (tempInfoHost[iit].charge - energyMin)) / sumP
                confPSum += confProbabilities[iit]

            ran = np.random.random()
            for iit in confIndexUsed:
                if ran < confProbabilities[iit]:
                    confIndex = iit
                    break

        fnTmp = infoHost.cavityFactor / (len(confIndexUsed) * confProbabilities[confIndex])
        energyNew = tempInfoHost[confIndex].charge
        p = min(1, fnTmp / (fragmentInfoHost[moveFragType].totalNum + 1) * np.exp(B - beta * (energyNew)))

        if np.random.random() < p:
            for iit in list(confIndexUnused):
                if distanceP(tempInfoHost[confIndex].position, tempInfoHost[iit].position, period) <= infoHost.cutoff:
                    confIndexUnused.remove(iit)

            if fragmentInfoHost[moveFragType].totalNum < fragmentInfoHost[moveFragType].maxNum:
                tempInfoHost[confIndex].type = fragmentInfoHost[moveFragType].totalNum
                fragmentInfoHost[moveFragType].totalNum += 1
                needUpdate = True
                break

    if needUpdate:
        GupdateAdd(numBlocks, cp.numThreadsPerBlock)(fragmentInfoDevice, residueInfoDevice, atomInfoDevice, tempFragDevice, tempInfoDevice, moveFragType, fragmentInfoHost[moveFragType].totalNum)
    return needUpdate
