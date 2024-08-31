import cupy as cp
from .gcmc_h import *


__all__=["Min",
"rotate_atoms",
"rotate_atoms_shared",
"randomFragment",
"fast_round",
"distanceP",
"calc_vdw_nbfix",
"calc_elec",
"calcProtEnergy",
"calcFragEnergy",
"calcEnergy",
]

def Min(a, b):
    return a if not b < a else b


def rotate_atoms(atoms: Atom, num_atoms: int, axis: cp.ndarray, angle: float) :
    # Normalize the axis vector
    axis = axis / cp.linalg.norm(axis)

    # Convert the angle to radians
    angle *= 2 * cp.pi

    # Rotate the atoms with a number less than 128
    for idx in range(min(num_atoms, 128)):
        atom_position = cp.copy(atoms[idx]['position'])

        # Compute the rotation matrix
        c = cp.cos(angle)
        s = cp.sin(angle)
        C = 1 - c

        R = cp.zeros((3, 3))
        R[0][0] = axis[0] * axis[0] * C + c
        R[0][1] = axis[0] * axis[1] * C - axis[2] * s
        R[0][2] = axis[0] * axis[2] * C + axis[1] * s
        R[1][0] = axis[1] * axis[0] * C + axis[2] * s
        R[1][1] = axis[1] * axis[1] * C + c
        R[1][2] = axis[1] * axis[2] * C - axis[0] * s
        R[2][0] = axis[2] * axis[0] * C - axis[1] * s
        R[2][1] = axis[2] * axis[1] * C + axis[0] * s
        R[2][2] = axis[2] * axis[2] * C + c

        # Apply the rotation matrix
        atoms[idx]['position'] = cp.dot(R, atom_position)


def rotate_atoms_shared(atoms, num_atoms, axis, angle):
    axis = axis / cp.linalg.norm(axis)

    # Compute the rotation matrix
    c = cp.cos(angle)
    s = cp.sin(angle)
    C = 1 - c

    R = cp.zeros((3, 3))
    R[0][0] = axis[0] * axis[0] * C + c
    R[0][1] = axis[0] * axis[1] * C - axis[2] * s
    R[0][2] = axis[0] * axis[2] * C + axis[1] * s
    R[1][0] = axis[1] * axis[0] * C + axis[2] * s
    R[1][1] = axis[1] * axis[1] * C + c
    R[1][2] = axis[1] * axis[2] * C - axis[0] * s
    R[2][0] = axis[2] * axis[0] * C - axis[1] * s
    R[2][1] = axis[2] * axis[1] * C + axis[0] * s
    R[2][2] = axis[2] * axis[2] * C + c

    # Rotate the atoms
    for i in range(num_atoms):
        atom_position = cp.copy(atoms[i]['position'])
        atoms[i]['position'] = cp.dot(R, atom_position)


def randomFragment(SharedInfo, SharedFragmentInfo, GTempInfo, Ggrid, rng_states):
    randomR = cp.random.uniform(0, 1, 3) * SharedInfo['grid_dx']
    randomThi = cp.random.uniform(0, 1, 3)
    randomPhi = cp.random.uniform(0, 1) * 2 * cp.pi
    gridN = cp.random.randint(0, SharedInfo['totalGridNum'])
    randomR += Ggrid[gridN * 3: gridN * 3 + 3]
    #roate all frag atoms
    rotate_atoms_shared(SharedFragmentInfo['atoms'], SharedFragmentInfo['num_atoms'], randomThi, randomPhi)

    #translate all frag atoms 
    for i in range(SharedFragmentInfo['num_atoms']):
        SharedFragmentInfo['atoms'][i]['position'] += randomR

    GTempInfo['position'] = randomR
    GTempInfo['type'] = -1


def fast_round(a):
    return int(a + 0.5) if a >= 0 else int(a - 0.5)


def distanceP(x, y, period):
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    dz = x[2] - y[2]

    dx -= fast_round(dx / period[0]) * period[0]
    dy -= fast_round(dy / period[1]) * period[1]
    dz -= fast_round(dz / period[2]) * period[2]

    return cp.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def calc_vdw_nbfix(sigma, epsilon, dist_sqrd):
    """
    //If NBFIX entry exists for the pair, then this calculation
    //!V(Lennard-Jones) = 4*Eps,i,j[(sigma,i,j/ri,j)**12 - (sigma,i,j/ri,j)**6]
    // sigma and Eps are the nbfix entries
    // units from force field: sigma (nm), epsilon (kJ)
    """   
    sigma_sqrd = sigma * sigma * 100
    sigma_dist_sqrd = sigma_sqrd / dist_sqrd
    E_vdw = 4 * epsilon * (cp.power(sigma_dist_sqrd, 6) - cp.power(sigma_dist_sqrd, 3))
    return E_vdw


def calc_elec(charge1, charge2, dist):
    E_elec = 1388.431112 * (charge1 * charge2) / dist
    return E_elec


def calcProtEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo, sh_energy):
    maxResidueNum = GfragmentInfo['startRes']

    for resi in range(maxResidueNum):
        if distanceP(GTempInfo['position'], GresidueInfo[resi]['position'], SharedInfo['cryst']) > SharedInfo['cutoff']:
            continue#>=15 skip

        resiStart = GresidueInfo[resi]['atomStart']
        resiEnd = resiStart + GresidueInfo[resi]['atomNum']
        resiEnergy = 0

        for atomi in range(resiStart, resiEnd):
            for atomj in range(SharedFragmentInfo['num_atoms']):
                distance = distanceP(GatomInfo[atomi]['position'], SharedFragmentInfo['atoms'][atomj]['position'], SharedInfo['cryst'])
                # The `typeij` variable is used to calculate the index for accessing the parameters
                # related to the interaction between two atom types in the force field. It is
                # calculated based on the types of atoms involved in the interaction.
                typeij = SharedFragmentInfo['atoms'][atomj]['type'] * SharedInfo['ffYNum'] + GatomInfo[atomi]['type']
                resiEnergy += calc_vdw_nbfix(Gff[typeij * 2], Gff[typeij * 2 + 1], distance ** 2)
                resiEnergy += calc_elec(SharedFragmentInfo['atoms'][atomj]['charge'], GatomInfo[atomi]['charge'], distance)
        
        sh_energy += resiEnergy


def calcFragEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo, sh_energy):
    for fragType in range(SharedInfo['fragTypeNum']):
        startResidueNum = GfragmentInfo[fragType]['startRes']
        endResidueNum = startResidueNum + GfragmentInfo[fragType]['totalNum']

        for resi in range(startResidueNum, endResidueNum):
            if resi == SharedFragmentInfo['startRes']:
                continue

            if distanceP(GTempInfo['position'], GresidueInfo[resi]['position'], SharedInfo['cryst']) > SharedInfo['cutoff']:
                continue

            resiStart = GresidueInfo[resi]['atomStart']
            resiEnd = resiStart + GresidueInfo[resi]['atomNum']
            resiEnergy = 0

            for atomi in range(resiStart, resiEnd):
                for atomj in range(SharedFragmentInfo['num_atoms']):
                    distance = distanceP(GatomInfo[atomi]['position'], SharedFragmentInfo['atoms'][atomj]['position'], SharedInfo['cryst'])
                    typeij = SharedFragmentInfo['atoms'][atomj]['type'] * SharedInfo['ffYNum'] + GatomInfo[atomi]['type']
                    resiEnergy += calc_vdw_nbfix(Gff[typeij * 2], Gff[typeij * 2 + 1], distance ** 2)
                    resiEnergy += calc_elec(SharedFragmentInfo['atoms'][atomj]['charge'], GatomInfo[atomi]['charge'], distance)
            
            sh_energy += resiEnergy


def calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo):
    sh_energy = cp.zeros(1)

    calcProtEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo, sh_energy)
    calcFragEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo, sh_energy)

    GTempInfo['charge'] = sh_energy[0]
