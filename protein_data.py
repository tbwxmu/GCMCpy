"""

    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved    
    	Mingtian Zhao, Alexander D. MacKerell Jr.        
    E-mail: 
    	zhaomt@outerbanks.umaryland.edu
    	alex@outerbanks.umaryland.edu


"""


import re

class Atom:
    def __init__(self, serial, name, residue, sequence, x, y, z):
        self.serial = serial
        self.name = name
        self.residue = residue
        self.sequence = sequence
        self.x = x
        self.y = y
        self.z = z
        self.s = None


def read_pdb(pdb_file):
    '''Read a PDB file and return the crystal xyz and a list of atoms'''

    s = open(pdb_file, 'r').read()
    p = s.strip().split('\n')
    cryst = None
    for line in p:
        if line.lower().startswith('cryst1 '):#pdb have use this key word crstl
            cryst = line.split()[1:4]
            cryst = [float(i) for i in cryst]
            break

    atoms = []
    for line in p:
        if (line.lower().startswith('atom ') or line.lower().startswith('hetatm')) and len(line) > 54:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            serial = line[6:11].strip()
            name = line[12:16].strip()
            residue = line[17:20].strip()
            sequence = line[22:26].strip()
            atom = Atom(serial, name, residue, sequence, x, y, z)
            atom.s = line
            atoms.append(atom)

    if cryst is None:
        x = [atom.x for atom in atoms]
        y = [atom.y for atom in atoms]
        z = [atom.z for atom in atoms]
        cryst = [max(x)-min(x), max(y)-min(y), max(z)-min(z)]
    
    return cryst,atoms,s



def read_itp_block(itp_file):
    '''Read a Gromacs .itp file and return a list of atoms'''

    s = open(itp_file, 'r').read()
    blocks = []
    current_block = "title"
    blocks.append([current_block])
    for line in s.split('\n'):
        line = re.sub(r';.*', '', line)
        line = re.sub(r'#.*', '', line)
        match = re.match(r'\[(.*)\]', line)
        if match:
            current_block = match.group(1).strip()
            blocks.append([current_block])
        else:
            if current_block and line.strip():
                blocks[-1].append(line.strip())

    return blocks

def read_itp(itp_file):
    '''Read a Gromacs .itp file and return a list of atoms'''

    blocks = read_itp_block(itp_file)
    moleculetypes = []
    for blocknum in range(len(blocks)):
        block = blocks[blocknum]
        if block[0] == 'moleculetype':
            for line in block[1:]:
                line = line.strip()
                if line:
                    pattern = r'(\w+)\s+(\d+)'
                    match = re.match(pattern, line)
                    if match:
                        moleculetypes.append([match.group(1)])
                        break
            for atomblock in blocks[blocknum+1:]:
                if atomblock[0] == 'atoms':
                    for line in atomblock[1:]:
                        line = line.strip()
                        if line:
                            pattern = r'(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+([0-9\.\-]+)'
                            match = re.match(pattern, line)
                            if match:
                                moleculetypes[-1].append([match.group(2),match.group(3),match.group(4),match.group(5),match.group(7)])
                            # else:
                            #     print(atomblock[0],line)
                            #     break
                    break
    return moleculetypes[-1][1:]

def read_psf(psf_file):
    '''Read a CHARMM .psf file and return a list of atoms'''

    s = open(psf_file, 'r').read()

    p = s.strip().split('\n')
    p = p[1:] # skip the first line
    p = [line.split('*', 1)[0] for line in p]
    p = [line.split('!', 1)[0] for line in p]
    p = [line.split('#', 1)[0] for line in p]

    for i,line in enumerate(p):
        try:
            lineN = int(line)
            linei = i
            break
        except:
            pass

    p = p[linei + lineN + 1:]


    for i,line in enumerate(p):
        try:
            lineN = int(line)
            linei = i
            break
        except:
            pass

    p = p[linei+ 1:]

    atoms_top = []
    n = 0
    for line in p:
        line = line.strip()
        pattern = r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"

        match = re.match(pattern, line)
        if match:
            n += 1
            if n > lineN:
                break
            atoms_top.append([match.group(6),match.group(3),match.group(4),match.group(5),match.group(7)])
    
    return atoms_top
            # atoms_top.append([match.group(2),match.group(3),match.group(4),match.group(5),match.group(7)])



def read_top(top_file):
    '''Read a Gromacs .top file and return a list of atoms'''

    s = open(top_file, 'r').read()


    # blocks = {}
    # current_block = "title"
    # blocks[current_block] = []
    # for line in s.split('\n'):
    #     line = re.sub(r';.*', '', line)
    #     match = re.match(r'\[(.*)\]', line)
    #     if match:
    #         current_block = match.group(1).strip()
    #         if current_block not in blocks:
    #             blocks[current_block] = []
    #     else:
    #         if current_block and line.strip():
    #             blocks[current_block].append(line.strip())
    blocks = []
    current_block = "title"
    blocks.append([current_block])
    for line in s.split('\n'):
        line = re.sub(r';.*', '', line)
        line = re.sub(r'#.*', '', line)
        match = re.match(r'\[(.*)\]', line)
        if match:
            current_block = match.group(1).strip()
            blocks.append([current_block])
        else:
            if current_block and line.strip():
                blocks[-1].append(line.strip())
    
    content = re.sub(r';.*', '', s)
    content = re.sub(r'#ifdef .*?#endif', '', content, flags=re.DOTALL)

    pattern = r'^\s*#include "(.*?)"\s*$'
    matches = re.findall(pattern, content, re.MULTILINE)
    matches = [i.strip() for i in matches]

    itps = []
    for file in matches:
        if file.endswith('.itp'):
            itps += [read_itp_block(file)]
    
    molecules = []
    for block in blocks:
        if block[0] == 'molecules':
            for line in block[1:]:
                line = line.strip()
                if line:
                    pattern = r'(\w+)\s+(\d+)'
                    match = re.match(pattern, line)
                    if match:
                        molecules.append([match.group(1),match.group(2)])
            
            break
    
    moleculetypes = []

    for blocknum in range(len(blocks)):
        block = blocks[blocknum]
        if block[0] == 'moleculetype':
            for line in block[1:]:
                line = line.strip()
                if line:
                    pattern = r'(\w+)\s+(\d+)'
                    match = re.match(pattern, line)
                    if match:
                        moleculetypes.append([match.group(1)])
                        break
            for atomblock in blocks[blocknum+1:]:
                if atomblock[0] == 'atoms':
                    for line in atomblock[1:]:
                        line = line.strip()
                        if line:
                            pattern = r'(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+([0-9\.\-]+)'
                            match = re.match(pattern, line)
                            if match:
                                moleculetypes[-1].append([match.group(2),match.group(3),match.group(4),match.group(5),match.group(7)])
                            # else:
                            #     print(blocknum,line)
                            #     break
                    break

    for itp in itps:
        for blocknum in range(len(itp)):
            block = itp[blocknum]
            if block[0] == 'moleculetype':
                for line in block[1:]:
                    line = line.strip()
                    if line:
                        pattern = r'(\w+)\s+(\d+)'
                        match = re.match(pattern, line)
                        if match:
                            moleculetypes.append([match.group(1)])
                            break
                for atomblock in itp[blocknum+1:]:
                    if atomblock[0] == 'atoms':
                        for line in atomblock[1:]:
                            line = line.strip()
                            if line:
                                pattern = r'(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+([0-9\.\-]+)'
                                match = re.match(pattern, line)
                                if match:
                                    moleculetypes[-1].append([match.group(2),match.group(3),match.group(4),match.group(5),match.group(7)])
                                # else:
                                #     print(atomblock[0],line)
                                #     break
                        break
    
    atoms_top = []
    for molecule in molecules:
        for moleculetype in moleculetypes:
            if molecule[0] == moleculetype[0]:
                # print(molecule[0])
                # print(len(moleculetype))
                for i in range(int(molecule[1])):
                    atoms_top+=moleculetype[1:]
                break
    # print(moleculetype)
    # print(atoms_top[233336])
    return atoms_top, s
    # for i in moleculetypes:
    #     print(i[0])
    
    # print(molecules)
    # print([i[0] for i in blocks])      
    # # print(blocks['molecules'])
    # # print(blocks['moleculetype'])
    # for block in itps:
    #     print([i[0] for i in block])    
    #     for i in block:
    #         if i[0] == 'moleculetype':
    #             print(i)


def read_ff(ff_file):
    '''Read a Gromacs .ff file and return a list of atoms'''    
    nb_dict = {}
    nbfix_dict = {}
    blocks = read_itp_block(ff_file)#list of block, keyword for each list block

    for block in blocks:
        if block[0] == 'atomtypes':#charmm36.ff/silcs.itp charmm36.ff/ffnonbonded.itp
            for line in block[1:]:
                line = line.strip()
                if line:
                    pattern = r'(\S+)\s+(\S+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+(\S+)\s+([0-9\.\-]+)+\s+([0-9\.\-]+)'
                    match = re.match(pattern, line)
                    if match:
                        nb_dict[match.group(1)] = [float(match.group(6)),float(match.group(7))]
    
        if block[0] == 'nonbond_params' or block[0] == 'pairtypes':#charmm36.ff/ffnonbonded.itp
            for line in block[1:]:
                line = line.strip()
                if line:
                    pattern = r'(\S+)\s+(\S+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)\s+([0-9\.\-]+)'
                    match = re.match(pattern, line)
                    if match:
                        nbfix_dict[(match.group(1),match.group(2))] = [float(match.group(4)),float(match.group(5))]

    return nb_dict,nbfix_dict
