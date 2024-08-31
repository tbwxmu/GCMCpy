"""

    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved    
    	Mingtian Zhao, Alexander D. MacKerell Jr.        
    E-mail: 
    	zhaomt@outerbanks.umaryland.edu
    	alex@outerbanks.umaryland.edu

"""

import sys
import numpy as np
import random
from values import *
import copy
import re


class GCMCDataset:
    
    def get_grid(self):
        print("Getting grid points...")

        x_values = [atom.x for atom in self.atoms]
        y_values = [atom.y for atom in self.atoms]
        z_values = [atom.z for atom in self.atoms]

        

        print("x direction: min =", min(x_values), "max =", max(x_values),end = '\t')
        print("y direction: min =", min(y_values), "max =", max(y_values),end = '\t')
        print("z direction: min =", min(z_values), "max =", max(z_values))

        x_center = (min(x_values) + max(x_values)) / 2.0
        y_center = (min(y_values) + max(y_values)) / 2.0
        z_center = (min(z_values) + max(z_values)) / 2.0

        print("x center =", x_center, end = '\t')
        print("y center =", y_center, end = '\t')
        print("z center =", z_center)

        self.startxyz = [x_center - self.cryst[0]/2.0, y_center - self.cryst[1]/2.0, z_center - self.cryst[2]/2.0]

        print("startxyz =", self.startxyz, end = '\t')

        self.endxyz = [x_center + self.cryst[0]/2.0, y_center + self.cryst[1]/2.0, z_center + self.cryst[2]/2.0]

        print("endxyz =", self.endxyz)

        # self.grid_n = [int((self.endxyz[0] - self.startxyz[0]) / self.grid_dx), int((self.endxyz[1] - self.startxyz[1]) / self.grid_dx), int((self.endxyz[2] - self.startxyz[2]) / self.grid_dx)]

        # print("grid_n =", self.grid_n)

        



        if self.cavity_bias:
            print("Using cavity bias: True")

            print("Grid dx =", self.grid_dx)
            self.grid_n = [int((self.endxyz[0] - self.startxyz[0]) / self.grid_dx), int((self.endxyz[1] - self.startxyz[1]) / self.grid_dx), int((self.endxyz[2] - self.startxyz[2]) / self.grid_dx)]
            print("grid_n =", self.grid_n)

            grid = {(x,y,z) for x in range(self.grid_n[0]) for y in range(self.grid_n[1]) for z in range(self.grid_n[2])}
            
            
            # change_dx = set()

            # for x in set(np.arange(0, self.cavity_bias_dx + self.grid_dx/2, self.grid_dx)) | set(np.arange(0,-self.cavity_bias_dx - self.grid_dx/2, -self.grid_dx)):
            #     for y in set(np.arange(0, self.cavity_bias_dx + self.grid_dx/2, self.grid_dx)) | set(np.arange(0,-self.cavity_bias_dx - self.grid_dx/2, -self.grid_dx)):
            #         for z in set(np.arange(0, self.cavity_bias_dx + self.grid_dx/2, self.grid_dx)) | set(np.arange(0,-self.cavity_bias_dx - self.grid_dx/2, -self.grid_dx)):
            #             change_dx.add((x, y, z))
            
            # print("change_dx =", change_dx)

            # for atom in self.atoms:
            #     for dx in change_dx:
            #         x = int((atom.x - self.startxyz[0] + dx[0]) / self.grid_dx) % self.grid_n[0]
            #         y = int((atom.y - self.startxyz[1] + dx[1]) / self.grid_dx) % self.grid_n[1]
            #         z = int((atom.z - self.startxyz[2] + dx[2]) / self.grid_dx) % self.grid_n[2]
            #         grid.discard(x + y * self.grid_n[0] + z * self.grid_n[0] * self.grid_n[1])
            for atom in self.atoms:
                x = int((atom.x - self.startxyz[0]) / self.grid_dx) % self.grid_n[0]
                y = int((atom.y - self.startxyz[1]) / self.grid_dx) % self.grid_n[1]
                z = int((atom.z - self.startxyz[2]) / self.grid_dx) % self.grid_n[2]
                grid.discard((x,y,z))
            
            

            # grid_sum = np.sum(self.grid)
            self.cavity_bias_factor = len(grid) / (self.grid_n[0] * self.grid_n[1] * self.grid_n[2])
            print("The number of grid points =", len(grid))
            print("cavity_bias_factor =", self.cavity_bias_factor)
            if self.cavity_bias_factor < 0.05:
                print("Error: cavity_bias_factor is too small, please decrease cavity_bias_dx")
                sys.exit()
        else:
            print("Using cavity bias: False")

            self.cavity_bias_factor = 1.0
            self.grid_dx = 1.0

            print("Grid dx =", self.grid_dx)
            self.grid_n = [int((self.endxyz[0] - self.startxyz[0]) / self.grid_dx), int((self.endxyz[1] - self.startxyz[1]) / self.grid_dx), int((self.endxyz[2] - self.startxyz[2]) / self.grid_dx)]
            print("grid_n =", self.grid_n)

            grid = {(x,y,z) for x in range(self.grid_n[0]) for y in range(self.grid_n[1]) for z in range(self.grid_n[2])}
            print("The number of grid points =", len(grid))
            print("cavity_bias_factor =", self.cavity_bias_factor)
            
        self.grid = np.zeros(len(grid) * 3, dtype = np.float32)
        for i, grid_point in enumerate(grid):
            x, y, z = grid_point
            self.grid[i*3] = x * self.grid_dx + self.startxyz[0]
            self.grid[i*3+1] = y * self.grid_dx + self.startxyz[1]
            self.grid[i*3+2] = z * self.grid_dx + self.startxyz[2]
        
        del grid
        
            # self.grid = np.zeros(self.grid_n[0] * self.grid_n[1] * self.grid_n[2], dtype = np.int32)
            # print("The number of grid points =", len(self.grid))
            # change_dx = [((i-1) * self.cavity_bias_dx,(j-1) * self.cavity_bias_dx, (k-1) * self.cavity_bias_dx) for i in range(3) for j in range(3) for k in range(3)]
            # change_dx = set(change_dx)
            # # print('change_dx =', change_dx)
            # for atom in self.atoms:
            #     for dx in change_dx:
            #         x = int((atom.x - self.startxyz[0] + dx[0]) / self.grid_dx) % self.grid_n[0]
            #         y = int((atom.y - self.startxyz[1] + dx[1]) / self.grid_dx) % self.grid_n[1]
            #         z = int((atom.z - self.startxyz[2] + dx[2]) / self.grid_dx) % self.grid_n[2]

            #         self.grid[x + y * self.grid_n[0] + z * self.grid_n[0] * self.grid_n[1]] += 1
            # grid_sum = sum(bool(x) for x in self.grid)
            # self.cavity_bias_factor = 1.0 - grid_sum / len(self.grid)
            # print("The number of grid points with atoms =", grid_sum)
            # print("The cavity bias factor =", self.cavity_bias_factor)
            # if self.cavity_bias_factor < 0.1 and self.cavity_bias_factor > 0.0:
            #     print("Error: The cavity bias factor is too small. Please decrease the cavity bias dx.")
            #     sys.exit(1) 
 
    def get_move(self):
        self.moveArray = np.array(random.choices(range(len(self.fragmentName)), weights=self.mctime, k=self.mcsteps),dtype=np.int32)
        # self.moveArray = np.random.choice(np.arange(len(self.fragmentName),dtype=np.int32), p=self.mctime, size=self.mcsteps)


        for i, n in enumerate(self.moveArray):
            if self.fragmentName[n] == 'SOL':
                self.moveArray[i] = self.moveArray[i] * 4 + random.choices(range(len(self.attempt_prob_water)), weights=self.attempt_prob_water, k=1)[0]
            else:
                self.moveArray[i] = self.moveArray[i] * 4 + random.choices(range(len(self.attempt_prob_frag)), weights=self.attempt_prob_frag, k=1)[0]

        # for i, n in enumerate(self.moveArray):
        #     if self.fragmentName[n] == 'SOL':
        #         self.moveArray[i] = self.moveArray[i] * 4 + np.random.choice(range(len(self.attempt_prob_water)), p=self.attempt_prob_water, size=1)[0]
        #     else:
        #         self.moveArray[i] = self.moveArray[i] * 4 + np.random.choice(range(len(self.attempt_prob_frag)), p=self.attempt_prob_frag, size=1)[0]

        #attempt_prob_ 4 type add del tranlate rot
        self.moveArray_n = [0 for i in range(len(self.attempt_prob_frag) * len(self.fragmentName))]
        self.moveArray_frag = [0 for i in range(len(self.fragmentName))]
        #count each frag each move type 
        for i in self.moveArray:
            self.moveArray_n[i] += 1
            self.moveArray_frag[i//4] += 1
        
        movementName = ['Insert', 'Delete', 'Translate', 'Rotate']
        for i in range(len(self.fragmentName)):
            print(f"Solute {self.fragmentName[i]}\t Movement {self.moveArray_frag[i]} times", end='\t')
            for j in range(len(self.attempt_prob_frag)):
                print(f"{movementName[j]} {self.moveArray_n[i*4+j]} times", end='\t')
            print()


        
    def set_fixCut(self):
        
        x0 = float('inf')
        y0 = float('inf')
        z0 = float('inf')
        n = -1
        for atom in self.fix_atoms:
            x, y, z = atom.x, atom.y, atom.z
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) > self.fixCutoff:
                n += 1
                x0, y0, z0 = x, y, z
                atom.sequence2 = n
            else:
                atom.sequence2 = n

    def update_data(self):
        self.fragmentInfo = np.empty(len(self.fragmentName), dtype=AtomArray_dtype)
        for i, frag in enumerate(self.fragments):
            # self.fragmentInfo[i]['name'] = self.fragmentName[i]
            self.fragmentInfo[i]['name'] = np.array(self.fragmentName[i].ljust(4)[:4], dtype='S4')

            
            self.fragmentInfo[i]['muex'] = self.fragmuex[i]
            self.fragmentInfo[i]['conc'] = self.fragconc[i]
            self.fragmentInfo[i]['confBias'] = self.fragconf[i]
            self.fragmentInfo[i]['mcTime'] = self.mctime[i]

            self.fragmentInfo[i]['totalNum'] = len(self.fraglist[i])

            # Two times of the number of fragments or the number of fragments plus the number of fragments in the largest configuration
            #阿伏伽德罗常数指摩尔微粒（可以是分子、原子、离子、电子等）所含的微粒的数目，符号表示为NA。阿伏伽德罗常量一般取值为6.022×1023/mol
            maxNum1 = self.fragconc[i] * self.volume * MOLES_TO_MOLECULES * 2 + self.moveArray_n[i*4]
            maxNum2 = len(self.fraglist[i]) + self.moveArray_n[i*4] 

            self.fragmentInfo[i]['maxNum'] = max(maxNum1, maxNum2)

            print(f"Solute {self.fragmentName[i]}: Total number: {self.fragmentInfo[i]['totalNum']}, Max number: {self.fragmentInfo[i]['maxNum']}")

            self.fragmentInfo[i]['num_atoms'] = len(frag)

            atom_center = np.zeros(3)
            for atom in frag:
                atom_center += np.array([atom.x, atom.y, atom.z])
            atom_center /= len(frag)
            # print(f"Fragment {self.fragmentName[i]}: Center of conf: {atom_center}", end='\t')
            
            for atom in frag:
                atom.x -= atom_center[0]
                atom.y -= atom_center[1]
                atom.z -= atom_center[2]
            # atom_center = np.zeros(3)

            for j, atom in enumerate(frag):
                self.fragmentInfo[i]['atoms'][j]['position'][0] = atom.x
                self.fragmentInfo[i]['atoms'][j]['position'][1] = atom.y
                self.fragmentInfo[i]['atoms'][j]['position'][2] = atom.z
                self.fragmentInfo[i]['atoms'][j]['charge'] = atom.charge
                self.fragmentInfo[i]['atoms'][j]['type'] = atom.typeNum

                # atom_center += np.array([atom.x, atom.y, atom.z])
        if len(self.fix_atoms) == 0:
            TotalResidueNum = sum([self.fragmentInfo[i]['maxNum'] for i in range(len(self.fragmentName))])
        else:
            TotalResidueNum = self.fix_atoms[-1].sequence2 + 1 + sum([self.fragmentInfo[i]['maxNum'] for i in range(len(self.fragmentName))])
        TotalAtomNum = len(self.fix_atoms) + sum([self.fragmentInfo[i]['maxNum'] * self.fragmentInfo[i]['num_atoms'] for i in range(len(self.fragmentName))]) 

        self.residueInfo = np.empty(TotalResidueNum, dtype=Residue_dtype)
        self.atomInfo = np.empty(TotalAtomNum, dtype=Atom_dtype)

        n = -1
        for i, atom in enumerate(self.fix_atoms):

            sequence = atom.sequence2 
            if sequence != n:
                n = sequence
                self.residueInfo[sequence]['atomNum'] = 0
                self.residueInfo[sequence]['atomStart'] = i
                self.residueInfo[sequence]['position'][0] = 0
                self.residueInfo[sequence]['position'][1] = 0
                self.residueInfo[sequence]['position'][2] = 0
            
            self.residueInfo[sequence]['atomNum'] += 1
            self.residueInfo[sequence]['position'][0] += atom.x
            self.residueInfo[sequence]['position'][1] += atom.y
            self.residueInfo[sequence]['position'][2] += atom.z

            self.atomInfo[i]['position'][0] = atom.x
            self.atomInfo[i]['position'][1] = atom.y
            self.atomInfo[i]['position'][2] = atom.z
            self.atomInfo[i]['charge'] = atom.charge
            self.atomInfo[i]['type'] = atom.typeNum
        if len(self.fix_atoms) == 0:
            sequence = -1
        for i in range(sequence + 1):
            self.residueInfo[i]['position'][0] /= self.residueInfo[i]['atomNum']
            self.residueInfo[i]['position'][1] /= self.residueInfo[i]['atomNum']
            self.residueInfo[i]['position'][2] /= self.residueInfo[i]['atomNum']
        
        for i, frag in enumerate(self.fragments):
            self.fragmentInfo[i]['startRes'] = sequence + 1
            for j in range(self.fragmentInfo[i]['maxNum']):
                sequence += 1
                self.residueInfo[sequence]['atomNum'] = self.fragmentInfo[i]['num_atoms']
                self.residueInfo[sequence]['atomStart'] = self.residueInfo[sequence - 1]['atomStart'] + self.residueInfo[sequence - 1]['atomNum']
                self.residueInfo[sequence]['position'][0] = 0
                self.residueInfo[sequence]['position'][1] = 0
                self.residueInfo[sequence]['position'][2] = 0
                if j < self.fragmentInfo[i]['totalNum']:
                    for k, atom in enumerate(self.fraglist[i][j]):
                        self.residueInfo[sequence]['position'][0] += atom.x
                        self.residueInfo[sequence]['position'][1] += atom.y
                        self.residueInfo[sequence]['position'][2] += atom.z

                        self.atomInfo[self.residueInfo[sequence]['atomStart'] + k]['position'][0] = atom.x
                        self.atomInfo[self.residueInfo[sequence]['atomStart'] + k]['position'][1] = atom.y
                        self.atomInfo[self.residueInfo[sequence]['atomStart'] + k]['position'][2] = atom.z
                        self.atomInfo[self.residueInfo[sequence]['atomStart'] + k]['charge'] = atom.charge
                        self.atomInfo[self.residueInfo[sequence]['atomStart'] + k]['type'] = atom.typeNum
                    
                    self.residueInfo[sequence]['position'][0] /= self.residueInfo[sequence]['atomNum']
                    self.residueInfo[sequence]['position'][1] /= self.residueInfo[sequence]['atomNum']
                    self.residueInfo[sequence]['position'][2] /= self.residueInfo[sequence]['atomNum']
                

                

                # self.residueInfo[sequence]['position'][0] /= self.residueInfo[sequence]['atomNum']
                # self.residueInfo[sequence]['position'][1] /= self.residueInfo[sequence]['atomNum']
                # self.residueInfo[sequence]['position'][2] /= self.residueInfo[sequence]['atomNum']

        
        self.ff = np.empty(len(self.ff_pairs) * 2, dtype=np.float32)
        for i, pair in enumerate(self.ff_pairs):
            self.ff[i * 2] = pair[0]
            self.ff[i * 2 + 1] = pair[1]
        
        
        self.SimInfo = np.empty(1, dtype=Info_dtype)


        self.SimInfo[0]['ffXNum'] = len(self.atomtypes1)
        self.SimInfo[0]['ffYNum'] = len(self.atomtypes2)

        self.SimInfo[0]['mcsteps'] = self.mcsteps
        self.SimInfo[0]['cutoff'] = self.cutoff
        self.SimInfo[0]['grid_dx'] = self.grid_dx
        self.SimInfo[0]['startxyz'][0] = self.startxyz[0]
        self.SimInfo[0]['startxyz'][1] = self.startxyz[1]
        self.SimInfo[0]['startxyz'][2] = self.startxyz[2]
        self.SimInfo[0]['cryst'][0] = self.cryst[0]
        self.SimInfo[0]['cryst'][1] = self.cryst[1]
        self.SimInfo[0]['cryst'][2] = self.cryst[2]

        self.SimInfo[0]['cavityFactor'] = self.cavity_bias_factor

        self.SimInfo[0]['fragTypeNum'] = len(self.fragmentName)

        self.SimInfo[0]['totalGridNum'] = len(self.grid) // 3
        self.SimInfo[0]['totalResNum'] = TotalResidueNum
        self.SimInfo[0]['totalAtomNum'] = TotalAtomNum

        self.SimInfo[0]['showInfo'] = self.showInfo

        self.SimInfo[0]['PME'] = self.PME

        self.SimInfo[0]['seed'] = random.randint(0, (2**32)-1)

        # 
        # print('showInfo', self.SimInfo[0]['showInfo'])

        # n = 0
        # for frag in self.fragmentInfo:
        #     for i in range(frag['num_atoms']):
        #         for atom in self.fix_atoms:
        #             n +=1
        #             if n % 100 == 0:
        #                 type1 = frag['atoms'][i]['type']
        #                 type2 = atom.typeNum
        #                 # print(type1,type2,self.atomtypes1[type1])
        #                 print(type1, type2, self.atomtypes1[type1], self.atomtypes2[type2], self.ff[(type1 * self.SimInfo[0]['ffXNum'] + type2)* 2], self.ff[(type1 * self.SimInfo[0]['ffXNum'] + type2)* 2 + 1])
                

        # for i in self.atomtypes1:
        #     print(i, end=' ')
        # print()
        # for i in self.atomtypes2:
        #     print(i, end=' ')
        # print()

        


            # atom_center /= len(frag)
            # print(f"Fragment {self.fragmentName[i]}: The final center of conf: {atom_center}")
        

    def get_data(self):

        print("Getting data...")
        
        for i in range(len(self.fragmentName)):
            print(f"Solute %s: Total number: %d" % (self.fragmentName[i], self.fragmentInfo[i]['totalNum']))

        # s = 'REMARK    GENERATED BY pyGCMC\nTITLE     Protein\nREMARK    THIS IS A SIMULATION BOX\n'        
        s = 'CRYST1  %.3f  %.3f  %.3f  90.00  90.00  90.00 P 1           1\n' % (self.cryst[0], self.cryst[1], self.cryst[2])

        for atom in self.fix_atoms:
            s += atom.s
            s += '\n'

        try:
            atomNum = int(self.fix_atoms[-1].serial) + 1
        except:
            atomNum = 1

        try:
            residueNum = int(self.fix_atoms[-1].sequence) + 1
        except:
            residueNum = 1

        # atom = self.fragments[0][0]
        # print(atom.name, atom.residue, atom.type)

        for i in range(len(self.fragmentName)):
            for j in range(self.fragmentInfo[i]['totalNum']):
                resNum = self.fragmentInfo[i]['startRes'] + j
                res = self.residueInfo[resNum]
                for k in range(res['atomNum']):
                    atom = self.atomInfo[res['atomStart'] + k]
                    s += 'ATOM  %5d  %-4s%-4s %4d    %8.3f%8.3f%8.3f  1.00  0.00\n' % ((atomNum - 1) % 99999 + 1, self.fragments[i][k].name[:4], self.fragmentName[i][:4], (residueNum - 1) % 9999 + 1, atom['position'][0], atom['position'][1], atom['position'][2])
                    atomNum += 1
                residueNum += 1
        s += 'END\n'

        self.PDBString = s

        if self.topologyType == 'top':
            
            s = copy.deepcopy(self.TOPString)
            try:
                pattern = r'\[\s*molecules\s*\]'
                parts = re.split(pattern, s)
                # print(parts[1])
                parts[1] = parts[1].strip()+'\n'
                p = re.findall(r'(\S+?)\s+(\d+)', parts[1])
                d = {i[0]:i[1] for i in p}
                # print(d)

                for i in range(len(self.fragmentName)):
                    if self.fragmentName[i] in d:
                        ni = self.fragmentName[i]

                        parts[1] = re.sub('%s\\s+%s' % (ni,d[ni]), '%-s\\t%d' % (self.fragmentName[i], self.fragmentInfo[i]['totalNum']), parts[1])  
                    else:
                        parts[1] += '%-s\t%d\n' % (self.fragmentName[i], self.fragmentInfo[i]['totalNum'])
                
                self.TOPString = parts[0] + '\n[ molecules ]\n' + parts[1]
                # print(self.TOPString)

            except:
                print("Error: writing top file error")
                sys.exit(1)






        # print(s)