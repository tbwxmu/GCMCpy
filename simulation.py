"""

    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved    
    	Mingtian Zhao, Alexander D. MacKerell Jr.        
    E-mail: 
    	zhaomt@outerbanks.umaryland.edu
    	alex@outerbanks.umaryland.edu

"""

import sys
import numpy as np

class GCMCSimulation:
    
    def get_simulation(self):

        
        self.get_fragment()
        self.get_forcefield()

        self.get_move()
        self.get_grid()

        self.show_parameters()

        
        
        if self.pdb_file is None:
            print("Error: pdb file not set")
            sys.exit(1)

        if self.top_file is None and self.psf_file is None:
            print("Error: top file or psf file not set")
            sys.exit(1)
        
        self.atomtypes1 = []
        self.atomtypes2 = []
        for frag in self.fragments:
            for atom in frag:
                if atom.type not in self.atomtypes2:
                    self.atomtypes1.append(atom.type)
                    self.atomtypes2.append(atom.type)
        
        for atom in self.atoms:
            if atom.type not in self.atomtypes2:
                self.atomtypes2.append(atom.type)
        
        print(f"Solute atom type number: {len(self.atomtypes1)}")
        print(f"Total atom type number: {len(self.atomtypes2)}")

        self.ff_pairs = []

        for i, type1 in enumerate(self.atomtypes1):
            for j, type2 in enumerate(self.atomtypes2):
                ff_pair = [0,0]
                if (type1, type2) in self.nbfix_dict:
                    ff_pair[0] = self.nbfix_dict[(type1, type2)][0]
                    ff_pair[1] = self.nbfix_dict[(type1, type2)][1]
                elif (type2, type1) in self.nbfix_dict:
                    ff_pair[0] = self.nbfix_dict[(type2, type1)][0]
                    ff_pair[1] = self.nbfix_dict[(type2, type1)][1]
                else:
                    sigma1 = self.nb_dict[type1][0]
                    sigma2 = self.nb_dict[type2][0]
                    sigma = (sigma1 + sigma2) / 2
                    epsilon1 = self.nb_dict[type1][1]
                    epsilon2 = self.nb_dict[type2][1]
                    epsilon = np.sqrt(epsilon1 * epsilon2)

                    ff_pair[0] = sigma
                    ff_pair[1] = epsilon
                self.ff_pairs.append(ff_pair)

        
        print(f"Total FF pair number: {len(self.ff_pairs)}")
        
        # print(self.ff_pairs)

        for atom in self.atoms:
            atom.typeNum = self.atomtypes2.index(atom.type)
            
        for frag in self.fragments:
            for atom in frag:
                atom.typeNum = self.atomtypes2.index(atom.type)


        self.fix_atoms = []
        relax_atoms = [[] for i in range(len(self.fragments))]
        for atom in self.atoms:
            if atom.residue in self.fragmentName:
                relax_atoms[self.fragmentName.index(atom.residue)].append(atom)
            else:
                self.fix_atoms.append(atom)
        

        self.fraglist =[[relax_atoms[i][j:j + len(self.fragments[i])] for j in range(0, len(relax_atoms[i]), len(self.fragments[i]))] for i in range(len(self.fragments))]
        
        print(f"Total fixed atom number: {len(self.fix_atoms)}")
        print(f"Total relax atom number: {len(self.atoms) - len(self.fix_atoms)}")
        
        # for i, frag in enumerate(self.fragments):
        #     print(f"Fragment {self.fragmentName[i]} number: {len(self.fraglist[i])}")


        self.set_fixCut()

        self.update_data()