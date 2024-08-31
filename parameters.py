"""

    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved    
    	Mingtian Zhao, Alexander D. MacKerell Jr.        
    E-mail: 
    	zhaomt@outerbanks.umaryland.edu
    	alex@outerbanks.umaryland.edu

"""

import sys

class GCMCParameters:
    
    def get_fragmuex(self, fragmuex):
        
        fragmuex = [float(i) for i in fragmuex if len(i) > 0]
        if len(fragmuex) != len(self.fragmentName):
            print("Error: fragmuex number not match")
            sys.exit(1)
        else:
            self.fragmuex = fragmuex
    
    def get_fragconf(self, fragconf):

        fragconf = [int(i) for i in fragconf if len(i) > 0]

        for i in fragconf:
            if i <= 0:
                print("Error: fragconf number <= 0")
                sys.exit(1)

        if len(fragconf) == 1:
            fragconf = fragconf * len(self.fragmentName)
            self.fragconf = fragconf
        elif len(fragconf) == len(self.fragmentName):
            self.fragconf = fragconf
        else:
            print("Error: fragconf number not match")
            sys.exit(1)
        
        if all(x == 1 for x in self.fragconf):
            self.configurational_bias = False

    def get_mctime(self, mctime):

        mctime = [float(i) for i in mctime if len(i) > 0]
        if len(mctime) != len(self.fragmentName):
            print("Error: mctime number not match")
            sys.exit(1)
        else:
            self.mctime = mctime

    def get_fragconc(self, fragconc):

        fragconc = [float(i) for i in fragconc if len(i) > 0]
        if len(fragconc) != len(self.fragmentName):
            print("Error: fragconc number not match")
            sys.exit(1)
        else:
            self.fragconc = fragconc


    def show_parameters(self):
        print(f"MC steps: {self.mcsteps}")
        print("Solute Name: \t\t",'\t\t'.join(self.fragmentName))
        print("Solute Muex: \t\t",'\t\t'.join([str(i) for i in self.fragmuex]))
        print("Solute Conc: \t\t",'\t\t'.join([str(i) for i in self.fragconc]))
        print("Solute ConfB: \t",'\t\t'.join([str(i) for i in self.fragconf]))
        print("Solute mcTime: \t",'\t\t'.join([str(i) for i in self.mctime]))

