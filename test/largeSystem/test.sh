#!/bin/bash

# This script is used to test the functionality of the program.
cd ../../
pip install .
cd test/largeSystem

# pygcmc -p memcrowdatp_cg.pdb -s memcrowdatp_cg.psf -o output.txt -e 1024 -n 10000

gcmc -v -p gcmc.4.inp > gcmc.4.out

rm charmm36.ff 
rm -rf ../../pyGCMC.egg-info ../../build ../../gcmc/*.o ../../.eggs ../../.vscode
