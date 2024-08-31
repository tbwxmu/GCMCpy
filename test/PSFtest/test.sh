#!/bin/bash

# This script is used to test the functionality of the program.
cd ../../
pip install .
cd test/PSFtest

pygcmc -p step1_pdbreader.pdb -t step1_pdbreader.top -o output.txt -e 1024 -n 1000 

rm charmm36.ff output.txt 
rm -rf ../../pyGCMC.egg-info ../../build ../../gcmc/*.o ../../.eggs ../../.vscode
