#!/bin/bash

# This script is used to test the functionality of the program.
cd ../../
pip install .
cd test/PSFtest

pygcmc -p step1_pdbreader.pdb -s step1_pdbreader.psf -o output.txt -e 1024 -w

rm charmm36.ff output.txt 
rm -rf ../../pyGCMC.egg-info ../../build ../../gcmc/*.o ../../.eggs ../../.vscode
