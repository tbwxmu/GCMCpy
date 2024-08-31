#!/bin/bash

# This script is used to test the functionality of the program.
cd ../..
pip install .
cd test/5uul
pygcmc -w -p 5uul_proa_silcs.1.prod.25.rec.pdb -t 5uul_proa_silcs.1.gc.25.top -o output.txt -u 28.21,10.77,7.26,-6.16,-11.67,-17.03,-57.31,-92.64,-5.62 -f 400,15,15,10,20,300,200,100,2 -n 50000 -m 0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.800 

rm charmm36.ff output.txt 
cd ..
rm -rf ../pyGCMC.egg-info ../build ../gcmc/*.o ../.eggs ../.vscode ../gcmc/__pycache__
cd 5uul
