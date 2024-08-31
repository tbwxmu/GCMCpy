#!/bin/bash

# This script is used to test the functionality of the program.
cd ..
pip install .
cd test

gcmc -p 2water.pdb -t 2water.top -n 1 -e 15 -f 100000 -o output.txt

#rm charmm36.ff output.txt 
#rm -rf ../gcmc.egg-info ../build ../gcmc/*.o ../.eggs ../.vscode
