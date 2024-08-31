#!/bin/bash

# This script is used to test the functionality of the program.
cd ..
pip install .
cd test

gcmc -p 6v3g_silcs.1.pdb -t 6v3g_silcs.1.top -n 1 -e 15 -o output.txt

#rm charmm36.ff output.txt 
#rm -rf ../gcmc.egg-info ../build ../gcmc/*.o ../.eggs ../.vscode
