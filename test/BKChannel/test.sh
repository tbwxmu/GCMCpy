#!/bin/bash

# This script is used to test the functionality of the program.
export OPWD=`pwd`
cd ../..
pip install .
cd $OPWD
#for ((i=0;i<100;i++))
#do
#echo "seed = $i" &>> tmp~
pygcmc -p 6v3g_silcs.1.pdb -t 6v3g_silcs.1.top -o output.txt -e 51 -n 1 
# gcmc -p 6v3g_silcs.1.pdb -t 6v3g_silcs.1.top -o output.txt -e 54 -n 1
#done

rm charmm36.ff output.txt 
cd ../
rm -rf ../gcmc.egg-info ../build ../gcmc/*.o ../.eggs ../.vscode
