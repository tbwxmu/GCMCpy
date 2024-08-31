"""

    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved    
    	Mingtian Zhao, Alexander D. MacKerell Jr.        
    E-mail: 
    	zhaomt@outerbanks.umaryland.edu
    	alex@outerbanks.umaryland.edu

"""

from gcmc import GCMC
from packages import *

def main():
    # file_output = open('Analyze_output.txt', 'w')
    # original_output = sys.stdout
    # sys.stdout = Tee(sys.stdout, file_output)

    startTime = time.time()
    print('Start GCMC simulation at %s...' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    

    parser = argparse.ArgumentParser(description="pyGCMC - A python package for GCMC simulation")

    parser.add_argument(
        "-p",
        "--pdb-file",
        dest="pdb_file",
        required=True,
        help="The file .pdb for GCMC",
        metavar="file.pdb",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--top-file",
        dest="top_file",
        required=False,
        help="The file .top for GCMC",
        metavar="file.top",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--psf-file",
        dest="psf_file",
        required=False,
        help="The file .psf for GCMC",
        metavar="file.psf",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--out-file",
        dest="out_file",
        required=False,
        help="The output file for GCMC",
        metavar="file.txt",
        type=str,
    )
    parser.add_argument(
        "-u",
        "--fragmuex",
        dest="fragmuex",
        required=False,
        help="The value of solute muex(splice by , with no space), if the first value is negative, then follow the -u or --fragmuex without space",
        metavar="muex1,muex2,...",
        type=str,
    ) 
    parser.add_argument(
        "-f",
        "--fragconf",
        dest="fragconf",
        required=False,
        help="The value of solute conf(splice by , with no space). Or only one value for all solutes",
        metavar="conf1,conf2,... or conf",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--mcsteps",
        dest="mcsteps",
        required=False,
        help="The number of MC steps",
        metavar="mcsteps",
        type=int,
    )
    parser.add_argument(
        "-m",
        "--mctime",
        dest="mctime",
        required=False,
        help="The mctime of solutes(splice by , with no space)",
        metavar="mctime1,mctime2,...",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--fragconc",
        dest="fragconc",
        required=False,
        help="The value of solute concentration(splice by , with no space)",
        metavar="conc1,conc2,...",
        type=str,
    )

    parser.add_argument(
        "-y",
        "--cavitybias-dx",
        dest="cavity_bias_dx",
        required=False,
        help="The value of cavity bias dx(if dx <= 0, then no cavity bias)",
        metavar="cavity_bias_dx",
        type=float,
    )

    parser.add_argument(
        "-e",
        "--seed",
        dest="seed",
        required=False,
        help="The seed of random number",
        metavar="seed",
        type=int,
    )

    parser.add_argument(
        "-P",
        "--PME",
        dest="PME",
        required=False,
        help="Enable PME(Default: Disable)",
        action='store_true',
    )


    parser.add_argument(
        "-w",
        "--show-info",
        dest="show_info",
        required=False,
        help="Show the information of solutes",
        action='store_true',
    )

    parser.add_argument(
        "-h",
        "--show-info",
        dest="show_info",
        required=False,
        help="Show the help information ",
        action='store_true',
    )

    args = parser.parse_args()
    
    gcmc = GCMC()

    pdb_file = args.pdb_file
    top_file = args.top_file
    out_file = args.out_file
    psf_file = args.psf_file

    if out_file is not None:
        file_output = open(out_file, 'w')
        original_output = sys.stdout
        sys.stdout = Tee(sys.stdout, file_output)
        print(f"Using output file: {out_file}")

    if args.fragmuex is not None:
        fragmuex = args.fragmuex
        fragmuex = fragmuex.split(',')
        gcmc.get_fragmuex(fragmuex)
        print(f"Using solute muex: {fragmuex}")    

    if args.fragconf is not None:
        fragconf = args.fragconf
        fragconf = fragconf.split(',')
        gcmc.get_fragconf(fragconf)
        print(f"Using solute conf: {fragconf}")

    if args.mcsteps is not None:
        gcmc.mcsteps = args.mcsteps
        print(f"Using MC steps: {args.mcsteps}")

    if args.mctime is not None:
        mctime = args.mctime
        mctime = mctime.split(',')
        gcmc.get_mctime(mctime)
        print(f"Using solute mctime: {mctime}")

    if args.fragconc is not None:
        fragconc = args.fragconc
        fragconc = fragconc.split(',')
        gcmc.get_fragconc(fragconc)
        print(f"Using solute concentration: {fragconc}")
    
    if args.cavity_bias_dx is not None:
        gcmc.grid_dx = args.cavity_bias_dx
        if gcmc.grid_dx <= 0:
            gcmc.cavity_bias = False
            print(f"No cavity bias")
        else:
            print(f"Using cavity bias dx: {args.cavity_bias_dx}")

    if args.seed is not None:
        gcmc.seed = args.seed
        random.seed(gcmc.seed)
        print(f"Using seed: {args.seed}")

    if args.PME:
        gcmc.PME = True
        print(f"Using PME")
    
    if args.show_info:
        gcmc.showInfo = True
        print(f"Showing the information of solutes")

    gcmc.get_pdb(pdb_file)

        
    
    



    if top_file is not None:
        gcmc.get_top(top_file)
    
    if psf_file is not None:
        gcmc.get_psf(psf_file)


    
    gcmc.get_simulation()


    endTime = time.time()
    print(f"Python time used: {endTime - startTime} s")


    # Strat GPU GCMC simulation
    gcmc.run()


    gcmc.get_data()


    # Write the output pdb file
    outPDBName = pdb_file.rsplit('.', 1)[0] + '_out.pdb'

    open(outPDBName, 'w').write(gcmc.PDBString)

    
    print('GCMC simulation finished at %s...' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    

    endTime = time.time()
    print(f"Time used: {endTime - startTime} s")


    # if top_file is not None:
    #     try:
    #         # top = protein_data.read_top(top_file)
    #         protein_data.read_top(top_file)
    #     except:
    #         print(f"Error reading top file: {top_file}")
    #         sys.exit(1)
    #     # print(f"top atom number: {len(top)}")

    if out_file is not None:
        sys.stdout = original_output
        file_output.close()


