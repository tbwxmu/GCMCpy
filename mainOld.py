"""

    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved    
    	Mingtian Zhao, Alexander D. MacKerell Jr.        
    E-mail: 
    	zhaomt@outerbanks.umaryland.edu
    	alex@outerbanks.umaryland.edu

"""

from .gcmc import GCMC
from .packages import *


def main():
    # file_output = open('Analyze_output.txt', 'w')
    # original_output = sys.stdout
    # sys.stdout = Tee(sys.stdout, file_output)


    startTime = time.time()
    
    


    parser = argparse.ArgumentParser(description='Perform GCMC Simulation')

    parser.add_argument('-p', '--paramfile', type=str, required=True, 
                        help='[Required] input parameter file')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='[Optional] verbose output')
    parser.add_argument('--logfile', type=str, 
                        help='[Optional] log file, if not specified, then output will be stdout')
    parser.add_argument('--debug', action='store_true', 
                        help='[Optional] for debug purpose')
    parser.add_argument('--version', action='version', version='GCMC version', 
                        help='Show program\'s version number and exit')
    args = parser.parse_args()



    out_file = args.logfile

    if out_file is not None:
        file_output = open(out_file, 'w')
        original_output = sys.stdout
        sys.stdout = Tee(sys.stdout, file_output)
        # print(f"Using output file: {out_file}")


    print('Start GCMC simulation at %s...' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



    if out_file is not None:
        print(f"Using output file: {out_file}")

    gcmc = GCMC()

    try:
        gcmc.load_parameters(args.paramfile)
    except:
        print(f"Error reading parameter file: {args.paramfile}")
        sys.exit(1)



    pdb_file = gcmc.config_dict['pdb'][0]
    top_file = gcmc.config_dict['top'][0]
    top_out_file = gcmc.config_dict['op_top'][0]
    pdb_out_file = gcmc.config_dict['op_pdb'][0]
    print(f"Using pdb file: {pdb_file}")
    print(f"Using top file: {top_file}")
    print(f"Using output pdb file: {pdb_out_file}")
    print(f"Using output top file: {top_out_file}")


    fragName = gcmc.config_dict['fragname'][0].split()
    gcmc.fragmentName = [i.upper() for i in fragName]


    fragmuex = gcmc.config_dict['fragmuex'][0].split()
    gcmc.get_fragmuex(fragmuex)


    fragconf = gcmc.config_dict['fragconf'][0].split()
    gcmc.get_fragconf(fragconf)

    fragconc = gcmc.config_dict['fragconc'][0].split()
    gcmc.get_fragconc(fragconc)

    mctime = gcmc.config_dict['mctime'][0].split()
    gcmc.get_mctime(mctime)

    for i, frag in enumerate(fragName):
        print(f"Fragment {frag}:\tmuex: {gcmc.fragmuex[i]}\tconf: {gcmc.fragconf[i]}\tconc: {gcmc.fragconc[i]}\tmctime: {gcmc.mctime[i]}")

    gcmc.mcsteps = int(gcmc.config_dict['mcsteps'][0])
    print(f"Using MC steps: {gcmc.mcsteps}")

    grid_dx = float(gcmc.config_dict['grid_dx'][0])
    gcmc.grid_dx = grid_dx
    if not gcmc.bool_parameters(gcmc.config_dict.get('use_cavity_bias', ['False'])[0]):
        gcmc.cavity_bias = False
        print(f"No cavity bias")
    else:
        print(f"Using cavity bias dx: {grid_dx}")
    
    if 'random_seed' in gcmc.config_dict:
        gcmc.seed = int(gcmc.config_dict['random_seed'][0])
        random.seed(gcmc.seed)
        print(f"Using seed: {gcmc.seed}")

    if gcmc.bool_parameters(gcmc.config_dict.get('PME', ['False'])[0]):
        gcmc.PME = True
        print(f"Using PME")
    
    gcmc.get_pdb(pdb_file)
    gcmc.get_top(top_file)

    gcmc.get_simulation()

    endTime = time.time()
    print(f"Python time used: {endTime - startTime} s")

    sys.stdout.flush()

    # Start GPU GCMC simulation
    gcmc.run()

    gcmc.get_data()

    open(pdb_out_file, 'w').write(gcmc.PDBString)
    open(top_out_file, 'w').write(gcmc.TOPString)





    
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