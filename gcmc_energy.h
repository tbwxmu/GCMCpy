#include "gcmc.h"

#ifndef GCMC_ENERGY_H_
#define GCMC_ENERGY_H_

extern "C"{

        __device__ __host__ inline float Min(float a,float b)
        {
            return !(b<a)?a:b;	
        }

        // Device kernel function
        __device__ inline void rotate_atoms(Atom *atoms, int num_atoms, float axis[3], float angle) {
            // Normalize the axis vector
            float norm = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            axis[0] /= norm;
            axis[1] /= norm;
            axis[2] /= norm;

            // Convert the angle to radians and multiply by 2*PI so it rotates between 0 and 2*PI
            // angle *= 2 * M_PI;

            int idx = threadIdx.x;

            // Rotate the atoms with number less than 128

            if (idx < num_atoms) {
                float atom_position[3] = {atoms[idx].position[0], atoms[idx].position[1], atoms[idx].position[2]};

                // Compute the rotation matrix
                float c = cos(angle);
                float s = sin(angle);
                float C = 1 - c;
                float R[3][3];

                R[0][0] = axis[0] * axis[0] * C + c;
                R[0][1] = axis[0] * axis[1] * C - axis[2] * s;
                R[0][2] = axis[0] * axis[2] * C + axis[1] * s;

                R[1][0] = axis[1] * axis[0] * C + axis[2] * s;
                R[1][1] = axis[1] * axis[1] * C + c;
                R[1][2] = axis[1] * axis[2] * C - axis[0] * s;

                R[2][0] = axis[2] * axis[0] * C - axis[1] * s;
                R[2][1] = axis[2] * axis[1] * C + axis[0] * s;
                R[2][2] = axis[2] * axis[2] * C + c;

                // Apply the rotation matrix
                atoms[idx].position[0] = atom_position[0] * R[0][0] + atom_position[1] * R[0][1] + atom_position[2] * R[0][2];
                atoms[idx].position[1] = atom_position[0] * R[1][0] + atom_position[1] * R[1][1] + atom_position[2] * R[1][2];
                atoms[idx].position[2] = atom_position[0] * R[2][0] + atom_position[1] * R[2][1] + atom_position[2] * R[2][2];
            }
        }

        // Device kernel function
        __device__ inline void rotate_atoms_shared(Atom *atoms, int num_atoms, float axis[3], float angle) {

            // Declare shared memory
            __shared__ float sh_axis[3];
            __shared__ float sh_R[3][3];

            int idx = threadIdx.x;

            // Assign axis to shared memory
            if (idx < 3) {
                sh_axis[idx] = axis[idx];
            }

            // Normalize the axis vector
            if (idx == 0) {
                float norm = sqrt(sh_axis[0] * sh_axis[0] + sh_axis[1] * sh_axis[1] + sh_axis[2] * sh_axis[2]);
                sh_axis[0] /= norm;
                sh_axis[1] /= norm;
                sh_axis[2] /= norm;

                // Compute the rotation matrix
                float c = cos(angle);
                float s = sin(angle);
                float C = 1 - c;

                sh_R[0][0] = sh_axis[0] * sh_axis[0] * C + c;
                sh_R[0][1] = sh_axis[0] * sh_axis[1] * C - sh_axis[2] * s;
                sh_R[0][2] = sh_axis[0] * sh_axis[2] * C + sh_axis[1] * s;
 
                sh_R[1][0] = sh_axis[1] * sh_axis[0] * C + sh_axis[2] * s;
                sh_R[1][1] = sh_axis[1] * sh_axis[1] * C + c;
                sh_R[1][2] = sh_axis[1] * sh_axis[2] * C - sh_axis[0] * s;

                sh_R[2][0] = sh_axis[2] * sh_axis[0] * C - sh_axis[1] * s;
                sh_R[2][1] = sh_axis[2] * sh_axis[1] * C + sh_axis[0] * s;
                sh_R[2][2] = sh_axis[2] * sh_axis[2] * C + c;
            }

            __syncthreads(); // Ensure that the shared memory has been initialized before continuing

            // Rotate the atoms 
            for (int i = idx; i < num_atoms; i += blockDim.x) {
                float atom_position[3] = {atoms[i].position[0], atoms[i].position[1], atoms[i].position[2]};
                
                // Apply the rotation matrix
                atoms[i].position[0] = atom_position[0] * sh_R[0][0] + atom_position[1] * sh_R[0][1] + atom_position[2] * sh_R[0][2];
                atoms[i].position[1] = atom_position[0] * sh_R[1][0] + atom_position[1] * sh_R[1][1] + atom_position[2] * sh_R[1][2];
                atoms[i].position[2] = atom_position[0] * sh_R[2][0] + atom_position[1] * sh_R[2][1] + atom_position[2] * sh_R[2][2];
            }
            // // Rotate the atoms with number less than 128
            // if (idx < num_atoms) {
            //     float atom_position[3] = {atoms[idx].position[0], atoms[idx].position[1], atoms[idx].position[2]};
                
            //     // Apply the rotation matrix
            //     atoms[idx].position[0] = atom_position[0] * sh_R[0][0] + atom_position[1] * sh_R[0][1] + atom_position[2] * sh_R[0][2];
            //     atoms[idx].position[1] = atom_position[0] * sh_R[1][0] + atom_position[1] * sh_R[1][1] + atom_position[2] * sh_R[1][2];
            //     atoms[idx].position[2] = atom_position[0] * sh_R[2][0] + atom_position[1] * sh_R[2][1] + atom_position[2] * sh_R[2][2];
            // }
        }

        __device__ void randomFragment(const InfoStruct &SharedInfo, AtomArray &SharedFragmentInfo, Atom *GTempInfo, const float *Ggrid, curandState *rng_states) {

            int tid = threadIdx.x;

            __shared__ float randomR[3];
            __shared__ float randomThi[3];
            __shared__ float randomPhi;
            __shared__ int gridN;

            if (tid < 3){
                randomR[tid] = curand_uniform(rng_states) * SharedInfo.grid_dx;
            }
            if (tid >= 3 && tid < 6){
                randomThi[tid - 3] = curand_uniform(rng_states);
            }
            if (tid == 6){
                randomPhi = curand_uniform(rng_states) * 2 * PI;
            }
            if (tid == 7){
                gridN = curand(rng_states) % SharedInfo.totalGridNum;
            }

            __syncthreads();
            if (tid < 3){
                randomR[tid] += Ggrid[gridN * 3 + tid];
            }

            __syncthreads();

            
            rotate_atoms_shared(SharedFragmentInfo.atoms, SharedFragmentInfo.num_atoms, randomThi, randomPhi);

            __syncthreads();


            for (int i= tid ; i < SharedFragmentInfo.num_atoms; i += blockDim.x){
                SharedFragmentInfo.atoms[i].position[0] += randomR[0];
                SharedFragmentInfo.atoms[i].position[1] += randomR[1];
                SharedFragmentInfo.atoms[i].position[2] += randomR[2];
            }


            if (tid < 3){
                GTempInfo->position[tid] = randomR[tid];
            }
            if (tid == 4)
                GTempInfo->type = -1;




        }

        __device__ __host__ inline float fast_round(const float a) {
            return a >= 0 ? (int)(a + 0.5f) : (int)(a - 0.5f);
        }

        __device__ __host__ inline float distanceP(const float x[3], const float y[3], const float period[3]){
            float dx = x[0] - y[0];
            float dy = x[1] - y[1];
            float dz = x[2] - y[2];

            dx -= fast_round(dx / period[0]) * period[0];
            dy -= fast_round(dy / period[1]) * period[1];
            dz -= fast_round(dz / period[2]) * period[2];

            return sqrtf(dx * dx + dy * dy + dz * dz);
        }

        
        //If NBFIX entry exists for the pair, then this calculation
        //!V(Lennard-Jones) = 4*Eps,i,j[(sigma,i,j/ri,j)**12 - (sigma,i,j/ri,j)**6]
        // sigma and Eps are the nbfix entries
        // units from force field: sigma (nm), epsilon (kJ)

        __device__ inline float calc_vdw_nbfix (float sigma, float epsilon, float dist_sqrd)
        {
            // convert sigma from nm to Angstroms to match dist
            // sigma *= 10;

            float sigma_sqrd = sigma * sigma * 100;
            float sigma_dist_sqrd = sigma_sqrd / dist_sqrd;
            float E_vdw = 4*epsilon * ( pow(sigma_dist_sqrd, 6) - pow(sigma_dist_sqrd, 3) );

            //cout << "(calc_vdw_nbfix) epsilon: " << epsilon << " sigma_sqrd: " << sigma_sqrd << " dist_sqrd: " << dist_sqrd << " E_vdw: " << E_vdw << endl;

            return E_vdw;
        }

        
        // E_elec = (1/4*pi*eps_0)*(q1*q2/d)
        // units from force field: charge (e)
        // units from this program: distance (A)
        // eps_0 = 8.85e-12 (C^2/J*m)
        // kel = 1/(4*pi*eps_0)
        //     = 8.99e9 (J*m/C^2)
        //     = 1389.3 (kJ/mol * A/e^2)
        //     = 332.05 (kcal/mol * A/e^2)
        // from http://users.mccammon.ucsd.edu/~blu/Research-Handbook/physical-constant.html
        __device__ inline float calc_elec (float charge1, float charge2, float dist)
        {
            // float kel = 331.843 * KCAL_TO_KJ; //1389.3;  what's 331.843?

            // float E_elec = kel * (charge1 * charge2) / dist / eps;
            // float E_elec = 1389.3 * (charge1 * charge2) / dist;
            // E_elec = 0;


            // Differences in the electrostatic energies:
            //  
            // (*) The conversion from charge units to kcal/mol in CHARMM is based 
            // on the value 332.0716 whereas AMBER uses 18.2223**2 or 332.0522173.
            // The actual value is somewhat lower than both these values
            // (~331.843)!  To convert the charges to "CHARMM", they should be
            // multiplied by 1.000058372.  This was not done within this file.
            // [When this is done, the charges are not rounded and therefore
            // non-integral charges for the residues are apparent.]  To get around
            // this problem either the charges can be scaled within CHARMM (which
            // will still lead to non-integral charge) or in versions of CHARMM
            // beyond c25n3, and through the application of the "AMBER" keyword in
            // pref.dat, the AMBER constant can be used.  By default, the "fast"
            // routines cannot be used with the AMBER-style impropers.  In the
            // later versions of CHARMM, the AMBER keyword circumvents this
            // problem.
            // 
            // Ref: https://home.chpc.utah.edu/~cheatham/cornell_rtf


            float E_elec = 1388.431112 * (charge1 * charge2) / dist;
            return E_elec;
        }


        __device__ inline void calcProtEnergy(const InfoStruct SharedInfo, AtomArray &SharedFragmentInfo , AtomArray *GfragmentInfo,
                                    residue *GresidueInfo, Atom *GatomInfo, const float *Gff, Atom *GTempInfo, float *sh_energy){
            
            int tid = threadIdx.x;
            __shared__ int maxResidueNum;
            if (tid == 0)
                maxResidueNum = GfragmentInfo->startRes;
            
            
            __syncthreads();

            for (int resi = tid;resi < maxResidueNum; resi+= numThreadsPerBlock){

                // if (distanceP(GTempInfo->position, GresidueInfo[resi].position, SharedInfo.cryst) > SharedInfo.cutoff) {
                if (distanceP(GTempInfo->position, GresidueInfo[resi].position, SharedInfo.cryst) > SharedInfo.cutoff) {
                    continue;
                }

                int resiStart = GresidueInfo[resi].atomStart;
                int resiEnd = GresidueInfo[resi].atomStart + GresidueInfo[resi].atomNum;
                float resiEnergy = 0;
                for (int atomi = resiStart; atomi < resiEnd; atomi++){
                    // int atomType = GatomInfo[atomi].type;
                    // float atomCharge = GatomInfo[atomi].charge;
                    // float atomEnergy = 0;
                    for (int atomj = 0; atomj < SharedFragmentInfo.num_atoms; atomj++){

                        float distance = distanceP(GatomInfo[atomi].position, SharedFragmentInfo.atoms[atomj].position, SharedInfo.cryst);
                        int typeij = SharedFragmentInfo.atoms[atomj].type * SharedInfo.ffYNum + GatomInfo[atomi].type;
                        // float sigma = Gff[typeij * 2];
                        // float epsilon = Gff[typeij * 2 + 1];
                        resiEnergy += calc_vdw_nbfix(Gff[typeij * 2], Gff[typeij * 2 + 1], distance * distance);
                        resiEnergy += calc_elec(SharedFragmentInfo.atoms[atomj].charge, GatomInfo[atomi].charge, distance);
                        // float sigma = Gff[typeij * 2];
                        // float energy = Gff[atomType * SharedInfo.atomTypeNum + SharedFragmentInfo.atoms[atomj].type] * atomCharge * SharedFragmentInfo.atoms[atomj].charge / distance;
                        // resiEnergy += energy;
                    }
                    // resiEnergy += atomEnergy;
                }
                sh_energy[tid] += resiEnergy;
            }


        }


        __device__ inline void calcFragEnergy(const InfoStruct SharedInfo, AtomArray &SharedFragmentInfo , AtomArray *GfragmentInfo,
                                    residue *GresidueInfo, Atom *GatomInfo, const float *Gff, Atom *GTempInfo, float *sh_energy){
            
            int tid = threadIdx.x;
            __shared__ int startResidueNum;
            __shared__ int endResidueNum;


            for (int fragType = 0; fragType < SharedInfo.fragTypeNum; fragType++){

                if (tid == 0){
                    startResidueNum = GfragmentInfo[fragType].startRes;
                    endResidueNum = GfragmentInfo[fragType].startRes + GfragmentInfo[fragType].totalNum;
                }

                __syncthreads();

                for (int resi = tid + startResidueNum;resi < endResidueNum; resi+= numThreadsPerBlock){

                    if (resi == SharedFragmentInfo.startRes)
                        continue;

                    // if (distanceP(GTempInfo->position, GresidueInfo[resi].position, SharedInfo.cryst) > SharedInfo.cutoff) {
                    if (distanceP(GTempInfo->position, GresidueInfo[resi].position, SharedInfo.cryst) > SharedInfo.cutoff) {
                        continue;
                    }

                    int resiStart = GresidueInfo[resi].atomStart;
                    int resiEnd = GresidueInfo[resi].atomStart + GresidueInfo[resi].atomNum;
                    float resiEnergy = 0;
                    for (int atomi = resiStart; atomi < resiEnd; atomi++){
                        // int atomType = GatomInfo[atomi].type;
                        // float atomCharge = GatomInfo[atomi].charge;
                        // float atomEnergy = 0;
                        for (int atomj = 0; atomj < SharedFragmentInfo.num_atoms; atomj++){

                            float distance = distanceP(GatomInfo[atomi].position, SharedFragmentInfo.atoms[atomj].position, SharedInfo.cryst);
                            int typeij = SharedFragmentInfo.atoms[atomj].type * SharedInfo.ffYNum + GatomInfo[atomi].type;
                            // float sigma = Gff[typeij * 2];
                            // float epsilon = Gff[typeij * 2 + 1];
                            resiEnergy += calc_vdw_nbfix(Gff[typeij * 2], Gff[typeij * 2 + 1], distance * distance);
                            resiEnergy += calc_elec(SharedFragmentInfo.atoms[atomj].charge, GatomInfo[atomi].charge, distance);
                            // float sigma = Gff[typeij * 2];
                            // float energy = Gff[atomType * SharedInfo.atomTypeNum + SharedFragmentInfo.atoms[atomj].type] * atomCharge * SharedFragmentInfo.atoms[atomj].charge / distance;
                            // resiEnergy += energy;
                        }
                        // resiEnergy += atomEnergy;
                    }
                    sh_energy[tid] += resiEnergy;
                }
                
                __syncthreads();
            }
            


        }
        

        __device__ void calcEnergy(const InfoStruct SharedInfo, AtomArray &SharedFragmentInfo , AtomArray *GfragmentInfo,
                                    residue *GresidueInfo, Atom *GatomInfo, const float *Gff, Atom *GTempInfo){

            __shared__ float sh_energy[numThreadsPerBlock];

            int tid = threadIdx.x;

            sh_energy[tid] = 0;

            __syncthreads();

            calcProtEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo, sh_energy);
            
            __syncthreads();

            calcFragEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, GTempInfo, sh_energy);

            __syncthreads();

            for (int s = numThreadsPerBlock / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sh_energy[tid] += sh_energy[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0){

                GTempInfo->charge = sh_energy[0];

            }

        }

    }

#endif