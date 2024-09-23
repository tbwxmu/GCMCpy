/*
    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved
        Mingtian Zhao, Alexander D. MacKerell Jr.
    E-mail:
        zhaomt@outerbanks.umaryland.edu
        alex@outerbanks.umaryland.edu
*/


#include "gcmc.h"

#include "gcmc_energy.h"



extern "C"{
    
        __global__ void Gmove_add(const InfoStruct *Ginfo, AtomArray *GfragmentInfo, residue *GresidueInfo, 
                            Atom *GatomInfo, const float *Ggrid, const float *Gff, const int moveFragType,
                            AtomArray *GTempFrag, Atom *GTempInfo, curandState *d_rng_states) {
                            
            __shared__ InfoStruct SharedInfo;
            __shared__ AtomArray SharedFragmentInfo;

            // Calculate thread ID
            int threadId = numThreadsPerBlock * blockIdx.x + threadIdx.x;
            curandState *rng_states = &d_rng_states[threadId];

            int tid = threadIdx.x;

            // Copy global information to shared memory
            if (threadIdx.x == 0) {
                SharedInfo = Ginfo[0];
            }
            
            // Copy fragment information to shared memory and initialize start residue
            if (threadIdx.x == 1) {
                SharedFragmentInfo = GfragmentInfo[moveFragType];
                SharedFragmentInfo.startRes = -1;
            }

            // Synchronize threads to ensure all data is copied
            __syncthreads();

            // Generate random fragment
            randomFragment(SharedInfo, SharedFragmentInfo, &GTempInfo[blockIdx.x], Ggrid, rng_states);

            __syncthreads();

            // Copy shared fragment information back to global memory
            if (tid == 0)
                GTempFrag[blockIdx.x] = SharedFragmentInfo;

            // Calculate energy
            calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, &GTempInfo[blockIdx.x]);
        }



        __global__ void GupdateAdd(AtomArray *GfragmentInfo, residue *GresidueInfo, 
                            Atom *GatomInfo,
                            AtomArray *GTempFrag, Atom *GTempInfo, const int moveFragType, const int totalNum) {
            
            int tid = threadIdx.x;
            int bid = blockIdx.x;

            __shared__ int bidStartRes;
            __shared__ int bidStartAtom;
            __shared__ int bidAtomNum;

            if (tid == 0 && bid == 0){
                GfragmentInfo[moveFragType].totalNum = totalNum;
            }

            if (GTempInfo[bid].type == -1)
                return;

            if (tid == 1){
                bidStartRes = GfragmentInfo[moveFragType].startRes + GTempInfo[bid].type;
            }
            __syncthreads();
            if (tid == 0){
                bidAtomNum = GresidueInfo[bidStartRes].atomNum;
                bidStartAtom = GresidueInfo[bidStartRes].atomStart;
                GresidueInfo[bidStartRes].position[0] = GTempInfo[bid].position[0];
                GresidueInfo[bidStartRes].position[1] = GTempInfo[bid].position[1];
                GresidueInfo[bidStartRes].position[2] = GTempInfo[bid].position[2];
            }
            __syncthreads();
            for (int i = tid; i < bidAtomNum; i += numThreadsPerBlock) {
                GatomInfo[bidStartAtom + i].position[0] = GTempFrag[bid].atoms[i].position[0];
                GatomInfo[bidStartAtom + i].position[1] = GTempFrag[bid].atoms[i].position[1];
                GatomInfo[bidStartAtom + i].position[2] = GTempFrag[bid].atoms[i].position[2];
                GatomInfo[bidStartAtom + i].type = GTempFrag[bid].atoms[i].type;
                GatomInfo[bidStartAtom + i].charge = GTempFrag[bid].atoms[i].charge;
            }
        }





        bool move_add(const InfoStruct *infoHost, InfoStruct *infoDevice, AtomArray *fragmentInfoHost, AtomArray *fragmentInfoDevice, residue *residueInfoDevice, Atom *atomInfoDevice, const float *gridDevice, const float *ffDevice,
                      const int moveFragType, AtomArray *tempFragDevice, Atom *tempInfoHost, Atom *tempInfoDevice, curandState *rngStatesDevice){

            // Check if the total number of fragments is equal to the maximum number of fragments
            if (fragmentInfoHost[moveFragType].totalNum == fragmentInfoHost[moveFragType].maxNum)
                return false;

            // Get the number of blocks for the CUDA kernel
            const int numBlocks = fragmentInfoHost[moveFragType].confBias;

            // Call the CUDA kernel to add a fragment
            Gmove_add<<<numBlocks, numThreadsPerBlock>>>(infoDevice, fragmentInfoDevice, residueInfoDevice, atomInfoDevice, gridDevice, ffDevice, moveFragType, tempFragDevice, tempInfoDevice, rngStatesDevice);

            // Copy the temporary information from the device to the host
            cudaMemcpy(tempInfoHost, tempInfoDevice, sizeof(Atom)*numBlocks, cudaMemcpyDeviceToHost);

            // Get the period and calculate the number of bars and B
            const float *period = infoHost->cryst;
            const float numBars = period[0] * period[1] * period[2] * fragmentInfoHost[moveFragType].conc * MOLES_TO_MOLECULES;
            const float B = beta * fragmentInfoHost[moveFragType].muex + log(numBars);

            // Initialize the sets for the configuration indices
            std::unordered_set<unsigned int> confIndexUnused;
            std::unordered_set<unsigned int> confIndexUsed;

            // Initialize the vector for the configuration probabilities
            std::vector<double> confProbabilities;
            confProbabilities.resize(numBlocks);

            // Initialize the configuration index and the update flag
            int confIndex;
            bool needUpdate = false;

            // Insert all configuration indices into the unused set
            for (int i = 0; i < numBlocks; i++)
                confIndexUnused.insert(i);
            while (confIndexUnused.size() > 0){
                // Get the first unused configuration index and move it to the used set
                auto it = *confIndexUnused.begin();
                confIndexUsed.clear();
                confIndexUsed.insert(it);
                confIndexUnused.erase(it);

                // Initialize the sum of probabilities and the minimum energy
                double sumP = 0;
                float energyMin = tempInfoHost[it].charge;

                // For each unused configuration index
                for (auto iit = confIndexUnused.begin(); iit != confIndexUnused.end();){
                    // If the distance between the current configuration and the unused configuration is less than or equal to the cutoff
                    if ( distanceP(tempInfoHost[it].position, tempInfoHost[*iit].position, period) <= infoHost->cutoff ){
                        // If the energy of the unused configuration is less than the minimum energy, update the minimum energy
                        if (tempInfoHost[*iit].charge < energyMin)
                            energyMin = tempInfoHost[*iit].charge;

                        // Move the unused configuration index to the used set
                        confIndexUsed.insert(*iit);
                        iit = confIndexUnused.erase(iit);
                    }
                    else
                        iit++;
                }

                // For each used configuration index, calculate the probability and add it to the sum of probabilities
                for (auto iit: confIndexUsed) {
                    confProbabilities[iit] = exp(- beta * (tempInfoHost[iit].charge - energyMin ));
                    sumP += confProbabilities[iit];
                }

                // If the sum of probabilities is zero, set the configuration index to the first used configuration index
                if (sumP == 0) {
                    confIndex = it;
                } else {
                    // Otherwise, normalize the probabilities and generate a random number
                    double confPSum = 0;
                    for (auto iit : confIndexUsed)
                    {
                        // confPSum += confProbabilities[iit] / sumP; // confPSum += why not  confPSum =
                        // confProbabilities[iit] = confPSum;
                        confProbabilities[iit] = confProbabilities[iit] / sumP;
                    }
                    float ran = (float)rand() / (float)RAND_MAX;

                    // Find the configuration index that corresponds to the random number
                    for (auto iit : confIndexUsed){
                        confIndex = iit;

                        if (ran < confProbabilities[iit] ){				
                            break;
                        }
                    }
                }

                // Calculate the new energy and the probability of the configuration index
                float energyNew = tempInfoHost[confIndex].charge;
                confProbabilities[confIndex] =  exp(-beta * (tempInfoHost[confIndex].charge - energyMin )) / sumP;

                // Calculate the temporary factor and the acceptance probability
                float fnTmp = infoHost->cavityFactor / ( confIndexUsed.size() * confProbabilities[confIndex] );
                float diff = energyNew;
                float n =  fragmentInfoHost[moveFragType].totalNum;
                float p = Min(1, fnTmp / (n + 1) * exp(B - beta * diff));

                // Generate a random number
                float ran = (float) rand() / (float)RAND_MAX;// why need two time rando number??

                // If the random number is less than the acceptance probability
                if (ran < p)
                {
                    // For each unused configuration index
                    for (auto iit = confIndexUnused.begin();iit != confIndexUnused.end();){
                        // If the distance between the current configuration and the unused configuration is less than or equal to the cutoff, remove the unused configuration index
                        if (distanceP(tempInfoHost[confIndex].position, tempInfoHost[*iit].position, period) <= infoHost->cutoff){
                            iit = confIndexUnused.erase(iit);
                        }
                        else
                            iit ++ ;
                    }

                    // If the total number of fragments is less than the maximum number of fragments, update the type of the current configuration and increase the total number of fragments
                    if (fragmentInfoHost[moveFragType].totalNum < fragmentInfoHost[moveFragType].maxNum){
                        tempInfoHost[confIndex].type = fragmentInfoHost[moveFragType].totalNum;
                        fragmentInfoHost[moveFragType].totalNum += 1;
                        needUpdate = true;
                        // char tempName[5];  
                        // strncpy(tempName, fragmentInfoHost[moveFragType].name, 4);  
                        // tempName[4] = '\0'; 
                        // printf("Fragment %4s inserted. fnTmp %f / (n %f + 1) * exp(B %f - beta %f * diff %f) = %f\n", tempName, fnTmp, n, B, beta, diff, fnTmp / (n + 1) * exp(B - beta * diff));
                    }
                }
            }

            if (needUpdate){//should not after while loop, this may lead multiple inserts at one time
                // Copy the temporary information from the host to the device
                cudaMemcpy(tempInfoDevice, tempInfoHost, sizeof(Atom)*numBlocks, cudaMemcpyHostToDevice);

                // Call the CUDA kernel to update the added fragment
                GupdateAdd<<<numBlocks, numThreadsPerBlock>>>(fragmentInfoDevice, residueInfoDevice, atomInfoDevice, tempFragDevice, tempInfoDevice, moveFragType, fragmentInfoHost[moveFragType].totalNum);
            }

            // Return whether an update is needed
            return needUpdate;
        
        }
    }