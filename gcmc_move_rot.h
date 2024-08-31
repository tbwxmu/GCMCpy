#include "gcmc.h"
#include "gcmc_energy.h"

extern "C"{

        __global__ void Gmove_rot(const InfoStruct *Ginfo, AtomArray *GfragmentInfo, residue *GresidueInfo, 
                Atom *GatomInfo, const float *Ggrid, const float *Gff, const int moveFragType,
                AtomArray *GTempFrag, Atom *GTempInfo, curandState *d_rng_states) {
            __shared__ InfoStruct SharedInfo;
            __shared__ AtomArray SharedFragmentInfo;

            __shared__ float energyTemp;

            __shared__ float randomR[3];

            int threadId = numThreadsPerBlock * blockIdx.x + threadIdx.x;
            curandState *rng_states = &d_rng_states[threadId];


            int tid = threadIdx.x;

            if (threadIdx.x == 0) {
                SharedInfo = Ginfo[0];
            }
            
            if (threadIdx.x == 1) {
                SharedFragmentInfo = GfragmentInfo[moveFragType];
                SharedFragmentInfo.startRes = GTempInfo[blockIdx.x].type + GfragmentInfo[moveFragType].startRes;
                GTempInfo[blockIdx.x].position[0] = GresidueInfo[SharedFragmentInfo.startRes].position[0];
                GTempInfo[blockIdx.x].position[1] = GresidueInfo[SharedFragmentInfo.startRes].position[1];
                GTempInfo[blockIdx.x].position[2] = GresidueInfo[SharedFragmentInfo.startRes].position[2];

            }

            __syncthreads();

            for (int i=tid;i<SharedFragmentInfo.num_atoms;i+=numThreadsPerBlock){
                SharedFragmentInfo.atoms[i].position[0] = GatomInfo[GresidueInfo[SharedFragmentInfo.startRes].atomStart + i].position[0];
                SharedFragmentInfo.atoms[i].position[1] = GatomInfo[GresidueInfo[SharedFragmentInfo.startRes].atomStart + i].position[1];
                SharedFragmentInfo.atoms[i].position[2] = GatomInfo[GresidueInfo[SharedFragmentInfo.startRes].atomStart + i].position[2];

            }
            
            __syncthreads();


            // if (SharedInfo.PME == 0)
                calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, &GTempInfo[blockIdx.x]);

            __syncthreads();

            if (tid == 3){
                energyTemp = GTempInfo[blockIdx.x].charge;
            }

            __shared__ float randomThi[3];
            __shared__ float randomPhi;
            __shared__ float center[3];


            if (tid < 3){
                randomThi[tid] = curand_uniform(rng_states);
                center[tid] = 0;

            }
            if (tid == 3){
                randomPhi = curand_uniform(rng_states) * 2 * PI;
            }

            __syncthreads();

            if (tid < 3){
                for (int i=0;i<SharedFragmentInfo.num_atoms;i++){
                    center[tid] += SharedFragmentInfo.atoms[i].position[tid] / SharedFragmentInfo.num_atoms;
                }
            }


            __syncthreads();

            for (int i=tid;i<SharedFragmentInfo.num_atoms;i+=numThreadsPerBlock){
                SharedFragmentInfo.atoms[i].position[0] -= center[0];
                SharedFragmentInfo.atoms[i].position[1] -= center[1];
                SharedFragmentInfo.atoms[i].position[2] -= center[2];

            }
            __syncthreads();

            rotate_atoms_shared(SharedFragmentInfo.atoms, SharedFragmentInfo.num_atoms, randomThi, randomPhi);


            __syncthreads();

            for (int i=tid;i<SharedFragmentInfo.num_atoms;i+=numThreadsPerBlock){
                SharedFragmentInfo.atoms[i].position[0] += center[0];
                SharedFragmentInfo.atoms[i].position[1] += center[1];
                SharedFragmentInfo.atoms[i].position[2] += center[2];

            }


            __syncthreads();



            calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, &GTempInfo[blockIdx.x]);

            __syncthreads();

            if (tid == 3){
                GTempInfo[blockIdx.x].charge = GTempInfo[blockIdx.x].charge - energyTemp;
            }


            
            if (tid == 0)
                GTempFrag[blockIdx.x] = SharedFragmentInfo;

                

        }

        __global__ void GupdateRot(AtomArray *GfragmentInfo, residue *GresidueInfo, 
                            Atom *GatomInfo,
                            AtomArray *GTempFrag, Atom *GTempInfo, const int moveFragType) {
            
            int tid = threadIdx.x;
            int bid = blockIdx.x;

            __shared__ int bidStartRes;
            __shared__ int bidStartAtom;
            __shared__ int bidAtomNum;



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
            }
        }


        bool move_rot(const InfoStruct *infoHost, InfoStruct *infoDevice, AtomArray *fragmentInfoHost, AtomArray *fragmentInfoDevice, residue *residueInfoDevice, Atom *atomInfoDevice, const float *gridDevice, const float *ffDevice,
                      const int moveFragType, AtomArray *tempFragDevice, Atom *tempInfoHost, Atom *tempInfoDevice, curandState *rngStatesDevice){

            // Determine the number of blocks
            const int numBlocks = min(fragmentInfoHost[moveFragType].confBias, fragmentInfoHost[moveFragType].totalNum);

            if (numBlocks == 0){
                return false;
            }

            // Initialize a vector with fragment indices
            std::vector<int> nums(fragmentInfoHost[moveFragType].totalNum);
            for (int i = 0; i < fragmentInfoHost[moveFragType].totalNum; ++i) {
                nums[i] = i;
            }

            // Shuffle the vector
            for (int i = 0; i < numBlocks; ++i) {
                int j = i + rand() % (fragmentInfoHost[moveFragType].totalNum - i);
                std::swap(nums[i], nums[j]);
            }

            // Assign shuffled indices to TempInfo
            for (int i=0;i < numBlocks;i++){
                tempInfoHost[i].type = nums[i];
            }

            // Copy TempInfo to device memory
            cudaMemcpy(tempInfoDevice, tempInfoHost, sizeof(Atom)*numBlocks, cudaMemcpyHostToDevice);

            // Call the CUDA kernel to perform translation move
            Gmove_rot<<<numBlocks, numThreadsPerBlock>>>(infoDevice, fragmentInfoDevice, residueInfoDevice, atomInfoDevice, gridDevice, ffDevice, moveFragType, tempFragDevice, tempInfoDevice, rngStatesDevice);

            // Copy TempInfo back to host memory
            cudaMemcpy(tempInfoHost, tempInfoDevice, sizeof(Atom)*numBlocks, cudaMemcpyDeviceToHost);


            
            // Get the period and calculate the number of bars and B
            const float *period = infoHost->cryst;
            // const float numBars = period[0] * period[1] * period[2] * fragmentInfoHost[moveFragType].conc * MOLES_TO_MOLECULES;
            // const float B = beta * fragmentInfoHost[moveFragType].muex + log(numBars);

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
                        confPSum += confProbabilities[iit] / sumP;
                        confProbabilities[iit] = confPSum;
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
                float fnTmp = 1 / ( confIndexUsed.size() * confProbabilities[confIndex] );
                float diff = energyNew;
                // float n =  fragmentInfoHost[moveFragType].totalNum;
                float p = Min(1, fnTmp * exp( - beta * diff));

                // Generate a random number
                float ran = (float) rand() / (float)RAND_MAX;

                int tempInfoHostType = tempInfoHost[confIndex].type;

                for (auto iit = confIndexUsed.begin();iit != confIndexUsed.end();){
                    tempInfoHost[*iit].type = -1;
                    iit ++ ;

                }

                // If the random number is less than the acceptance probability
                if (ran < p)
                {
                    // For each unused configuration index
                    for (auto iit = confIndexUnused.begin();iit != confIndexUnused.end();){
                        // If the distance between the current configuration and the unused configuration is less than or equal to the cutoff, remove the unused configuration index
                        if (distanceP(tempInfoHost[confIndex].position, tempInfoHost[*iit].position, period) <= infoHost->cutoff){
                            tempInfoHost[*iit].type = -1;
                            iit = confIndexUnused.erase(iit);
                        }
                        else
                            iit ++ ;
                    }



                    tempInfoHost[confIndex].type = tempInfoHostType;

                    // Set the update flag to true
                    needUpdate = true;

                  
                }
            }

            // printf("Rotate move %d done\n", moveFragType);

            if (needUpdate){
                // Copy the temporary information from the host to the device
                cudaMemcpy(tempInfoDevice, tempInfoHost, sizeof(Atom)*numBlocks, cudaMemcpyHostToDevice);

                // Call the CUDA kernel to update the added fragment
                GupdateRot<<<numBlocks, numThreadsPerBlock>>>(fragmentInfoDevice, residueInfoDevice, atomInfoDevice, tempFragDevice, tempInfoDevice, moveFragType);
            }

            // Return whether an update is needed
            return needUpdate;
        }
    }