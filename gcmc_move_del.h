/*
    © Copyright 2023 - University of Maryland, Baltimore   All Rights Reserved
        Mingtian Zhao, Alexander D. MacKerell Jr.
    E-mail:
        zhaomt@outerbanks.umaryland.edu
        alex@outerbanks.umaryland.edu
*/

#include "gcmc.h"
#include "gcmc_energy.h"

extern "C"
{

    // Kernel for deleting a fragment in GCMC simulation
    __global__ void Gmove_del(const InfoStruct *Ginfo, AtomArray *GfragmentInfo, residue *GresidueInfo,
                              Atom *GatomInfo, const float *Ggrid, const float *Gff, const int moveFragType,
                              AtomArray *GTempFrag, Atom *GTempInfo, curandState *d_rng_states)
    {
        __shared__ InfoStruct SharedInfo;
        __shared__ AtomArray SharedFragmentInfo;

        int threadId = numThreadsPerBlock * blockIdx.x + threadIdx.x;
        curandState *rng_states = &d_rng_states[threadId];
        int tid = threadIdx.x;

        if (threadIdx.x == 0)
        {
            SharedInfo = Ginfo[0];
        }

        if (threadIdx.x == 1)
        {
            SharedFragmentInfo = GfragmentInfo[moveFragType];
            SharedFragmentInfo.startRes = GTempInfo[blockIdx.x].type + GfragmentInfo[moveFragType].startRes;
            GTempInfo[blockIdx.x].position[0] = GresidueInfo[SharedFragmentInfo.startRes].position[0];
            GTempInfo[blockIdx.x].position[1] = GresidueInfo[SharedFragmentInfo.startRes].position[1];
            GTempInfo[blockIdx.x].position[2] = GresidueInfo[SharedFragmentInfo.startRes].position[2];
        }

        __syncthreads();

        // Copy atom positions to shared memory
        for (int i = tid; i < SharedFragmentInfo.num_atoms; i += numThreadsPerBlock)
        {
            SharedFragmentInfo.atoms[i].position[0] = GatomInfo[GresidueInfo[SharedFragmentInfo.startRes].atomStart + i].position[0];
            SharedFragmentInfo.atoms[i].position[1] = GatomInfo[GresidueInfo[SharedFragmentInfo.startRes].atomStart + i].position[1];
            SharedFragmentInfo.atoms[i].position[2] = GatomInfo[GresidueInfo[SharedFragmentInfo.startRes].atomStart + i].position[2];
        }

        __syncthreads();

        if (tid == 0)
            GTempFrag[blockIdx.x] = SharedFragmentInfo;

        calcEnergy(SharedInfo, SharedFragmentInfo, GfragmentInfo, GresidueInfo, GatomInfo, Gff, &GTempInfo[blockIdx.x]);
    }

    // Kernel for updating the system after deleting a fragment
    __global__ void GupdateDel(AtomArray *GfragmentInfo, residue *GresidueInfo,
                               Atom *GatomInfo, AtomArray *GTempFrag, Atom *GTempInfo,
                               const int moveFragType, const int totalNum, const int conf_index)
    {
        int tid = threadIdx.x;

        __shared__ int bidStartRes;
        __shared__ int bidStartAtom;
        __shared__ int bidAtomNum;
        __shared__ int bidEndRes;
        __shared__ int bidEndAtom;

        if (tid == 0)
        {
            GfragmentInfo[moveFragType].totalNum = totalNum;
        }

        if (totalNum == 0)
            return;

        if (tid == 1)
        {
            bidStartRes = GfragmentInfo[moveFragType].startRes + GTempInfo[conf_index].type; // GTempInfo is temp temp_frags
            bidAtomNum = GresidueInfo[bidStartRes].atomNum;
            bidStartAtom = GresidueInfo[bidStartRes].atomStart;
            bidEndRes = GfragmentInfo[moveFragType].startRes + totalNum;
            bidEndAtom = GresidueInfo[bidEndRes].atomStart;
        }
        __syncthreads();

        if (tid == 0)
        {
            GresidueInfo[bidStartRes].position[0] = GresidueInfo[bidEndRes].position[0];
            GresidueInfo[bidStartRes].position[1] = GresidueInfo[bidEndRes].position[1];
            GresidueInfo[bidStartRes].position[2] = GresidueInfo[bidEndRes].position[2];
        }

        // Update atom positions
        for (int i = tid; i < bidAtomNum; i += numThreadsPerBlock)
        {
            GatomInfo[bidStartAtom + i].position[0] = GatomInfo[bidEndAtom + i].position[0];
            GatomInfo[bidStartAtom + i].position[1] = GatomInfo[bidEndAtom + i].position[1];
            GatomInfo[bidStartAtom + i].position[2] = GatomInfo[bidEndAtom + i].position[2];
        }
    }

    // Main function for deleting a fragment in GCMC simulation
    bool move_del(const InfoStruct *info, InfoStruct *Ginfo, AtomArray *fragmentInfo, AtomArray *GfragmentInfo,
                  residue *GresidueInfo, Atom *GatomInfo, const float *Ggrid, const float *Gff,
                  const int moveFragType, AtomArray *GTempFrag, Atom *TempInfo, Atom *GTempInfo, curandState *d_rng_states)
    {
        const int nBlock = 1;

        if (nBlock == 0)
        {
            return false;
        }

        // Initialize and shuffle fragment indices
        std::vector<int> nums(fragmentInfo[moveFragType].totalNum);
        for (int i = 0; i < fragmentInfo[moveFragType].totalNum; ++i)
        {
            nums[i] = i;
        }

        for (int i = 0; i < nBlock; ++i)
        {
            int j = i + rand() % (fragmentInfo[moveFragType].totalNum - i);
            std::swap(nums[i], nums[j]);
        }

        // Assign shuffled indices to TempInfo
        for (int i = 0; i < nBlock; i++)
        {
            TempInfo[i].type = nums[i];
        }

        // Copy TempInfo to device memory
        cudaMemcpy(GTempInfo, TempInfo, sizeof(Atom) * nBlock, cudaMemcpyHostToDevice);

        // Call the CUDA kernel to perform deletion
        Gmove_del<<<nBlock, numThreadsPerBlock>>>(Ginfo, GfragmentInfo, GresidueInfo, GatomInfo, Ggrid, Gff, moveFragType, GTempFrag, GTempInfo, d_rng_states);

        // Copy TempInfo back to host memory
        cudaMemcpy(TempInfo, GTempInfo, sizeof(Atom) * nBlock, cudaMemcpyDeviceToHost);

        // Calculate probabilities for each configuration
        std::vector<double> conf_p(nBlock);
        int conf_index = 0;
        float energy_max = TempInfo[0].charge;
        double sum_p = 0;

        for (int i = 1; i < nBlock; i++)
        {
            if (TempInfo[i].charge > energy_max)
            {
                energy_max = TempInfo[i].charge;
            }
        }

        for (int i = 0; i < nBlock; i++)
        {
            conf_p[i] = exp(beta * (TempInfo[i].charge - energy_max));
            sum_p += conf_p[i];
        }

        // Select a configuration based on the calculated probabilities
        if (sum_p != 0)
        {
            double conf_p_sum = 0;
            for (int i = 0; i < nBlock; i++)
            {
                conf_p_sum += conf_p[i] / sum_p;
                conf_p[i] = conf_p_sum;
            }
            float ran = (float)rand() / (float)RAND_MAX;

            for (int i = 0; i < nBlock; i++)
            {
                conf_index = i;
                if (ran < conf_p[i])
                {
                    break;
                }
            }
        }

        // Calculate nbar and B values
        const float *period = info->cryst;
        const float nbar = period[0] * period[1] * period[2] * fragmentInfo[moveFragType].conc * MOLES_TO_MOLECULES;
        const float B = beta * fragmentInfo[moveFragType].muex + log(nbar);

        float diff = -TempInfo[conf_index].charge;
        conf_p[conf_index] = exp(beta * (TempInfo[conf_index].charge - energy_max)) / sum_p;
        float fn = 1 / (nBlock * conf_p[conf_index]);
        float n = fragmentInfo[moveFragType].totalNum;
        float p = Min(1, n / fn * exp(-B - beta * diff));
        // float p = Min(1, fnTmp / (n + 1) * exp(B - beta * diff)); from add

        float ran = rand() / (float)RAND_MAX;

        // Decide whether to accept or reject the move
        if (ran < p)
        {
            // Accept the move
            fragmentInfo[moveFragType].totalNum -= 1;
            // del only with 1 block not mult style
            GupdateDel<<<1, numThreadsPerBlock>>>(GfragmentInfo, GresidueInfo, GatomInfo, GTempFrag, GTempInfo, moveFragType, fragmentInfo[moveFragType].totalNum, conf_index);
            return true;
        }
        else
        {
            // Reject the move
            return false;
        }
    }

} // extern "C"