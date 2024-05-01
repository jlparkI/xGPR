/*What follows is the simplex projection code (Reid et al. 2023).
We have not found this to provide any consistent improvement
in performance, in contrast to their results, although this
may be implementation or hyperparameter-tuning related. We are retaining
this code for now in case it turns out to be useful in future.*/
//Now apply the simplex projection to the temporary array. We
        //first have to sum the elements of the temporary array and
        //use the existing shared memory as storage to help with this.
        s_data[threadIdx.x] = 0;
        tempArrPos = (blockIdx.x << log2N);
        simplexProjPrefactor = sqrt( (T)paddedBufferSize - 1.);

        for (int i = threadIdx.x; i < (paddedBufferSize - 1); i += blockDim.x)
            s_data[threadIdx.x] += cArray[i + tempArrPos];

        __syncthreads();
        for (int i = blockDim.x/2; i > 0; i >>=1){
            if (threadIdx.x < i)
                s_data[threadIdx.x] += s_data[threadIdx.x + i];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            cArray[tempArrPos + paddedBufferSize - 1] = s_data[0] / simplexProjPrefactor;

        __syncthreads();
        bufferSum = s_data[0] / simplexProjPrefactor;
        bufferSum *= ( (sqrt( (T)paddedBufferSize) + 1) / ((T)paddedBufferSize - 1.) );
        simplexProjPrefactor = sqrt( (T)paddedBufferSize / ((T)paddedBufferSize - 1.) );

        for (int i=threadIdx.x; i < (paddedBufferSize - 1); i+=blockDim.x)
            cArray[i + tempArrPos] = (cArray[i + tempArrPos] * simplexProjPrefactor - bufferSum);

        __syncthreads();
