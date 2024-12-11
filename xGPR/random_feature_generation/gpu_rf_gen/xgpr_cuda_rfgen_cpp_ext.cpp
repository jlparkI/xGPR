/* Contains the wrapper code for the C++ extension for Cuda.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "basic_ops/basic_array_operations.h"
#include "rbf_ops/rbf_ops.h"
#include "rbf_ops/ard_ops.h"
#include "convolution_ops/convolution.h"
#include "convolution_ops/rbf_convolution.h"


namespace nb = nanobind;
using namespace std;

NB_MODULE(xgpr_cuda_rfgen_cpp_ext, m){
    m.def("cudaFastHadamardTransform2D", &cudaHTransform<float>,
            nb::arg("inputArr").noconvert());
    m.def("cudaFastHadamardTransform2D", &cudaHTransform<double>,
            nb::arg("inputArr").noconvert());
    m.def("cudaSRHT", &cudaSRHT2d<float>,
            nb::arg("inputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("numThreads"));
    m.def("cudaSRHT", &cudaSRHT2d<double>,
            nb::arg("inputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("numThreads"));

    m.def("cudaRBFFeatureGen", &RBFFeatureGen<float>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("fitIntercept"));
    m.def("cudaRBFFeatureGen", &RBFFeatureGen<double>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("fitIntercept"));

    m.def("cudaRBFGrad", &RBFFeatureGrad<float>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("gradArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("sigma"), nb::arg("fitIntercept"));
    m.def("cudaRBFGrad", &RBFFeatureGrad<double>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("gradArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("sigma"), nb::arg("fitIntercept"));

    m.def("cudaMiniARDGrad", &ardCudaGrad<float>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("precompWeights").noconvert(), nb::arg("sigmaMap").noconvert(),
            nb::arg("sigmaVals").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("fitIntercept"));
    m.def("cudaMiniARDGrad", &ardCudaGrad<double>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("precompWeights").noconvert(), nb::arg("sigmaMap").noconvert(),
            nb::arg("sigmaVals").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("fitIntercept"));

    m.def("cudaConv1dMaxpool", &conv1dMaxpoolFeatureGen<float>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(), nb::arg("convWidth"));
    m.def("cudaConv1dMaxpool", &conv1dMaxpoolFeatureGen<double>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(), nb::arg("convWidth"));

    m.def("cudaConv1dFGen", &convRBFFeatureGen<float>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(), nb::arg("convWidth"),
            nb::arg("scalingType"));
    m.def("cudaConv1dFGen", &convRBFFeatureGen<double>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(), nb::arg("convWidth"),
            nb::arg("scalingType"));

    m.def("cudaConvGrad", &convRBFFeatureGrad<float>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("sigma"), nb::arg("convWidth"),
            nb::arg("scalingType"));
    m.def("cudaConvGrad", &convRBFFeatureGrad<double>,
            nb::arg("inputArr").noconvert(), nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("sigma"), nb::arg("convWidth"),
            nb::arg("scalingType"));
}
