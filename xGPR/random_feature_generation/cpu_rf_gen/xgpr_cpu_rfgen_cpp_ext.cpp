/* Contains the wrapper code for the C++ extension for CPU.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "basic_ops/transform_functions.h"
#include "rbf_ops/rbf_ops.h"
#include "rbf_ops/ard_ops.h"
#include "convolution_ops/conv1d_operations.h"
#include "convolution_ops/rbf_convolution.h"



namespace nb = nanobind;
using namespace std;

NB_MODULE(xgpr_cpu_rfgen_cpp_ext, m){
    m.def("cpuFastHadamardTransform", &fastHadamard3dArray_<float>,
            nb::arg("inputArr").noconvert(), nb::arg("numThreads"));
    m.def("cpuFastHadamardTransform", &fastHadamard3dArray_<double>,
            nb::arg("inputArr").noconvert(), nb::arg("numThreads"));

    m.def("cpuFastHadamardTransform2D", &fastHadamard2dArray_<float>,
            nb::arg("inputArr").noconvert(), nb::arg("numThreads"));
    m.def("cpuFastHadamardTransform2D", &fastHadamard2dArray_<double>,
            nb::arg("inputArr").noconvert(), nb::arg("numThreads"));

    m.def("cpuSRHT", &SRHTBlockTransform<float>, nb::arg("inputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("numThreads"));
    m.def("cpuSRHT", &SRHTBlockTransform<double>, nb::arg("inputArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("numThreads"));

    m.def("cpuRBFFeatureGen", &rbfFeatureGen_<float>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("numThreads"),
            nb::arg("fitIntercept"));
    m.def("cpuRBFFeatureGen", &rbfFeatureGen_<double>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("numThreads"),
            nb::arg("fitIntercept"));

    m.def("cpuRBFGrad", &rbfGrad_<float>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("sigma"), nb::arg("numThreads"), nb::arg("fitIntercept"));
    m.def("cpuRBFGrad", &rbfGrad_<double>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("sigma"), nb::arg("numThreads"), nb::arg("fitIntercept"));
    m.def("cpuMiniARDGrad", &ardGrad_<float>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("precompWeights").noconvert(),
            nb::arg("sigmaMap").noconvert(), nb::arg("sigmaVals").noconvert(),
            nb::arg("gradArr").noconvert(), nb::arg("numThreads"),
            nb::arg("fitIntercept")); 
    m.def("cpuMiniARDGrad", &ardGrad_<double>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("precompWeights").noconvert(),
            nb::arg("sigmaMap").noconvert(), nb::arg("sigmaVals").noconvert(),
            nb::arg("gradArr").noconvert(), nb::arg("numThreads"),
            nb::arg("fitIntercept")); 

    m.def("cpuConv1dMaxpool", &conv1dMaxpoolFeatureGen_<float>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"), nb::arg("numThreads"));
    m.def("cpuConv1dMaxpool", &conv1dMaxpoolFeatureGen_<double>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"), nb::arg("numThreads"));

    m.def("cpuConv1dFGen", &convRBFFeatureGen_<float>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"), nb::arg("scalingType"),
            nb::arg("numThreads"));
    m.def("cpuConv1dFGen", &convRBFFeatureGen_<double>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"), nb::arg("scalingType"),
            nb::arg("numThreads"));

    m.def("cpuConvGrad", &convRBFGrad_<float>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("seqlengths").noconvert(),
            nb::arg("gradArr").noconvert(), nb::arg("sigma"),
            nb::arg("convWidth"), nb::arg("scalingType"), nb::arg("numThreads"));
    m.def("cpuConvGrad", &convRBFGrad_<double>, nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("seqlengths").noconvert(),
            nb::arg("gradArr").noconvert(), nb::arg("sigma"),
            nb::arg("convWidth"), nb::arg("scalingType"), nb::arg("numThreads"));
}
