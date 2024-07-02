/* Contains the wrapper code for the C++ extension for CPU.
 */

//#include <nanobind/nanobind.h>
//#include <nanobind/ndarray.h>
#include "basic_ops/transform_functions.h"
#include "rbf_ops/rbf_ops.h"
#include "rbf_ops/ard_ops.h"
#include "convolution_ops/conv1d_operations.h"
#include "convolution_ops/rbf_convolution.h"



namespace nb = nanobind;
using namespace std;

NB_MODULE(xgpr_cpu_rfgen_cpp_ext, m){
    m.def("cpuFastHadamardTransform", &fastHadamard3dArray_);
    m.def("cpuFastHadamardTransform2D", &fastHadamard2dArray_);
    m.def("cpuSRHT", &SRHTBlockTransform_);

    m.def("cpuRBFFeatureGen", &rbfFeatureGen_);
    m.def("cpuRBFGrad", &rbfGrad_);
    m.def("cpuMiniARDGrad", &ardGrad_);

    m.def("cpuConv1dMaxpool", &conv1dMaxpoolFeatureGen_);
    m.def("cpuConv1dFGen", &convRBFFeatureGen_);
    m.def("cpuConvGrad", &convRBFGrad_);
}
