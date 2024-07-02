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
    m.def("cpuFastHadamardTransform", &fastHadamard3dArray_<float>);
    m.def("cpuFastHadamardTransform", &fastHadamard3dArray_<double>);
    m.def("cpuFastHadamardTransform2D", &fastHadamard2dArray_<float>);
    m.def("cpuFastHadamardTransform2D", &fastHadamard2dArray_<double>);
    m.def("cpuSRHT", &SRHTBlockTransform<float>);
    m.def("cpuSRHT", &SRHTBlockTransform<double>);

    m.def("cpuRBFFeatureGen", &rbfFeatureGen_<float>);
    m.def("cpuRBFFeatureGen", &rbfFeatureGen_<double>);
    m.def("cpuRBFGrad", &rbfGrad_<float>);
    m.def("cpuRBFGrad", &rbfGrad_<double>);
    m.def("cpuMiniARDGrad", &ardGrad_<float>);
    m.def("cpuMiniARDGrad", &ardGrad_<double>);

    m.def("cpuConv1dMaxpool", &conv1dMaxpoolFeatureGen_<float>);
    m.def("cpuConv1dMaxpool", &conv1dMaxpoolFeatureGen_<double>);
    m.def("cpuConv1dFGen", &convRBFFeatureGen_<float>);
    m.def("cpuConv1dFGen", &convRBFFeatureGen_<double>);
    m.def("cpuConvGrad", &convRBFGrad_<float>);
    m.def("cpuConvGrad", &convRBFGrad_<double>);
}
