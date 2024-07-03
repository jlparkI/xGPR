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
    m.def("cudaFastHadamardTransform2D", &cudaHTransform<float>);
    m.def("cudaFastHadamardTransform2D", &cudaHTransform<double>);
    m.def("cudaSRHT", &cudaHTransform<float>);
    m.def("cudaSRHT", &cudaHTransform<double>);

    m.def("cudaRBFFeatureGen", &RBFFeatureGen<float>);
    m.def("cudaRBFFeatureGen", &RBFFeatureGen<double>);
    m.def("cudaRBFGrad", &RBFFeatureGrad<float>);
    m.def("cudaRBFGrad", &RBFFeatureGrad<double>);
    m.def("cudaMiniARDGrad", &ardCudaGrad<float>);
    m.def("cudaMiniARDGrad", &ardCudaGrad<double>);

    m.def("cudaConv1dMaxpool", &conv1dMaxpoolFeatureGen<float>);
    m.def("cudaConv1dMaxpool", &conv1dMaxpoolFeatureGen<double>);
    m.def("cudaConv1dFGen", &convRBFFeatureGen<float>);
    m.def("cudaConv1dFGen", &convRBFFeatureGen<double>);
    m.def("cudaConvGrad", &convRBFFeatureGrad<float>);
    m.def("cudaConvGrad", &convRBFFeatureGrad<double>);
}
