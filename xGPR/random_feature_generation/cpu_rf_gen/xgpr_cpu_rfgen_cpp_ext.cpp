/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
// C++ headers

// Library headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Project headers
#include "basic_ops/transform_functions.h"
#include "rbf_ops/rbf_ops.h"
#include "rbf_ops/ard_ops.h"
#include "convolution_ops/conv1d_operations.h"
#include "convolution_ops/rbf_convolution.h"
#include "classification_calcs/classification_calcs.h"


namespace nb = nanobind;

NB_MODULE(xgpr_cpu_rfgen_cpp_ext, m) {
    m.def("cpuFastHadamardTransform",
            &CPUHadamardTransformBasicCalculations::fastHadamard3dArray_<float>,
            nb::arg("inputArr").noconvert());
    m.def("cpuFastHadamardTransform",
            &CPUHadamardTransformBasicCalculations::fastHadamard3dArray_<double>,
            nb::arg("inputArr").noconvert());

    m.def("cpuFastHadamardTransform2D",
            &CPUHadamardTransformBasicCalculations::fastHadamard2dArray_<float>,
            nb::arg("inputArr").noconvert());
    m.def("cpuFastHadamardTransform2D",
            &CPUHadamardTransformBasicCalculations::fastHadamard2dArray_<double>,
            nb::arg("inputArr").noconvert());

    m.def("cpuSRHT",
            &CPUHadamardTransformBasicCalculations::SRHTBlockTransform<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("radem").noconvert());
    m.def("cpuSRHT",
            &CPUHadamardTransformBasicCalculations::SRHTBlockTransform<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("radem").noconvert());

    m.def("cpuRBFFeatureGen",
            &CPURBFKernelCalculations::rbfFeatureGen_<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("fitIntercept"));
    m.def("cpuRBFFeatureGen",
            &CPURBFKernelCalculations::rbfFeatureGen_<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(), nb::arg("fitIntercept"));

    m.def("cpuRBFGrad",
            &CPURBFKernelCalculations::rbfGrad_<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("sigma"), nb::arg("fitIntercept"));
    m.def("cpuRBFGrad", &CPURBFKernelCalculations::rbfGrad_<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(), nb::arg("gradArr").noconvert(),
            nb::arg("radem").noconvert(), nb::arg("chiArr").noconvert(),
            nb::arg("sigma"), nb::arg("fitIntercept"));

    m.def("cpuMiniARDGrad",
            &CPUARDKernelCalculations::ardGrad_<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("precompWeights").noconvert(),
            nb::arg("sigmaMap").noconvert(),
            nb::arg("sigmaVals").noconvert(),
            nb::arg("gradArr").noconvert(),
            nb::arg("fitIntercept"));
    m.def("cpuMiniARDGrad",
            &CPUARDKernelCalculations::ardGrad_<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("precompWeights").noconvert(),
            nb::arg("sigmaMap").noconvert(),
            nb::arg("sigmaVals").noconvert(),
            nb::arg("gradArr").noconvert(),
            nb::arg("fitIntercept"));

    m.def("cpuConv1dMaxpool",
            &CPUMaxpoolKernelCalculations::conv1dMaxpoolFeatureGen_<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"));
    m.def("cpuConv1dMaxpool",
            &CPUMaxpoolKernelCalculations::conv1dMaxpoolFeatureGen_<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"));

    m.def("cpuConv1dFGen",
            &CPURBFConvolutionKernelCalculations::convRBFFeatureGen_<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"),
            nb::arg("scalingType"));
    m.def("cpuConv1dFGen",
            &CPURBFConvolutionKernelCalculations::convRBFFeatureGen_<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(),
            nb::arg("convWidth"),
            nb::arg("scalingType"));

    m.def("cpuConvGrad",
            &CPURBFConvolutionKernelCalculations::convRBFGrad_<float>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(),
            nb::arg("gradArr").noconvert(),
            nb::arg("sigma"),
            nb::arg("convWidth"),
            nb::arg("scalingType"));
    m.def("cpuConvGrad",
            &CPURBFConvolutionKernelCalculations::convRBFGrad_<double>,
            nb::arg("inputArr").noconvert(),
            nb::arg("outputArr").noconvert(),
            nb::arg("radem").noconvert(),
            nb::arg("chiArr").noconvert(),
            nb::arg("seqlengths").noconvert(),
            nb::arg("gradArr").noconvert(),
            nb::arg("sigma"),
            nb::arg("convWidth"),
            nb::arg("scalingType"));

    m.def("cpuFindClassMeans",
            &CpuClassificationCalcs::cpuFindClassMeans_,
            nb::arg("input_arr").noconvert(),
            nb::arg("class_means").noconvert(),
            nb::arg("class_labels").noconvert(),
            nb::arg("class_counts").noconvert());

    m.def("cpuPrepPooledCovCalc",
            &CpuClassificationCalcs::cpuPrepPooledCovCalc_,
            nb::arg("input_arr").noconvert(),
            nb::arg("class_means").noconvert(),
            nb::arg("class_labels").noconvert(),
            nb::arg("class_prior_sqrts").noconvert());
}
