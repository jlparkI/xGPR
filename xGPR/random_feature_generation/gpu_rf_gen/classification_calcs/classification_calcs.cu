/* Copyright (C) 2025 Jonathan Parkinson
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
// C++ headers

// Library headers

// Project headers
#include "classification_calcs.h"


namespace CudaClassificationCalcs {

static constexpr int CL_NUM_THREADS_PER_BLOCK = 256;

__global__ void findClassMeansKernel(const double *input_ptr,
        double *class_means_ptr, const int32_t *class_label_ptr,
        int64_t *class_count_ptr, int x_dim0, int x_dim1,
        int nclasses) {
    const double *current_input = input_ptr;
    double *current_output = class_means_ptr + blockIdx.x * x_dim1;

    for (int i=0; i < x_dim0; i++) {
        int class_label = class_label_ptr[i];

        if (class_label != blockIdx.x) {
            current_input += x_dim1;
            continue;
        }

        if (threadIdx.x == 0)
            class_count_ptr[class_label] += 1;

        for (int j=threadIdx.x; j < x_dim1; j+=CL_NUM_THREADS_PER_BLOCK)
            current_output[j] += current_input[j];

        current_input += x_dim1;
    }
}



__global__ void prepPooledCovCalcKernel(double *input_ptr, const double *class_means_ptr,
        const int32_t *class_label_ptr, const double *class_prior_ptr,
        int x_dim0, int x_dim1, int nclasses) {
    int pos = blockIdx.x * CL_NUM_THREADS_PER_BLOCK + threadIdx.x;
    int class_label_pos = pos / x_dim1;
    int row_position = pos % x_dim1;

    if (pos < (x_dim0 * x_dim1)) {
        const int32_t class_label = class_label_ptr[class_label_pos];
        if (class_label >= 0 && class_label < nclasses) {
            int class_mean_position = class_label * x_dim1 + row_position;
            input_ptr[pos] = (input_ptr[pos] - class_means_ptr[class_mean_position]) *
                class_prior_ptr[class_label];
        }
    }
}


void cudaFindClassMeans_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
    nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> class_means,
    nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_labels,
    nb::ndarray<int64_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_counts) {
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a
    // failsafe -- so we do not need to provide detailed exception messages
    // here.
    int x_dim0 = input_arr.shape(0);
    int x_dim1 = input_arr.shape(1);
    size_t nclasses = class_means.shape(0);

    const double *input_ptr = static_cast<double*>(input_arr.data());
    double *class_means_ptr = static_cast<double*>(class_means.data());
    const int32_t *class_label_ptr = static_cast<int32_t*>(class_labels.data());
    int64_t *class_count_ptr = static_cast<int64_t*>(class_counts.data());

    if (x_dim0 == 0 || x_dim1 == 0)
        throw std::runtime_error("no datapoints");

    if (nclasses != class_counts.shape(0) ||
            nclasses == 0)
        throw std::runtime_error("incorrect number of classes");

    if (x_dim1 != class_means.shape(1))
        throw std::runtime_error("incorrect size for class means");

    if (x_dim0 != class_labels.shape(0))
        throw std::runtime_error("class labels and datapoints do not match");

    // Note that checking the maximum element here would require launching a kernel,
    // which is expensive if not necessary. To avoid this we instead check that
    // individual class labels are not in violation in the kernel itself, and
    // if they are, set them to be >= 0 and < nclasses.
    findClassMeansKernel<<<nclasses, CL_NUM_THREADS_PER_BLOCK>>>(input_ptr,
                class_means_ptr, class_label_ptr, class_count_ptr, x_dim0,
                x_dim1, nclasses);        

}


void cudaPrepPooledCovCalc_(nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> input_arr,
    nb::ndarray<double, nb::shape<-1,-1>, nb::device::cuda, nb::c_contig> class_means,
    nb::ndarray<int32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_labels,
    nb::ndarray<double, nb::shape<-1>, nb::device::cuda, nb::c_contig> class_priors) {
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a
    // failsafe -- so we do not need to provide detailed exception messages
    // here.
    int x_dim0 = input_arr.shape(0);
    int x_dim1 = input_arr.shape(1);
    size_t nclasses = class_means.shape(0);

    if (x_dim0 == 0 || x_dim1 == 0)
        throw std::runtime_error("no datapoints");

    if (nclasses != class_priors.shape(0) ||
            nclasses == 0)
        throw std::runtime_error("incorrect number of classes");

    if (x_dim1 != class_means.shape(1))
        throw std::runtime_error("incorrect size for class means");

    if (x_dim0 != class_labels.shape(0))
        throw std::runtime_error("class labels and datapoints do not match");

    double *input_ptr = static_cast<double*>(input_arr.data());
    const double *class_means_ptr = static_cast<double*>(class_means.data());
    const int32_t *class_label_ptr = static_cast<int32_t*>(class_labels.data());
    const double *class_prior_ptr = static_cast<double*>(class_priors.data());

    // Note that checking the maximum element here would require launching a kernel,
    // which is expensive if not necessary. To avoid this we instead check that
    // individual class labels are not in violation in the kernel itself, and
    // if they are, set them to be >= 0 and < nclasses.
    int num_elements = x_dim0 * x_dim1;
    int nblocks = (num_elements + CL_NUM_THREADS_PER_BLOCK) / CL_NUM_THREADS_PER_BLOCK;
    prepPooledCovCalcKernel<<<nblocks, CL_NUM_THREADS_PER_BLOCK>>>(input_ptr,
            class_means_ptr, class_label_ptr,
            class_prior_ptr, x_dim0, x_dim1, nclasses);        
}


}  // namespace CudaClassificationCalcs
