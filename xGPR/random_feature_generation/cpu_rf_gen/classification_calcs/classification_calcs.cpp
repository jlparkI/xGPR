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

namespace nb = nanobind;


namespace CpuClassificationCalcs {


void cpuFindClassMeans_(
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> class_means,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_labels,
nb::ndarray<int64_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_counts) {
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

    for (size_t i=0; i < x_dim0; i++) {
        if (class_label_ptr[i] >= nclasses || class_label_ptr[i] < 0) {
            throw std::runtime_error("class labels and datapoints "
                    "do not match");
        }
    }

    for (size_t i=0; i < x_dim0; i++) {
        int32_t class_label = class_label_ptr[i];
        class_count_ptr[class_label] += 1;
        double *cl_means_row = class_means_ptr + class_label * x_dim1;

        #pragma omp simd
        for (size_t j=0; j < x_dim1; j++)
            cl_means_row[j] += input_ptr[j];

        input_ptr += x_dim1;
    }
}



void cpuPrepPooledCovCalc_(
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> class_means,
nb::ndarray<int32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_labels,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> class_prior_sqrts) {
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a
    // failsafe -- so we do not need to provide detailed exception messages
    // here.
    int x_dim0 = input_arr.shape(0);
    int x_dim1 = input_arr.shape(1);
    size_t nclasses = class_means.shape(0);

    double *input_ptr = static_cast<double*>(input_arr.data());
    const double *class_means_ptr = static_cast<double*>(class_means.data());
    const int32_t *class_label_ptr = static_cast<int32_t*>(class_labels.data());
    const double *class_prior_ptr = static_cast<double*>(class_prior_sqrts.data());

    if (x_dim0 == 0 || x_dim1 == 0)
        throw std::runtime_error("no datapoints");

    if (nclasses != class_prior_sqrts.shape(0) ||
            nclasses == 0)
        throw std::runtime_error("incorrect number of classes");

    if (x_dim1 != class_means.shape(1))
        throw std::runtime_error("incorrect size for class means");

    if (x_dim0 != class_labels.shape(0))
        throw std::runtime_error("class labels and datapoints do not match");

    for (size_t i=0; i < x_dim0; i++) {
        if (class_label_ptr[i] >= nclasses || class_label_ptr[i] < 0) {
            throw std::runtime_error("class labels and datapoints "
                    "do not match");
        }
    }

    for (size_t i=0; i < x_dim0; i++) {
        int32_t class_label = class_label_ptr[i];
        double prior = class_prior_ptr[class_label];
        const double *cl_means_row = class_means_ptr + class_label * x_dim1;

        #pragma omp simd
        for (size_t j=0; j < x_dim1; j++)
            input_ptr[j] = (input_ptr[j] - cl_means_row[j]) * prior;

        input_ptr += x_dim1;
    }
}





int64_t cpu_welford_mean_variance(
nb::ndarray<double, nb::shape<-1,-1>, nb::device::cpu, nb::c_contig> input_arr,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> xmean,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> xmk,
nb::ndarray<double, nb::shape<-1>, nb::device::cpu, nb::c_contig> xsk,
int64_t start_ndatapoints) {
    // Perform safety checks. Any exceptions thrown here are handed off to Python
    // by the Nanobind wrapper. We do not expect the user to see these because
    // the Python code will always ensure inputs are correct -- these are a
    // failsafe -- so we do not need to provide detailed exception messages
    // here.
    int x_dim0 = input_arr.shape(0);
    int x_dim1 = input_arr.shape(1);

    if (x_dim1 != xmean.shape(0) ||
            x_dim1 != xmk.shape(0) ||
            x_dim1 != xsk.shape(0)) {
        throw std::runtime_error("Incorrect sized arrays passed "
                "to wrapped function.");
    }

    if (x_dim0 == 0 || x_dim1 == 0)
        throw std::runtime_error("no datapoints");

    int64_t output_ndatapoints = start_ndatapoints;
    const double *input_ptr = static_cast<double*>(input_arr.data());
    double *xmean_ptr = static_cast<double*>(xmean.data());
    double *xmk_ptr = static_cast<double*>(xmk.data());
    double *xsk_ptr = static_cast<double*>(xsk.data());

    for (size_t i=0; i < x_dim0; i++) {
        output_ndatapoints += 1;

        #pragma omp simd
        for (size_t j=0; j < x_dim1; j++) {
            // Update the running mean...
            double updated_mean = input_ptr[j] - xmean_ptr[j];
            updated_mean /= output_ndatapoints;
            xmean_ptr[j] += updated_mean;

            // Now separately update xmk and xsk.
            double updated_mk = input_ptr[j] - xmk_ptr[j];
            xmk_ptr[j] += updated_mk / output_ndatapoints;

            double updated_sk = updated_mk * (input_ptr[j] - xmk_ptr[j]);
            xsk_ptr[j] += updated_sk;
        }
        input_ptr += x_dim1;
    }

    return output_ndatapoints;
}





}  // namespace CpuClassificationCalcs
