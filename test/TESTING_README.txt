Many modules in xGPR rely on the fast Hadamard transform based operations --
for generating random features and for building preconditioners. The
fht_operations_tests check these core modules.

gradient_calc ensures the analytical gradients calculated for each kernel
are correct, and is a good "smoke test" to ensure a particular kernel
is generating random features correctly.

basic_dataset_tests checks a few fundamental features of the Dataset objects,
which are used throughout xGPR.

preconditioner_tests checks that preconditioner construction is functional.

approximate_nmll_tests ensures that NMLL approximation is functional and
gives results within some tolerance of expected, either with or without
preconditioning. (Performance is better with preconditioning.)

tuning and fitting_tests check hyperparameter tuning and model fitting
routines to ensure an expected level of performance is achieved.

static_layer_tests check the static layers to ensure they are functional
(these are additional kernels the user can select for which one stage
of the kernel can be calculated in advance).

complete_pipeline_tests load the data, tunes hyperparameters, fits the
model and checks performance for a specified kernel.

If a problem is encountered after making changes to xGPR,
it may be best to begin with the most
fundamental tests (fht_operations_tests) and work upwards to
least fundamental / optional features, as follows:

fht_operations
basic_dataset
gradient_calc
preconditioner
approximate_nmll
fitting
tuning
complete_pipeline


and then if the problem is encountered in the static_layer,
run those tests as appropriate.
