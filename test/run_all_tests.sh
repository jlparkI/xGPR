#!/bin/bash

#Run data handler tests
cd basic_dataset_tests
python basic_dataset_tests.py
cd ..

#Run fht tests
cd fht_operations_tests
python basic_fht_functions_test.py
python fht_conv1d_test.py
python poly_fht.py
python mm_conv1d_test.py
cd ..

#Run gradient tests
cd gradient_calc_tests
python conv1d_gradient_test.py
python fht_conv1d_gradient_test.py
python graph_conv1d_gradient_test.py
python linear_gradient_test.py
python matern_gradient_test.py
python rbf_gradient_test.py
python polysum_gradient_test.py
cd ..

#Run approximate NMLL tests
cd approximate_nmll_tests
python test_slq_nmll.py
cd ..

#Run preconditioner tests
cd preconditioner_tests
python preconditioner_test.py
cd ..

#Run static_layer tests
cd static_layer_tests
python basic_statlayer_tests.py
cd ..

#Run tuning tests
cd tuning_tests
python test_crude_bayes_tuning.py
python test_crude_grid_tuning.py
python test_crude_lbfgs_tuning.py
python test_fine_bayes_tuning.py
python test_fine_direct_tuning.py
python test_fitting_bayes_tuning.py
python test_fitting_direct_tuning.py
python test_sgd_tuning.py
cd ..

#Run fitting tests
cd fitting_tests
python test_cg_fit.py
python test_exact_fit.py
python test_lbfgs_fit.py
python test_offline_cg_fit.py
python test_pretransformed_fit.py
python test_svrg_fit.py
cd ..

#Complete pipeline tests
cd complete_pipeline_tests
python test_classic_poly.py
python test_conv1d_fit.py
python test_fht_conv1d_fit.py
python test_graphconv_fit.py
python test_linear.py
python test_matern_fit.py
python test_polysum.py
python test_rbf_fit.py
cd ..
