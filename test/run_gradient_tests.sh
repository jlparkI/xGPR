#!/bin/bash

#Run gradient tests
cd gradient_calc_tests
python fht_conv1d_gradient_test.py
python graph_conv1d_gradient_test.py
python linear_gradient_test.py
python matern_gradient_test.py
python rbf_gradient_test.py
python polysum_gradient_test.py
python mini_ard_gradient_test.py
cd ..
