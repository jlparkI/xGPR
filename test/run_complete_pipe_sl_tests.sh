#!/bin/bash

#Run static_layer tests
cd static_layer_tests
python basic_statlayer_tests.py
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
