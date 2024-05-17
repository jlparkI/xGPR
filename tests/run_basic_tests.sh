#!/bin/bash

#Run data handler tests
cd basic_dataset_tests
python basic_dataset_tests.py
cd ..

#Run fht tests
cd fht_operations_tests
python basic_fht_functions_test.py
python fht_conv1d_test.py
python rbf_fht.py
python ard_grad_fht.py
python variable_length_seqs_test.py
cd ..
