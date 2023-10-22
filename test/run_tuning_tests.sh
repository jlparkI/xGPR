#!/bin/bash

#Run tuning tests
cd tuning_tests
python test_crude_bayes_tuning.py
python test_crude_grid_tuning.py
python test_crude_lbfgs_tuning.py
python test_fine_bayes_tuning.py
python test_fine_direct_tuning.py
cd ..
