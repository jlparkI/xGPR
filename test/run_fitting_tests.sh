#!/bin/bash

#Run fitting tests
cd fitting_tests
python test_cg_fit.py
python test_exact_fit.py
python test_lbfgs_fit.py
python test_offline_cg_fit.py
python test_pretransformed_fit.py
cd ..
