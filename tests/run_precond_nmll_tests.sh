#!/bin/bash

#Run approximate NMLL tests
cd approximate_nmll_tests
python test_slq_nmll.py
cd ..

#Run preconditioner tests
cd preconditioner_tests
python preconditioner_test.py
cd ..
