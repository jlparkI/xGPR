#!/bin/bash

#Run gradient tests
cd gradient_calc_tests
python check_kernel_gradients.py
cd ..
