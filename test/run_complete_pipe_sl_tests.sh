#!/bin/bash

#Complete pipeline tests
cd complete_pipeline_tests
python discriminant_pipeline_test.py
python test_current_kernels.py
cd ..
