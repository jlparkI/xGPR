#!/bin/bash

#Run tuning tests
cd tuning_tests
python test_crude_tuning.py
python test_tuning.py
cd ..
