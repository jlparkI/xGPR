#!/bin/bash

# Add the newly installed cuda toolkit to path.
export PATH=/usr/local/cuda-11.0/bin/:$PATH
export PATH=/usr/local/cuda-11.0/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH


for PYVER in "cp39-cp39" "cp310-cp310" "cp311-cp311" "cp312-cp312"
do
    PYBIN="/opt/python/${PYVER}/bin"
    ${PYBIN}/pip install build
    ${PYBIN}/python -m build --wheel
done

rm build_instructions.txt
rm cuda_build_script.sh
rm cuda_setup_script.sh
rm pyproj_updater.py
rm *.rpm*
rm -rf build

cd dist
for wheel in $PWD/*.whl
do
    auditwheel repair $wheel
done
rm *.whl
cd ..
