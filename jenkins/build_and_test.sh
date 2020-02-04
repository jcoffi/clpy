#!/bin/bash

# A filename to record stderrs
ERRORS_FILENAME=$WORKSPACE/erros.log


# Detailed output
set -x
# Exit immediately if an error has occurred
set -e

# Specify a python environment
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell 3.6.5

# Set up a new temporary python modules environment
python -m venv venv
source venv/bin/activate


# Install dependencies
pip install -U pip
pip install 'Cython==0.29.10' 'pytest==4.3.0' 'mock==3.0.5' 'nose==1.3.7'

# Ignore occurred errors below
set +e
# Install clpy
python setup.py develop 2>&1 | tee build_log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  cat build_log >> $ERRORS_FILENAME
  exit 1
fi




# Run pytests
TEST_DIRS="
tests/clpy_tests/core_tests/
tests/clpy_tests/opencl_tests/
tests/clpy_tests/binary_tests/
tests/clpy_tests/ext_tests/
tests/clpy_tests/testing_tests/
tests/clpy_tests/io_tests/
tests/clpy_tests/padding_tests/
tests/clpy_tests/creation_tests/
tests/clpy_tests/manipulation_tests/
tests/clpy_tests/indexing_tests/
tests/clpy_tests/prof_tests/
tests/clpy_tests/statics_tests/
tests/clpy_tests/math_tests/
tests/clpy_tests/logic_tests/
tests/clpy_tests/random_tests/
tests/example_tests/
"

TEST_DIRS_IN_ROOT_DIR="
tests/install_tests/
"

TEST_FILES="
tests/clpy_tests/linalg_tests/test_product.py
tests/clpy_tests/sorting_tests/test_count.py
tests/clpy_tests/sorting_tests/test_search.py
tests/clpy_tests/sorting_tests/test_sort.py::TestSort
"

export CLPY_TEST_GPU_LIMIT=1

ERROR_HAS_OCCURRED=0

for d in $TEST_DIRS; do
  pushd $d
  python -m pytest  2>&1 | tee temporary_log
  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
  popd
done

for d in $TEST_DIRS_IN_ROOT_DIR; do
  python -m pytest $d  2>&1 | tee temporary_log
  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
done

for f in $TEST_FILES; do
  pushd $(dirname $f)
  python -m pytest $(basename $f) 2>&1 | tee temporary_log
  status=${PIPESTATUS[0]}
  if [[ status -ne 0 ]] && [[ status -ne 5 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
  popd
done

exit $ERROR_HAS_OCCURRED
