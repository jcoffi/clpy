# ClPy: OpenCL backend for CuPy

*ClPy* is an implementation of [CuPy](https://cupy.chainer.org/)'s OpenCL backend.
In other words, ClPy enables software written in CuPy to also work on OpenCL devices, not only on devices that support CUDA (NVIDIA).

## Current status

The current ClPy is a release-candidate version, forked from [CuPy v2.1.0](https://github.com/cupy/cupy/releases/tag/v2.1.0).
ClPy supports most of CuPy's functionalities.

* All core [ndarray](https://docs-cupy.chainer.org/en/v2.5.0/reference/ndarray.html)
* All core [universal functions](https://docs-cupy.chainer.org/en/v2.5.0/reference/ufunc.html)
* All core [custom kernels](https://docs-cupy.chainer.org/en/v2.5.0/reference/kernel.html)
* BLAS library compatible with cuBLAS
* Multiple devices (thus ChainerMN)

ClPy is still under development and has the following limitations.

* Other CUDA libraries (cuSPARSE, cuSOLVER, cuDnn, cuRAND, thrust) are not supported
* Half and complex are not supported
* No multiple command queue (Stream on CUDA)
* Dockerfile and some other files have not been updated and thus may not work

The whole CuPy suite of tests are passing (with the exception of tests related to unsupported libraries). See current [CuPy's test and example results](https://github.com/fixstars/ClPy/wiki/cupy_test_example_results).

Almost all [Chainer](https://chainer.org/) works.
See current [Chainer's test and example results](https://github.com/fixstars/ClPy/wiki/chainer_test_example_results).

## Recommended environments

We develop and test ClPy using the following environments.

* Primary machine
	* OS: Ubuntu 16.04.4 LTS
	* CPU: Core i7-7700
	* GPU: AMD Radeon Vega Frontier Edition (Air Cooled)
	* SDK: amdgpu-pro-18.20
* Secondary machine
	* OS: Ubuntu 16.04.4 LTS
	* CPU: Core i9-7900X
	* GPU: NVIDIA TITAN V
	* SDK: CUDA 9.2

We use Python 3.6.5 to develop ClPy, and currently do not check the behavior on other versions of Python.
We recommend those environments to all ClPy users. However, reports from other environments are welcome.

## Installation

### Setup OpenCL

Install and setup OpenCL environment.

`cl.h` and OpenCL libs (`libOpenCL.so`) must be able to be included and linked without any special path settings.

For example, for AMD APP SDK, the following environment variables should be set:

```sh
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${AMDAPPSDKROOT}/include
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${AMDAPPSDKROOT}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${AMDAPPSDKROOT}/lib/x86_64
```

In addition, add the needed ldconfig files to `/etc/ldconf.so.d/`, then execute `$ sudo ldconfig`.

### Install LLVM/Clang

The current ClPy version requires LLVM/Clang 4, 5, 6, 7, 8, 9, 10, or 11.
We **strongly** recommend building and installing LLVM/Clang from source.
However, at least in Ubuntu 16.04, you can use LLVM/Clang as provided by the Ubuntu official package repository.
In that case, you will need to set some environment variables as shown below.

```console
# apt install clang-6.0 libclang-6.0-dev
$ export PATH=/usr/lib/llvm-6.0/bin:${PATH}
$ export CPLUS_INCLUDE_PATH=/usr/lib/llvm-6.0/include:${CPLUS_INCLUDE_PATH}
$ export LIBRARY_PATH=/usr/lib/llvm-6.0/lib:${LIBRARY_PATH}
$ export LD_LIBRARY_PATH=/usr/lib/llvm-6.0/lib:${LD_LIBRARY_PATH}
```

### Install CLBlast

ClPy depends on [CLBlast 1.4.1](https://github.com/CNugteren/CLBlast/releases/tag/1.4.1) or newer.
Install it and set the paths if needed.

### Install ClPy

As ClPy uses `make` in its build process, please install it before installing ClPy.
Only install ClPy after installing OpenCL and LLVM/Clang.

```console
$ pip install cython
$ python setup.py install
```

## How to use

Run your CuPy code using the `-m clpy` option (e.g. `python -m clpy /path/to/chainer/examples/mnist/train_mnist.py -g0`).
This option adds aliases to CuPy by hooking `import cupy` and calls ClPy through `cupy.foobar`, thus no code modification is necessary.

If you don't want to have to run your code with the `-m` option, you must add `import clpy` before `import cupy` to your code.
`import clpy` adds the same aliases as `-m clpy`.

If you want to disable those aliases, set `export CLPY_NOT_HOOK_CUPY=1` and replace `cupy` with `clpy` (e.g. `import cupy` -> `import clpy`) in all files that uses CuPy (e.g. Chainer).

### Compatibility with Chainer

ClPy is confirmed to work with [Chainer v3.3.0](https://github.com/chainer/chainer/tree/v3.3.0).

### Tests

```console
$ pip install pytest nose
$ cd tests/you/want
$ python -m pytest test_you_want.py
```

## Development

1. All source codes (including comments) and commit messages should be written in English.
2. Issues and pull requests are welcome in any language (recommended in English or Japanese).
3. Detailed coding styles are the same as [CuPy's](https://docs-cupy.chainer.org/en/stable/contribution.html#coding-guidelines). Read and follow the guidelines before submitting PRs.

## Future plan

The next release will be v2.1.0rc2, and should include the following improvements.

* Improve chainer's example performance
* Multiple CommandQueue (CUDA Stream)
* Support for sorting algorithms
* -- and other functions and/or bug fixes that someone develops and/or requests...

We also plan on upgrading the base version from CuPy v2.1.0 to a latter version after releasing ClPy v2.1.0.

Check [github's issues and pull requests](https://github.com/fixstars/clpy/issues) to get the latest status.

## License

MIT License (see `LICENSE` file).

## Reference

Tomokazu Higuchi, Naoki Yoshifuji, Tomoya Sakai, Yoriyuki Kitta, Ryousei Takano, Tsutomu Ikegami, Kenjiro Taura (2019): "ClPy: A NumPy-compatible Library Accelerated with OpenCL", *2019 IEEE International Parallel and Distributed Processing Symposium Workshops*, pp.933-940, [doi:10.1109/IPDPSW.2019.00159](https://doi.org/10.1109/IPDPSW.2019.00159). [Presentation @ Scalable Deep Learning over Parallel and Distributed Infrastructures 2019](https://docs.google.com/presentation/d/1UtZgK9La7Pz_3Qwm2hXg13TwkvKuCSLlKsGebBXY2EA)
