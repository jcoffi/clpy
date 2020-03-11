import unittest

import numpy

import clpy
from clpy.core import core
from clpy import testing


def _compile_func(kernel_name, code):
    module = core.compile_with_cache(code)
    return module.get_function(kernel_name)


@testing.gpu
class TestFunction(unittest.TestCase):

    def test_python_scalar(self):
        code = '''
__kernel void test_kernel(
    __global const double* a, CArray_2 ai,
    double b,
    __global double* x, CArray_2 xi) {
  int i = get_local_id(0);
  x[get_CArrayIndexI_2(&xi, i)/sizeof(double)]
    = a[get_CArrayIndexI_2(&ai, i)/sizeof(double)] + b;
}
'''

        a_cpu = numpy.arange(24, dtype=numpy.float64).reshape((4, 6))
        a = clpy.array(a_cpu)
        b = float(2)
        x = clpy.empty_like(a)

        func = _compile_func('test_kernel', code)  # NOQA

        func.linear_launch(a.size, (a, b, x))

        expected = a_cpu + b
        testing.assert_array_equal(x, expected)

    def test_numpy_scalar(self):
        code = '''
__kernel void test_kernel(
    __global const double* a, CArray_2 ai,
    double b,
    __global double* x, CArray_2 xi) {
  int i = get_local_id(0);
  x[get_CArrayIndexI_2(&xi, i)/sizeof(double)]
    = a[get_CArrayIndexI_2(&ai, i)/sizeof(double)] + b;
}
'''

        a_cpu = numpy.arange(24, dtype=numpy.float64).reshape((4, 6))
        a = clpy.array(a_cpu)
        b = numpy.float64(2)
        x = clpy.empty_like(a)

        func = _compile_func('test_kernel', code)  # NOQA

        func.linear_launch(a.size, (a, b, x))

        expected = a_cpu + b
        testing.assert_array_equal(x, expected)

    def test_not_supported_scalar(self):
        code = '''
__kernel void test_kernel(char a) {
}
'''

        a = "2"

        func = _compile_func('test_kernel', code)  # NOQA

        with self.assertRaises(TypeError):
            func.linear_launch(1, a)
