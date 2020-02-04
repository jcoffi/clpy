# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaCArray(unittest.TestCase):

    def test_carray_argument_mutation(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + '''
void f(__global int* const __restrict__ arr, const CArray_2 arr_info) 
{
}
'''[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(CArray<int, 2> arr){}
            ''',
            '#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)

    def test_carray_member_function(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + '''
void f(__global int* const __restrict__ arr, const CArray_2 arr_info) 
{
    ((const size_t)arr_info.size_);
    ((const size_t*)arr_info.shape_);
    ((const size_t*)arr_info.strides_);
}
'''[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(CArray<int, 2> arr){
              arr.size();
              arr.shape();
              arr.strides();
            }
            ''',
            '#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)

    def test_carray_0_member_function(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + '''
void f(__global int* const __restrict__ arr, const CArray_0 arr_info) 
{
    ((const size_t)arr_info.size_);
    ((const size_t*)NULL);
    ((const size_t*)NULL);
}
'''[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(CArray<int, 0> arr){
              arr.size();
              arr.shape();
              arr.strides();
            }
            ''',
            '#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)

    def test_carray_1_member_function(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + '''
void f(__global int* const __restrict__ arr, const CArray_1 arr_info) 
{
    ((const size_t)arr_info.size_);
    ((const size_t*)&arr_info.shape_);
    ((const size_t*)&arr_info.strides_);
}
'''[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(CArray<int, 1> arr){
              arr.size();
              arr.shape();
              arr.strides();
            }
            ''',
            '#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
