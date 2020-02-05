# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaHalfTrick(unittest.TestCase):

    def test_type_half(self):
        supports_cl_khr_fp16 = clpy.backend.opencl.env.supports_cl_khr_fp16()
        options = ('-D__CLPY_ENABLE_CL_KHR_FP16'
                   if supports_cl_khr_fp16
                   else '', )
        x = clpy.backend.ultima.exec_ultima('',
                                            '#include <cupy/carray.hpp>',
                                            _options=options) + ('''
__clpy__half f() 
{
    __clpy__half a = 42.F;
    return a;
}
''' if supports_cl_khr_fp16 else '''
__clpy__half f() 
{
    __clpy__half a;constructor___clpy__half___left_paren____clpy__half_float__right_paren__(&a, 42.F);
    return a;
}
''')[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            half f(){
              half a = 42.f;
              return a;
            }
            ''',
            '#include <cupy/carray.hpp>',
            _options=options)
        self.assertEqual(x, y)

    def test_variable_named_half(self):
        x = '''
void f() 
{
    int __clpy__half = 1 / 2;
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              int half = 1/2;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_argument_named_half(self):
        x = '''
void f(int __clpy__half) 
{
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(int half){}
            ''')
        self.assertEqual(x[1:], y)

    def test_clpy_half(self):
        supports_cl_khr_fp16 = clpy.backend.opencl.env.supports_cl_khr_fp16()
        options = ('-D__CLPY_ENABLE_CL_KHR_FP16'
                   if supports_cl_khr_fp16
                   else '', )
        x = clpy.backend.ultima.exec_ultima('',
                                            '#include <cupy/carray.hpp>',
                                            _options=options) + ('''
void f() 
{
    __clpy__half half_ = 42.F;
    int __clpy__half = half_;
}
''' if supports_cl_khr_fp16 else '''
void f() 
{
    __clpy__half half_;constructor___clpy__half___left_paren____clpy__half_float__right_paren__(&half_, 42.F);
    int __clpy__half = operatorfloat___clpy__half_(&half_);
}
''')[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              __clpy__half half_ = 42.f;
              int __clpy__half = half_;
            }
            ''',
            '#include <cupy/carray.hpp>',
            _options=options)
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
