# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaHalfTrick(unittest.TestCase):

    def test_type_half(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + ('''
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
            '#include <cupy/carray.hpp>')
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
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + ('''
void f() 
{
    int __clpy__half = 42;
}
''')[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              int __clpy__half = 42;
            }
            ''',
            '#include <cupy/carray.hpp>')
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
