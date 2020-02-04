# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaHalfTrick(unittest.TestCase):

    def test_type_half(self):
        x = '''
__clpy__half f() 
{
    __clpy__half a;
    return (__clpy__half)(42.F);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            half f(){
              half a;
              return static_cast<half>(42.f);
            }
            ''')
        self.assertEqual(x[1:], y)

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
        x = '''
void f() 
{
    __clpy__half half_ = 42.F;
    int __clpy__half = half_;
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              __clpy__half half_ = 42.f;
              int __clpy__half = half_;
            }
            ''')
        self.assertEqual(x[1:], y)


if __name__ == "__main__":
    unittest.main()
