# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaCastConversion(unittest.TestCase):

    def test_function_style_cast(self):
        x = '''
void f() 
{
    (int)(3.F);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              int(3.F);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_static_cast(self):
        x = '''
void f() 
{
    (int)(3.F);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              static_cast<int>(3.F);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_const_cast(self):
        x = '''
void f() 
{
    const int a = 3;
    (int *)(&a);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              const int a = 3;
              const_cast<int*>(&a);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_reinterpret_cast(self):
        x = '''
void f() 
{
    int a = 3;
    (float *)(&a);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              int a = 3;
              reinterpret_cast<float*>(&a);
            }
            ''')
        self.assertEqual(x[1:], y)


if __name__ == "__main__":
    unittest.main()
