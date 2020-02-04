# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaOverloadMangling(unittest.TestCase):

    def test_functions_overloading(self):
        x = '''
void f() 
{
}
void f__left_paren__int__right_paren__(int arg) 
{
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){}
            void f(int arg){}
            ''')
        self.assertEqual(x[1:], y)

    def test_call_overloaded_functions(self):
        x = '''
void f() 
{
}
void f__left_paren__int__right_paren__(int arg) 
{
}
int main() 
{
    f();
    f__left_paren__int__right_paren__(42);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){}
            void f(int arg){}
            int main(){
              f();
              f(42);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_two_or_more_arguments_function_mangling(self):
        x = '''
int f() 
{
    return 0;
}
int f__left_paren__int__comma__double__right_paren__(int arg1, double arg2) 
{
    return 1;
}
int f__left_paren__constchar__pointer____comma__size_t__comma____dot____dot____dot____right_paren__(const char *str, size_t len, ...) 
{
    return 2;
}
int main() 
{
    f();
    f__left_paren__int__comma__double__right_paren__(42, 42.);
    f__left_paren__constchar__pointer____comma__size_t__comma____dot____dot____dot____right_paren__("abc", sizeof ("abc") / sizeof(char), 42, 42, 42, 42., 42.F, 42U, 42L, 42);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            int f(){return 0;}
            int f(int arg1, double arg2){return 1;}
            int f(const char* str, size_t len, ...){return 2;}
            int main(){
              f();
              f(42, 42.);
              f("abc", sizeof("abc")/sizeof(char), 42, 42, 42, 42., 42.f, 42u, 42l, 42);
            }
            ''')
        self.maxDiff = None
        self.assertEqual(x[1:], y)

    def test_no_mangle(self):
        x = '''
void f(int a) 
{
}
void f(double a) 
{
}
int main() 
{
    f(42);
    f(42.);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(int a){}
            __attribute__((annotate("clpy_no_mangle"))) void f(double a){}
            int main(){
              f(42);
              f(42.);
            }
            ''')
        self.assertEqual(x[1:], y)


if __name__ == "__main__":
    unittest.main()
