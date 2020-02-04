# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaTemplateInstantiation(unittest.TestCase):

    def test_template_function_definition(self):
        x = '''
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename>
            void f(){}
            ''')
        self.assertEqual(x[1:], y)

    def test_template_function_instantiation(self):
        x = '''
void f___left_angle__int__right_angle__() 
{
}
int main() 
{
    f___left_angle__int__right_angle__();
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename>
            void f(){}
            int main(){
              f<int>();
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_template_function_argument_type_deduction(self):
        x = '''
void f___left_angle__int__right_angle__(int t) 
{
}
int main() 
{
    f___left_angle__int__right_angle__(42);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename T>
            void f(T t){}
            int main(){
              f(42);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_template_class_definition(self):
        x = '''
;
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename>
            struct A{};
            ''')
        self.assertEqual(x[1:], y)

    def test_template_class_instantiation(self):
        x = '''
;
typedef struct A___left_angle__int__right_angle__ {
}A___left_angle__int__right_angle__;
A___left_angle__int__right_angle__ a;
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename>
            struct A{};
            A<int> a;
            ''')
        self.assertEqual(x[1:], y)

    def test_template_class_dependent_type_member(self):
        x = '''
;
typedef struct A___left_angle__int__right_angle__ {
    int t;
}A___left_angle__int__right_angle__;
void f() 
{
    A___left_angle__int__right_angle__ a;
    a.t = 0;
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename T>
            struct A{
              T t;
            };
            void f(){
              A<int> a;
              a.t = 0;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_non_type_template_argument(self):
        x = '''
int f___left_angle__30__right_angle__() 
{
    return 30;
}
int main() 
{
    int a = f___left_angle__30__right_angle__();
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<int N>
            int f(){
              return N;
            }
            int main(){
              int a = f<30>();
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_two_argument_template(self):
        x = '''
void f___left_angle__int__right_angle____left_angle__double__right_angle__() 
{
    int t;
    double u;
}
int main() 
{
    f___left_angle__int__right_angle____left_angle__double__right_angle__();
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            template<typename T, typename U>
            void f(){
              T t;
              U u;
            }
            int main(){
              f<int, double>();
            }
            ''')
        self.assertEqual(x[1:], y)


if __name__ == "__main__":
    unittest.main()
