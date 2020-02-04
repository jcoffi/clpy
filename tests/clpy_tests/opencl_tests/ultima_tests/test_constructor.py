# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaConstructor(unittest.TestCase):

    def test_user_defined_default_constructor(self):
        x = '''
typedef struct A {
    }A;
void constructor_A(A*const this) 
    {
    }
void f() 
{
    A a;constructor_A(&a);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            class A{
            public:
              A(){}
            };
            void f(){
              A a;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_explicit_constructor(self):
        x = '''
typedef struct A {
    int a;
    }A;
void constructor_A(A*const this, int t) 
    {
    this->a = t;
    }
void f() 
{
    A a;constructor_A(&a, 3);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            class A{
              int a;
            public:
              explicit A(int t):a{t}{}
            };
            void f(){
              A a{3};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_one_arg_constructor(self):
        x = '''
typedef struct A {
    int a;
    }A;
void constructor_A(A*const this, int t) 
    {
    this->a = t;
    }
void f() 
{
    A a;constructor_A(&a, 3);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            class A{
              int a;
            public:
              A(int t):a{t}{}
            };
            void f(){
              A a{3};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_one_arg_constructor_implicit_conversion(self):
        x = '''
typedef struct A {
    int a;
    }A;
void constructor_A(A*const this, int t) 
    {
    this->a = t;
    }
void f() 
{
    A a;constructor_A(&a, 3);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            class A{
              int a;
            public:
              A(int t):a{t}{}
            };
            void f(){
              A a = 3;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_two_arg_constructor(self):
        x = '''
typedef struct A {
    int a;
    float b;
    }A;
void constructor_A(A*const this, int t, float u) 
    {
    this->a = t;
    this->b = u;
    }
void f() 
{
    A a;constructor_A(&a, 42, 3.1400001F);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            class A{
              int a;
              float b;
            public:
              explicit A(int t, float u):a{t}, b{u}{}
            };
            void f(){
              A a{42, 3.14f};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_parenthesis_initializer_list_constructor(self):
        x = '''
typedef struct A {
    int a;
    float b;
    }A;
void constructor_A(A*const this, int t, float u) 
    {
    this->a = t;
    this->b = u;
    }
void f() 
{
    A a;constructor_A(&a, 42, 3.1400001F);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            class A{
              int a;
              float b;
            public:
              explicit A(int t, float u):a(t), b(u){}
            };
            void f(){
              A a{42, 3.14f};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_system_defined_default_constructor(self):
        x = '''
typedef struct A {
}A;
void f() 
{
    A a;
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            struct A{
            };
            void f(){
              A a;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_explicitlly_defaulted_default_constructor(self):
        x = '''
typedef struct A {
    }A;
void constructor_A(A*const this){};
void f() 
{
    A a;
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            struct A{
              A() = default;
            };
            void f(){
              A a;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_default_constructor_variable_group(self):
        x = '''
typedef struct A {
}A;
void f() 
{
    A a, b;
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            struct A{
            };
            void f(){
              A a, b;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_user_defined_default_constructor_variable_group(self):
        x = '''
typedef struct A {
    }A;
void constructor_A(A*const this) 
    {
    }
void f() 
{
    A a, b;constructor_A(&a);constructor_A(&b);
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            struct A{
              A(){}
            };
            void f(){
              A a, b;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_unnamed_struct(self):
        x = '''
void f() 
{
    struct {
        int t;
    } a,  b = {4},  c = {3};
}
'''
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(){
              struct{
                int t;
              }a, b = {4}, c{3};
            }
            ''')
        self.assertEqual(x[1:], y)


if __name__ == "__main__":
    unittest.main()
