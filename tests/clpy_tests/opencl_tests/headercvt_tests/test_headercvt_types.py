import unittest

import headercvt_test_utils as util


class TestHeadercvtTypes(unittest.TestCase):
    def setUp(self):
        util.check_existence_of_headercvt()

    @util.with_temp_wd
    def test_headercvt_typedef(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef int clpy_int;
        """)
        self.assertTrue(util.compile_with(wd, "cdef clpy_int foo = 0"))

    @util.with_temp_wd
    def test_headercvt_typedef_to_pointer(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef int* clpy_intptr;
        """)
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_intptr foo = <clpy_intptr>0"))

    @util.with_temp_wd
    def test_headercvt_typedef_to_tagged_struct(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int member;
        } clpy_struct_t;
        """)
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member = 0"))

    @util.with_temp_wd
    def test_headercvt_typedef_to_anonymous_struct(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef struct {
            int member;
        } clpy_struct_t;
        """)
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member = 0"))

    @util.with_temp_wd
    def test_headercvt_typedef_to_empty_struct(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef struct {
        } clpy_empty_struct_t;
        """)
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_empty_struct_t* foo"))

    @util.with_temp_wd
    def test_headercvt_typedef_to_struct_which_contains_an_array(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int member[100];
        } clpy_struct_t;
        """)
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member[0] = 0"))

    @util.with_temp_wd
    def test_headercvt_typedef_to_struct_having_a_pointer_variable(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int* ptr;
        } clpy_struct_t;
        """)
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.ptr = <int*>0"))

    """
    @util.with_temp_wd
    def test_headercvt_typedef_to_discretely_tagged_struct(self, wd):
        util.kick_headercvt_and_get_results(wd, \"""
        struct clpy_struct_tag{
            int member;
        };
        typedef struct clpy_struct_tag clpy_struct_t;
        \""")
        self.assertTrue(util.compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member = 0"))
        # TODO(nsakabe-fixstars):
        # Make headercvt support this case of decl and update this testcase.
        # Current error:
        # cdef extern from "CL/cl.h":
        #    cdef struct clpy_struct_tag:
        #        int member
        #    ctypedef struct clpy_struct_tag clpy_struct_t
        #                                   ^
        # ------------------------------------------------------------
        #
        # types.pxi:4:36: Syntax error in struct or union definition
    """

    @util.with_temp_wd
    def test_headercvt_typedef_to_implicitly_declared_pointer(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag *    clpy_pointer_to_struct_t;
        """)
        self.assertTrue(util.compile_with(
            wd,
            "cdef clpy_pointer_to_struct_t foo = <clpy_pointer_to_struct_t>0"))

    @util.with_temp_wd
    def test_headercvt_ignore_typedef_to_function_pointer(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        typedef void(*clpy_function_i_vstar_l_t)(int, void*, long);
        """)
        self.assertTrue(not util.contains(
            results["types"], "clpy_function_i_vstar_l_t"))
        self.assertTrue(util.compile_with(wd, ""))

    @util.with_temp_wd
    def test_headercvt_ignore_union_groupdecl(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        typedef union clpy_union_tag{
            int member;
        } clpy_union_t;
        """)
        self.assertTrue(not util.contains(results["types"], "clpy_union_t"))
        self.assertTrue(util.compile_with(wd, ""))

    @util.with_temp_wd
    def test_headercvt_ignore_union_recorddecl(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        union clpy_union_tag{
            int member;
        };
        """)
        self.assertTrue(util.compile_with(wd, ""))

    @util.with_temp_wd
    def test_headercvt_ignore_union_reference(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        typedef union clpy_union_tag{
            int member;
        } clpy_union_t;
        typedef clpy_union_t clpy_typedefed_union_t;
        """)
        self.assertTrue(not util.contains(
            results["types"], "clpy_typedefed_union_t"))
        self.assertTrue(util.compile_with(wd, ""))

    @util.with_temp_wd
    def test_headercvt_ignore_pthread_related_groupdecl(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        typedef struct{
            int* ptr;
        } clpy_pthread_struct_intptr;
        """)
        self.assertTrue(not util.contains(
            results["types"], "clpy_pthread_struct_intptr"))
        self.assertTrue(util.compile_with(wd, ""))

    @util.with_temp_wd
    def test_headercvt_ignore_pthread_related_recorddecl(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int* ptr;
        } clpy_pthread_struct_intptr;
        """)
        self.assertTrue(not util.contains(
            results["types"], "clpy_pthread_struct_intptr"))
        self.assertTrue(util.compile_with(wd, ""))
