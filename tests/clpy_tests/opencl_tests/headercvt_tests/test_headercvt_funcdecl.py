import unittest

import headercvt_test_utils as util


class TestHeadercvtFuncDecl(unittest.TestCase):
    def setUp(self):
        util.check_existence_of_headercvt()

    @util.with_temp_wd
    def test_headercvt_funcdecl_accept_case(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        void clSomeFunction(int, void *);
        """)
        self.assertTrue(util.compile_with(wd, "clSomeFunction(10, <void*>0)"))

    @util.with_temp_wd
    def test_headercvt_funcdecl_decline_case(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        void SomeFunction(int, void *);
        """)
        self.assertTrue(not util.contains(
            results["func_decl"], "SomeFunction"))
        self.assertTrue(util.compile_with(wd, ""))
