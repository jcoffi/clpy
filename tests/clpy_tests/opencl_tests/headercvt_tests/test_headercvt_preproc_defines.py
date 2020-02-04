import unittest

import headercvt_test_utils as util


class TestHeadercvtPreprocDefines(unittest.TestCase):
    def setUp(self):
        util.check_existence_of_headercvt()

    @util.with_temp_wd
    def test_headercvt_preproc_define_accept_case(self, wd):
        util.kick_headercvt_and_get_results(wd, """
        #define CL_SOME_VALUE 1
        """)
        self.assertTrue(util.compile_with(wd, "print(CL_SOME_VALUE)"))

    @util.with_temp_wd
    def test_headercvt_preproc_define_decline_case(self, wd):
        results = util.kick_headercvt_and_get_results(wd, """
        #define SOME_VALUE 1
        """)
        self.assertTrue(not util.contains(
            results["preprocessor_defines"], "SOME_VALUE"))
        self.assertTrue(util.compile_with(wd, ""))
