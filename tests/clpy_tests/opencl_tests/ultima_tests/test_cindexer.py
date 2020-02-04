# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import clpy
import unittest


class TestUltimaCIndexer(unittest.TestCase):

    def test_cindexer_argument_mutation(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + '''
void f(CIndexer_2 ind) 
{
}
'''[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(CIndexer<2> ind){}
            ''',
            '#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)

    def test_cindexer_member_function(self):
        x = clpy.backend.ultima.exec_ultima('', '#include <cupy/carray.hpp>') + '''
void f(CIndexer_2 ind) 
{
    ind_size;
}
'''[1:]
        y = clpy.backend.ultima.exec_ultima(
            '''
            void f(CIndexer<2> ind){
              ind.size();
            }
            ''',
            '#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
