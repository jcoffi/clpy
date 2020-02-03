import unittest

from clpy import testing


@testing.gpu
class TestRounding(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['h', 'i', 'q', 'f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_dtypes(['?', 'b'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_negative_8bit(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    def test_rint(self):
        self.check_unary('rint')
        self.check_unary_8bit('rint')

    def test_rint_negative(self):
        self.check_unary_negative('rint')
        self.check_unary_negative_8bit('rint')

    def test_floor(self):
        self.check_unary('floor')
        self.check_unary_8bit('floor')

    def test_ceil(self):
        self.check_unary('ceil')
        self.check_unary_8bit('ceil')

    def test_trunc(self):
        self.check_unary('trunc')
        self.check_unary_8bit('trunc')

    def test_fix(self):
        self.check_unary('fix')
        self.check_unary_8bit('fix')
