import unittest

from clpy import testing


@testing.gpu
class TestTrigonometric(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_binary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_unit(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return getattr(xp, name)(a)

    def test_sin(self):
        self.check_unary('sin')
        self.check_unary_8bit('sin')

    def test_cos(self):
        self.check_unary('cos')
        self.check_unary_8bit('cos')

    def test_tan(self):
        self.check_unary('tan')
        self.check_unary_8bit('tan')

    def test_arcsin(self):
        self.check_unary_unit('arcsin')

    def test_arccos(self):
        self.check_unary_unit('arccos')

    def test_arctan(self):
        self.check_unary('arctan')
        self.check_unary_8bit('arctan')

    def test_arctan2(self):
        self.check_binary('arctan2')
        self.check_binary_8bit('arctan2')

    def test_hypot(self):
        self.check_binary('hypot')
        self.check_binary_8bit('hypot')

    def test_deg2rad(self):
        self.check_unary('deg2rad')
        self.check_unary_8bit('deg2rad')

    def test_rad2deg(self):
        self.check_unary('rad2deg')
