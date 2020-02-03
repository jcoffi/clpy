import unittest

from clpy import testing


@testing.gpu
class TestHyperbolic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_unit(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return getattr(xp, name)(a)

    def test_sinh(self):
        self.check_unary('sinh')
        self.check_unary_8bit('sinh')

    def test_cosh(self):
        self.check_unary('cosh')
        self.check_unary_8bit('cosh')

    def test_tanh(self):
        self.check_unary('tanh')
        self.check_unary_8bit('tanh')

    def test_arcsinh(self):
        self.check_unary('arcsinh')
        self.check_unary_8bit('arcsinh')

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    def test_arccosh(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        return xp.arccosh(a)

    def test_arctanh(self):
        self.check_unary_unit('arctanh')
