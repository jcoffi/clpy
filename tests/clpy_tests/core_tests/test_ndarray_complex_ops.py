import unittest

from clpy import testing


@testing.gpu
class TestConj(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_clpy_array_almost_equal()
    def test_conj(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.conj()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_almost_equal()
    def test_conj_pass(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        y = x.conj()
        self.assertIs(x, y)
        return y


@testing.gpu
class TestAngle(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_8bit_integer=True)
    @testing.numpy_clpy_array_almost_equal()
    def test_angle(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return xp.angle(x)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_array_almost_equal()
    def test_angle_8bit(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return xp.angle(x)


@testing.gpu
class TestRealImag(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_clpy_array_almost_equal(accept_error=False)
    def test_real(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.real

    @testing.for_all_dtypes()
    @testing.numpy_clpy_array_almost_equal(accept_error=False)
    def test_imag(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.imag

    @testing.for_all_dtypes()
    @testing.numpy_clpy_array_almost_equal(accept_error=False)
    def test_real_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        x.real = testing.shaped_reverse_arange((2, 3), xp, dtype).real
        return x

    '''
    The reason for comment out is that ClPy does not support complex
    numbers, and there is no need to perform tests to handle imaginary parts.
    '''

#    @testing.for_dtypes('FD')
#    @testing.numpy_clpy_array_almost_equal(accept_error=False)
#    def test_imag_setter(self, xp, dtype):
#        x = testing.shaped_arange((2, 3), xp, dtype)
#        x.imag = testing.shaped_reverse_arange((2, 3), xp, dtype).real
#        return x

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_raises(accept_error=TypeError)
    def test_imag_setter_raise(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        x.imag = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x
