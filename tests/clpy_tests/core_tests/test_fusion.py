import mock
import numpy
import six
import unittest

import clpy
from clpy import testing


def fusion_default_array_equal():
    def res_func(func):
        def res(xxx, name, xp, dtype):
            f = getattr(clpy, name)

            @clpy.fuse(input_num=f.nin)
            def g(*args):
                return f(*args)

            val = func(xxx, name, xp, dtype)
            return g(*val)
        return res
    return res_func


@testing.gpu
class TestFusionElementwise(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_int_dtypes()
    @testing.numpy_clpy_array_equal()
    @fusion_default_array_equal()
    def check_unary_int(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 0, 1, 2, 3], dtype=dtype)
        return (a,)

    @testing.for_int_dtypes()
    @testing.numpy_clpy_array_equal()
    @fusion_default_array_equal()
    def check_binary_int(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 0, 1, 2, 3], dtype=dtype)
        b = xp.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype)
        return a, b

    def test_bitwise_and(self):
        self.check_binary_int('bitwise_and')

    def test_bitwise_or(self):
        self.check_binary_int('bitwise_or')

    def test_bitwise_xor(self):
        self.check_binary_int('bitwise_xor')

    def test_invert(self):
        self.check_unary_int('invert')

    def test_left_shift(self):
        self.check_binary_int('left_shift')

    def test_right_shift(self):
        self.check_binary_int('right_shift')


@testing.gpu
class TestFusionComparison(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    def test_greater(self):
        self.check_binary('greater')

    def test_greater_equal(self):
        self.check_binary('greater_equal')

    def test_less(self):
        self.check_binary('less')

    def test_less_equal(self):
        self.check_binary('less_equal')

    def test_not_equal(self):
        self.check_binary('not_equal')

    def test_equal(self):
        self.check_binary('equal')


@testing.gpu
class TestFusionContent(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes()
    @testing.numpy_clpy_array_equal()
    @fusion_default_array_equal()
    def check_unary_inf(self, name, xp, dtype):
        a = xp.array([-3, dtype('inf'), -1, -dtype('inf'), 0, 1, 2],
                     dtype=dtype)
        return (a,)

    @testing.for_float_dtypes()
    @testing.numpy_clpy_array_equal()
    @fusion_default_array_equal()
    def check_unary_nan(self, name, xp, dtype):
        a = xp.array(
            [-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, dtype('inf')],
            dtype=dtype)
        return (a,)

    def test_isfinite(self):
        self.check_unary_inf('isfinite')

    def test_isinf(self):
        self.check_unary_inf('isinf')

    def test_isnan(self):
        self.check_unary_nan('isnan')


@testing.gpu
class TestFusionOps(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    def test_logical_and(self):
        self.check_binary('logical_and')

    def test_logical_or(self):
        self.check_binary('logical_or')

    def test_logical_xor(self):
        self.check_binary('logical_xor')

    def test_logical_not(self):
        self.check_unary('logical_not')


@testing.gpu
class TestFusionTrigonometric(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_unit(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return (a,)

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
        self.check_unary_8bit('rad2deg')


@testing.gpu
class TestFusionHyperbolic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_unit1(self, name, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        return (a,)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_unit2(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return (a,)

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

    def test_arccosh(self):
        self.check_unary_unit1('arccosh')

    def test_arctanh(self):
        self.check_unary_unit2('arctanh')


@testing.gpu
class TestFusionRounding(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_dtypes(['h', 'i', 'q', 'f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return (a,)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_dtypes(['?', 'b'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_negative_8bit(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return (a,)

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


@testing.gpu
class TestFusionExplog(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    def test_exp(self):
        self.check_unary('exp')
        self.check_unary_8bit('exp')

    def test_expm1(self):
        self.check_unary('expm1')
        self.check_unary_8bit('expm1')

    def test_exp2(self):
        self.check_unary('exp2')
        self.check_unary_8bit('exp2')

    def test_log(self):
        with testing.NumpyError(divide='ignore'):
            self.check_unary('log')
            self.check_unary_8bit('log')

    def test_log10(self):
        with testing.NumpyError(divide='ignore'):
            self.check_unary('log10')
            self.check_unary_8bit('log10')

    def test_log2(self):
        with testing.NumpyError(divide='ignore'):
            self.check_unary('log2')
            self.check_unary_8bit('log2')

    def test_log1p(self):
        self.check_unary('log1p')
        self.check_unary_8bit('log1p')

    def test_logaddexp(self):
        self.check_binary('logaddexp')
        self.check_binary_8bit('logaddexp')

    def test_logaddexp2(self):
        self.check_binary('logaddexp2')
        self.check_binary_8bit('logaddexp2')


@testing.gpu
class TestFusionFloating(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_8bit(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.for_float_dtypes(name='ftype')
    @testing.for_dtypes(['i', 'l'], name='itype')
    @testing.numpy_clpy_allclose()
    def test_ldexp(self, xp, ftype, itype):
        a = xp.array([-3, -2, -1, 0, 1, 2, 3], dtype=ftype)
        b = xp.array([-3, -2, -1, 0, 1, 2, 3], dtype=itype)

        @clpy.fuse()
        def g(x, y):
            return clpy.ldexp(x, y)

        return g(a, b)

    def test_signbit(self):
        self.check_unary('signbit')

    def test_copysign(self):
        self.check_binary('copysign')
        self.check_binary_8bit('copysign')

    @testing.for_float_dtypes()
    def test_frexp(self, dtype):
        numpy_a = numpy.array([-300, -20, -10, -1, 0, 1, 10, 20, 300],
                              dtype=dtype)

        @clpy.fuse()
        def g(x):
            return clpy.frexp(x)

        numpy_b, numpy_c = g(numpy_a)

        clpy_a = clpy.array(numpy_a)
        clpy_b, clpy_c = g(clpy_a)

        testing.assert_allclose(clpy_b, numpy_b)
        testing.assert_array_equal(clpy_c, numpy_c)

    def test_nextafter(self):
        self.check_binary('nextafter')
        self.check_binary_8bit('nextafter')


@testing.gpu
class TestFusionArithmetic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return (a,)

    @testing.for_all_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_without_complex(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_without_bool(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return a, b

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return (a,)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        b = xp.array([4, 3, 2, 1, -1, -2], dtype=dtype)
        return a, b

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    @fusion_default_array_equal()
    def check_binary_negative_float(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        b = xp.array([4, 3, 2, 1, -1, -2], dtype=dtype)
        return a, b

    def test_add(self):
        self.check_binary('add')

    def test_reciprocal(self):
        with testing.NumpyError(divide='ignore', invalid='ignore'):
            self.check_unary('reciprocal')

    def test_multiply(self):
        self.check_binary('multiply')

    def test_divide(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('divide')

    def test_divide_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('divide')

    def test_power(self):
        self.check_binary('power')

    def test_power_negative(self):
        self.check_binary_negative_float('power')

    def test_subtract(self):
        # TODO(unno): boolean subtract causes DeprecationWarning in numpy>=1.13
        self.check_binary_without_bool('subtract')

    def test_true_divide(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('true_divide')

    def test_true_divide_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('true_divide')

    def test_floor_divide(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_without_complex('floor_divide')

    def test_floor_divide_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('floor_divide')

    def test_fmod(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_without_complex('fmod')

    def test_fmod_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('fmod')

    @testing.for_float_dtypes()
    @testing.numpy_clpy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)

        @clpy.fuse()
        def g(x):
            return clpy.modf(x)

        b, c = g(a)
        d = xp.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d

    def test_remainder(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_without_complex('remainder')

    def test_remainder_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('remainder')


class TestFusionUfunc(unittest.TestCase):

    @clpy.fuse()
    def sample_function(x, y, z):
        return clpy.square(clpy.add(x, y))

    def random_bool(self, seed=0):
        return testing.shaped_random((3, 3),
                                     xp=clpy,
                                     dtype=numpy.bool_,
                                     seed=seed)

    def random_int(self, lower=-1000, higher=1000, seed=0):
        return testing.shaped_random((3, 3),
                                     xp=clpy,
                                     dtype=numpy.int64,
                                     scale=(higher - lower),
                                     seed=seed) + lower

    def random_real(self, lower=-1000, higher=1000, seed=0):
        return testing.shaped_random((3, 3),
                                     xp=clpy,
                                     dtype=numpy.float64,
                                     scale=(higher - lower),
                                     seed=seed) + lower

    def check(self, func, n, gen, args=None):
        if args is None:
            args = ((),) * n
        self._check(func, n, gen, args, True)
        self._check(func, n, gen, args, False)

    def _check(self, func, n, gen, args, omit_nin, error_types=None):
        assert n == len(args), (n, args)
        nin = n if not omit_nin else None
        if error_types is None:
            error_types = ()

        @clpy.fuse(input_num=nin)
        def f(*x):
            return func(*x)

        # Prepare input arrays
        if not isinstance(gen, tuple):
            gen = (gen,) * n
        data0 = tuple([g(*a) for g, a in zip(gen, args)])
        data1 = tuple([_.copy() for _ in data0])

        # Invoke non-fused function
        try:
            ret0 = func(*data0)  # Non-fused
            err0 = None
        except Exception as e:
            if type(e) not in error_types:
                raise
            ret0 = None
            err0 = e

        # Invoke fused function
        try:
            ret1 = f(*data1)  # Fused
            err1 = None
        except Exception as e:
            if type(e) not in error_types:
                raise
            ret1 = None
            err1 = e

        self.assertEqual(err0 is None, err1 is None)
        if err0 is not None:
            # Error
            self.assertEqual(type(err0), type(err1))
            self.assertEqual(str(err0), str(err1))
            arrs0 = None
            arrs1 = None
        else:
            # Success
            self.assertEqual(ret0 is None, ret1 is None)
            if ret0 is None:
                # Both return values are None
                ret0 = ()
                ret1 = ()
            else:
                # Return values must have the same type
                self.assertEqual(type(ret0), type(ret1))

                if not isinstance(ret0, tuple):
                    # Single arrays are returned
                    ret0 = (ret0,)
                    ret1 = (ret1,)
                else:
                    # Tuples are returned
                    self.assertEqual(len(ret0), len(ret1))

            # Concatenate return values and input arrays
            arrs0 = ret0 + data0
            arrs1 = ret1 + data1

            # Test they have same values
            for nf, f in zip(arrs0, arrs1):
                numpy.testing.assert_array_almost_equal(nf.get(), f.get())

        return err0 is not None, (arrs0, arrs1)

    def check_reduce(self, func, n, reduce_f, gen, args=None):
        if args is None:
            args = ((),) * n
        self._check_reduce(func, n, reduce_f, gen, args, True)
        self._check_reduce(func, n, reduce_f, gen, args, False)

    def _check_reduce(self, func, n, reduce_f, gen, args, omit_nin):
        nin = n if not omit_nin else None

        @clpy.fuse(input_num=nin, reduce=reduce_f)
        def f(*x):
            return func(*x)

        data = [gen(*a) for a in args]
        ret0 = reduce_f(func(*data))  # Non-fused
        ret1 = f(*data)  # Fused
        numpy.testing.assert_array_almost_equal(ret0.get(), ret1.get())

    @testing.for_all_dtypes_combination(
        names=['src_dtype', 'dst_dtype'], full=True, no_complex=True)
    def test_out_arg(self, src_dtype, dst_dtype):

        def func(x, y, z):
            return clpy.add(x, y, out=z)

        dtypes = (src_dtype, src_dtype, dst_dtype)
        ret = self._check(
            func, 3,
            lambda iarg: clpy.arange(6).astype(dtypes[iarg]).reshape((2, 3)),
            [(_,) for _ in range(3)],
            True, error_types=(TypeError,))
        is_err, (arrs_n, arrs_f) = ret

        if not is_err:
            # The returned array must equal to z
            arr_ret = arrs_f[0]
            arr_z = arrs_f[3]
            testing.assert_array_equal(arr_ret, arr_z)

    def test_out_arg2(self):

        def func(x, y, z, u, v):
            clpy.add(x, y, out=z)
            clpy.subtract(z, x, out=u)
            clpy.multiply(z, x, out=v)
            return u

        ret = self._check(
            func, 5, self.random_int, ((),) * 5,
            True, error_types=(TypeError,))
        is_err, (arrs_n, arrs_f) = ret

        # Must succeed
        self.assertFalse(is_err)

        # The returned array must equal to u
        arr_ret = arrs_f[0]
        arr_u = arrs_f[4]
        testing.assert_array_equal(arr_ret, arr_u)

    def test_bitwise(self):
        self.check(clpy.bitwise_and, 2, self.random_int)
        self.check(clpy.bitwise_or, 2, self.random_int)
        self.check(clpy.bitwise_xor, 2, self.random_int)
        self.check(clpy.invert, 1, self.random_int)
        self.check(clpy.left_shift, 2, self.random_int, ((0, 20),) * 2)
        self.check(clpy.right_shift, 2, self.random_int, ((0, 20),) * 2)

    def test_compare(self):
        self.check(clpy.greater, 2, self.random_int)
        self.check(clpy.greater_equal, 2, self.random_int)
        self.check(clpy.less, 2, self.random_int)
        self.check(clpy.less_equal, 2, self.random_int)
        self.check(clpy.equal, 2, self.random_int)
        self.check(clpy.not_equal, 2, self.random_int)

    def test_logic_content(self):
        self.check(clpy.isfinite, 1, self.random_real)
        self.check(clpy.isinf, 1, self.random_real)
        self.check(clpy.isnan, 1, self.random_real)

    def test_logic_ops(self):
        self.check(clpy.logical_and, 2, self.random_int, ((0, 2),) * 2)
        self.check(clpy.logical_or, 2, self.random_int, ((0, 2),) * 2)
        self.check(clpy.logical_not, 1, self.random_int, ((0, 2),))
        self.check(clpy.logical_xor, 2, self.random_int, ((0, 2),) * 2)

    def test_trigonometric(self):
        self.check(clpy.sin, 1, self.random_real)
        self.check(clpy.cos, 1, self.random_real)
        self.check(clpy.tan, 1, self.random_real)
        self.check(clpy.arcsin, 1, self.random_real, ((-1, 1),))
        self.check(clpy.arccos, 1, self.random_real, ((-1, 1),))
        self.check(clpy.arctan, 1, self.random_real)
        self.check(clpy.hypot, 2, self.random_real)
        self.check(clpy.deg2rad, 1, self.random_real)
        self.check(clpy.rad2deg, 1, self.random_real)
        self.check(clpy.degrees, 1, self.random_real)
        self.check(clpy.radians, 1, self.random_real)

    def test_hyperbolic(self):
        self.check(clpy.sinh, 1, self.random_real, ((-10, 10),))
        self.check(clpy.cosh, 1, self.random_real, ((-10, 10),))
        self.check(clpy.tanh, 1, self.random_real, ((-10, 10),))
        self.check(clpy.arcsinh, 1, self.random_real, ((-10, 10),))
        self.check(clpy.arccosh, 1, self.random_real, ((1, 10),))
        self.check(clpy.arctanh, 1, self.random_real, ((0, 1),))

    def test_rounding(self):
        self.check(clpy.rint, 1, self.random_real)
        self.check(clpy.floor, 1, self.random_real)
        self.check(clpy.ceil, 1, self.random_real)
        self.check(clpy.trunc, 1, self.random_real)

    def test_explog(self):
        self.check(clpy.exp, 1, self.random_real, ((-10, 10),))
        self.check(clpy.expm1, 1, self.random_real, ((-10, 10),))
        self.check(clpy.exp2, 1, self.random_real, ((-10, 10),))
        self.check(clpy.log, 1, self.random_real, ((0, 10),))
        self.check(clpy.log10, 1, self.random_real, ((0, 10),))
        self.check(clpy.log2, 1, self.random_real, ((0, 10),))
        self.check(clpy.log1p, 1, self.random_real, ((-1, 10),))
        self.check(clpy.logaddexp, 2, self.random_real, ((0, 10),) * 2)
        self.check(clpy.logaddexp2, 2, self.random_real, ((0, 10),) * 2)

    def test_floating(self):
        self.check(clpy.signbit, 1, self.random_real)
        self.check(clpy.copysign, 2, self.random_real)
        self.check(clpy.ldexp, 2, self.random_int, ((1, 10),) * 2)
        self.check(clpy.frexp, 1, self.random_real, ((1, 1000),))
        self.check(clpy.nextafter, 2, self.random_real)

    def test_arithmetic(self):
        self.check(clpy.add, 2, self.random_real)
        self.check(clpy.reciprocal, 1, self.random_real)
        self.check(clpy.negative, 1, self.random_real)
        self.check(clpy.multiply, 2, self.random_real)
        self.check(clpy.divide, 2, self.random_real)
        self.check(clpy.power, 2, self.random_real, ((0, 10),) * 2)
        self.check(clpy.subtract, 2, self.random_real)
        self.check(clpy.true_divide, 2, self.random_int, ((1, 1000),) * 2)
        self.check(clpy.floor_divide, 2, self.random_real, ((1, 1000),) * 2)
        self.check(clpy.fmod, 2, self.random_real)
        self.check(clpy.mod, 2, self.random_int, ((1, 1000),) * 2)
        self.check(clpy.modf, 1, self.random_real)
        self.check(clpy.remainder, 2, self.random_int, ((1, 1000),) * 2)

    def test_misc(self):
        self.check(clpy.sqrt, 1, self.random_real, ((0, 1000),))
        self.check(clpy.square, 1, self.random_real)
        self.check(clpy.absolute, 1, self.random_real)
        self.check(clpy.abs, 1, self.random_real)
        self.check(clpy.sign, 1, self.random_real)
        self.check(clpy.maximum, 2, self.random_real)
        self.check(clpy.minimum, 2, self.random_real)
        self.check(clpy.fmax, 2, self.random_real)
        self.check(clpy.fmin, 2, self.random_real)

    def test_special(self):
        self.check(clpy.where, 3,
                   (self.random_bool,
                    lambda *args: self.random_int(*args, seed=0),
                    lambda *args: self.random_int(*args, seed=1)),
                   ((), (0, 100), (0, 100)))
        self.check(clpy.clip, 3,
                   (lambda *args: self.random_real(*args, seed=0),
                    lambda *args: self.random_real(*args, seed=1),
                    lambda *args: self.random_real(*args, seed=2)),
                   ((0, 1000), (0, 500), (500, 1000)))

    def test_reduce(self):
        self.check_reduce(clpy.bitwise_and, 2, clpy.sum, self.random_int)
        self.check_reduce(clpy.sqrt, 1, clpy.prod, self.random_int, ((1, 2),))
        self.check_reduce(clpy.sqrt, 1, clpy.prod, self.random_real, ((1, 2),))

        self.check_reduce(lambda x: x, 1, clpy.amax, self.random_int)
        self.check_reduce(lambda x: x, 1, clpy.amin, self.random_int)


@testing.gpu
class TestFusionMisc(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange((2, 3), xp, dtype)

        @clpy.fuse()
        def g(x):
            return getattr(clpy, name)(x)

        return g(a)

    @testing.for_all_dtypes(no_complex=True, no_8bit_integer=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_without_8bit(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange((2, 3), xp, dtype)

        @clpy.fuse()
        def g(x):
            return getattr(clpy, name)(x)

        return g(a)

    @testing.skip_when_disabled_cl_khr_fp16
    @testing.for_8bit_integer_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_8bit(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange((2, 3), xp, dtype)

        @clpy.fuse()
        def g(x):
            return getattr(clpy, name)(x)

        return g(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)

        @clpy.fuse()
        def g(x, y):
            return getattr(clpy, name)(x, y)
        return g(a, b)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'f', 'd'])
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x):
            return getattr(clpy, name)(x)

        return g(a)

    @testing.for_float_dtypes()
    @testing.numpy_clpy_array_equal()
    @fusion_default_array_equal()
    def check_binary_nan(self, name, xp, dtype):
        a = xp.array([-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, 2],
                     dtype=dtype)
        b = xp.array([numpy.NAN, numpy.NAN, 1, 0, numpy.NAN, -1, -2],
                     dtype=dtype)
        return a, b

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_external_clip(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)

        @clpy.fuse()
        def g(x, y, z):
            return clpy.clip(x, y, z)

        ty = numpy.dtype(dtype).type
        return g(a, ty(3), ty(13))

    @testing.with_requires('numpy>=1.11.2')
    def test_sqrt(self):
        # numpy.sqrt is broken in numpy<1.11.2
        self.check_unary_without_8bit('sqrt')
        self.check_unary_8bit('sqrt')

    def test_square(self):
        self.check_unary('square')

    def test_absolute(self):
        self.check_unary('absolute')

    def test_absolute_negative(self):
        self.check_unary_negative('absolute')

    def test_sign(self):
        self.check_unary('sign', no_bool=True)

    def test_sign_negative(self):
        self.check_unary_negative('sign', no_bool=True)

    def test_maximum(self):
        self.check_binary('maximum')

    def test_maximum_nan(self):
        self.check_binary_nan('maximum')

    def test_minimum(self):
        self.check_binary('minimum')

    def test_minimum_nan(self):
        self.check_binary_nan('minimum')

    def test_fmax(self):
        self.check_binary('fmax')

    def test_fmax_nan(self):
        self.check_binary_nan('fmax')

    def test_fmin(self):
        self.check_binary('fmin')

    def test_fmin_nan(self):
        self.check_binary_nan('fmin')


@testing.gpu
class TestFusionFuse(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse1(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            w = x * y + z
            (x, w) = (w, x)
            return z * w + y + x

        return g(a, b, c)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse2(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            x += z
            clpy.add(x, y, z)
            return z

        return g(a, b, c)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse3(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            x = 10 + (-x) * (x - y) + 10
            x = 2 * (100 - x - 30)
            x = x / (y + 1 / y)
            return z // x + x // z + 100 // x + 100 // z

        return g(a, b, c)

    @testing.for_int_dtypes()
    @testing.numpy_clpy_array_equal()
    def test_fuse4(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            x = x * y % z + 10 % x << x << y >> z
            return x + (1 << y) + (1 << z) + (120 >> y) + (120 >> y)

        return g(a, b, c)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse5(self, xp, dtype):
        a = xp.arange(15, dtype=dtype)
        b = a * a[::-1]
        a = a * 3 + 11
        c = (a * b) % 63

        @clpy.fuse()
        def g(x, y, z):
            x = ~(x & y) | (x ^ z) ^ (z | y)
            y = 109 & y
            z = 109 | z
            z = 88 ^ z
            return x + y + z

        return g(a, b, c)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse6(self, xp, dtype):
        a = xp.arange(15, dtype=dtype)
        b = a * a[::-1]
        a = a * 3 + 11
        c = (a * b) % 63

        @clpy.fuse()
        def g(x, y, z):
            x = ~(x & y) | (x ^ z) ^ (z | y)
            y &= 109
            z |= 109
            z ^= 88
            return x + y + z

        return g(a, b, c)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse7(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        def toi(x):
            return clpy.where(x, 1, 0)

        @clpy.fuse()
        def g(p, q, r, s, t, u):
            x = toi(p == q) + toi(r < s) + toi(t > u)
            x += toi(p != r) + toi(q <= t) + toi(s >= u)
            return x

        return g(a, b, c, a, b, c)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse8(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        def toi(x):
            return clpy.where(x, 1, 0)

        @clpy.fuse()
        def g(p, q, r):
            x = toi(2 == p) + toi(2 != q) + toi(3 > r)
            y = toi(2 < p) + toi(2 >= q) + toi(3 <= r)
            return x + y << 3

        return g(a, b, c)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse9(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            x *= y
            x += y
            x = x / y
            z %= y
            x += y + z

        g(a, b, c)
        return a

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse10(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            a = x
            a += y
            clpy.add(x, y, z)

        g(a, b, c)
        return c

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse11(self, xp, dtype):
        a = xp.array([2, 2, 2, 2, 3, 3, 3, 3], dtype=dtype)
        b = xp.array([2, 2, 3, 3, 2, 2, 3, 3], dtype=dtype)
        c = xp.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=dtype)

        @clpy.fuse()
        def g(x, y, z):
            a = x
            a += y
            clpy.add(x, y, z)
            return y

        res = g(a, b, c)
        return c + res

    # NumPy 1.9 accepts itruediv between integers
    @testing.with_requires('numpy>=1.10')
    @unittest.skipUnless(six.PY3, 'Only for py3')
    @testing.for_int_dtypes()
    @testing.numpy_clpy_raises()
    def test_fuse_int_itruediv_py3_raises(self, xp, dtype):
        a = xp.array(3, dtype=dtype)
        b = xp.array(2, dtype=dtype)

        @clpy.fuse()
        def g(x, y):
            x /= y

        g(a, b)

    @unittest.skipUnless(six.PY2, 'Only for py2')
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_fuse_int_ifloordiv_py2(self, xp, dtype):
        a = xp.array(3, dtype=dtype)
        b = xp.array(2, dtype=dtype)

        @clpy.fuse()
        def g(x, y):
            x /= y

        g(a, b)
        return a

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_reduce1(self, xp, dtype):
        a = xp.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=dtype)
        b = xp.array([[2, 2, 3, 3], [2, 2, 3, 3]], dtype=dtype)
        c = xp.array([[2, 3, 2, 3], [2, 3, 2, 3]], dtype=dtype)

        @clpy.fuse(reduce=clpy.sum)
        def g(x, y, z):
            return x * y + z

        return g(a, b, c)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_reduce2(self, xp, dtype):
        a = xp.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=dtype)
        b = xp.array([[2, 2, 3, 3], [2, 2, 3, 3]], dtype=dtype)
        c = xp.array([[2, 3, 2, 3], [2, 3, 2, 3]], dtype=dtype)

        @clpy.fuse(reduce=clpy.sum)
        def g(x, y, z):
            return x * y + z

        return g(a, b, c, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_reduce3(self, xp, dtype):
        a = xp.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=dtype)
        b = xp.array([[2, 2, 3, 3], [2, 2, 3, 3]], dtype=dtype)
        c = xp.array([[2, 3, 2, 3], [2, 3, 2, 3]], dtype=dtype)

        @clpy.fuse(reduce=clpy.sum)
        def g(x, y, z):
            return x * y + z

        return g(a, b, c, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_reduce4(self, xp, dtype):
        a = xp.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=dtype)

        @clpy.fuse(reduce=clpy.prod)
        def g(x):
            return x

        return g(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_reduce5(self, xp, dtype):
        a = xp.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=dtype)

        @clpy.fuse(reduce=clpy.max)
        def g(x):
            return x

        return g(a, axis=0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_clpy_array_equal()
    def test_reduce6(self, xp, dtype):
        a = xp.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=dtype)

        @clpy.fuse(reduce=clpy.min)
        def g(x):
            return x

        return g(a, axis=0)


@testing.gpu
class TestFusionDecorator(unittest.TestCase):
    @testing.numpy_clpy_array_equal()
    def test_without_paren(self, xp):
        @clpy.fuse
        def func_wo_paren(x):
            """Fuse without parentheses"""
            return x + x

        self.assertEqual(func_wo_paren.__name__, 'func_wo_paren')
        self.assertEqual(func_wo_paren.__doc__, 'Fuse without parentheses')

        a = xp.array([1])
        return func_wo_paren(a)

    @testing.numpy_clpy_array_equal()
    def test_with_paren(self, xp):
        @clpy.fuse()
        def func_w_paren(x):
            """Fuse with parentheses"""
            return x + x

        self.assertEqual(func_w_paren.__name__, 'func_w_paren')
        self.assertEqual(func_w_paren.__doc__, 'Fuse with parentheses')

        a = xp.array([1])
        return func_w_paren(a)


@testing.gpu
class TestFusionKernelName(unittest.TestCase):

    def check(self, xp, func, expected_name, is_elementwise):
        a = xp.array([2, 3], 'f')
        b = xp.array([2, 3], 'f')
        c = xp.array([2, 3], 'f')

        # Test kernel name (with mock)
        if xp is clpy:
            target = (
                'clpy.core.core.ElementwiseKernel' if is_elementwise
                else 'clpy.core.core.ReductionKernel')

            with mock.patch(target) as Kernel:
                func(a, b, c)
                Kernel.assert_called_once()
                self.assertEqual(Kernel.call_args[1]['name'], expected_name)

        # Test there's no error in computation (without mock)
        return func(a, b, c)

    @testing.numpy_clpy_array_equal()
    def test_elementwise(self, xp):
        def func(a, b, c):
            @clpy.fuse()
            def func_a1(x, y, z):
                return (x + y) * z

            return func_a1(a, b, c)

        return self.check(xp, func, 'func_a1', True)

    @testing.numpy_clpy_array_equal()
    def test_elementwise_with_name(self, xp):
        def func(a, b, c):
            @clpy.fuse(kernel_name='abc')
            def func_a1(x, y, z):
                return (x + y) * z

            return func_a1(a, b, c)

        return self.check(xp, func, 'abc', True)

    @testing.numpy_clpy_array_equal()
    def test_reduction(self, xp):
        def func(a, b, c):
            @clpy.fuse(reduce=clpy.sum)
            def func_a1(x, y, z):
                return (x + y) * z

            return func_a1(a, b, c)

        return self.check(xp, func, 'func_a1', False)

    @testing.numpy_clpy_array_equal()
    def test_reduction_with_name(self, xp):
        def func(a, b, c):
            @clpy.fuse(reduce=clpy.sum, kernel_name='abc')
            def func_a1(x, y, z):
                return (x + y) * z

            return func_a1(a, b, c)

        return self.check(xp, func, 'abc', False)
