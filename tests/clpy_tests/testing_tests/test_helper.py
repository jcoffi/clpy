import re
import unittest

import numpy
import six

import clpy
from clpy import testing
from clpy.testing import helper


class TestContainsSignedAndUnsigned(unittest.TestCase):

    def test_include(self):
        kw = {'x': numpy.int32, 'y': numpy.uint32}
        self.assertTrue(helper._contains_signed_and_unsigned(kw))

        kw = {'x': numpy.float32, 'y': numpy.uint32}
        self.assertTrue(helper._contains_signed_and_unsigned(kw))

    def test_signed_only(self):
        kw = {'x': numpy.int32}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))

        kw = {'x': numpy.float}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))

    def test_unsigned_only(self):
        kw = {'x': numpy.uint32}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))


class TestCheckCupyNumpyError(unittest.TestCase):

    tbs = {
        clpy: 'xxxx',
        numpy: 'yyyy'
    }

    def test_both_success(self):
        @testing.helper.numpy_clpy_raises()
        def dummy_both_success(self, xp):
            pass

        with self.assertRaises(AssertionError):
            dummy_both_success(self)

    def test_clpy_error(self):
        @testing.helper.numpy_clpy_raises()
        def dummy_clpy_error(self, xp):
            if xp is clpy:
                raise Exception(self.tbs.get(clpy))

        with six.assertRaisesRegex(self, AssertionError, self.tbs.get(clpy)):
            dummy_clpy_error(self)

    def test_numpy_error(self):
        @testing.helper.numpy_clpy_raises()
        def dummy_numpy_error(self, xp):
            if xp is numpy:
                raise Exception(self.tbs.get(numpy))

        with six.assertRaisesRegex(self, AssertionError, self.tbs.get(numpy)):
            dummy_numpy_error(self)

    def test_clpy_numpy_different_error(self):
        @testing.helper.numpy_clpy_raises()
        def dummy_clpy_numpy_different_error(self, xp):
            if xp is clpy:
                raise TypeError(self.tbs.get(clpy))
            elif xp is numpy:
                raise ValueError(self.tbs.get(numpy))

        # Use re.S mode to ignore new line characters
        pattern = re.compile(
            self.tbs.get(clpy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_clpy_numpy_different_error(self)

    def test_clpy_derived_error(self):
        @testing.helper.numpy_clpy_raises()
        def dummy_clpy_derived_error(self, xp):
            if xp is clpy:
                raise ValueError(self.tbs.get(clpy))
            elif xp is numpy:
                raise Exception(self.tbs.get(numpy))

        dummy_clpy_derived_error(self)  # Assert no exceptions

    def test_numpy_derived_error(self):
        @testing.helper.numpy_clpy_raises()
        def dummy_numpy_derived_error(self, xp):
            if xp is clpy:
                raise Exception(self.tbs.get(clpy))
            elif xp is numpy:
                raise IndexError(self.tbs.get(numpy))

        # NumPy errors may not derive from CuPy errors, i.e. CuPy errors should
        # be at least as explicit as the NumPy error
        pattern = re.compile(
            self.tbs.get(clpy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_numpy_derived_error(self)

    def test_same_error(self):
        @testing.helper.numpy_clpy_raises(accept_error=Exception)
        def dummy_same_error(self, xp):
            raise Exception(self.tbs.get(xp))

        dummy_same_error(self)

    def test_clpy_derived_unaccept_error(self):
        @testing.helper.numpy_clpy_raises(accept_error=ValueError)
        def dummy_clpy_derived_unaccept_error(self, xp):
            if xp is clpy:
                raise IndexError(self.tbs.get(clpy))
            elif xp is numpy:
                raise Exception(self.tbs.get(numpy))

        # Neither `IndexError` nor `Exception` is derived from `ValueError`,
        # therefore expect an error
        pattern = re.compile(
            self.tbs.get(clpy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_clpy_derived_unaccept_error(self)

    def test_numpy_derived_unaccept_error(self):
        @testing.helper.numpy_clpy_raises(accept_error=ValueError)
        def dummy_numpy_derived_unaccept_error(self, xp):
            if xp is clpy:
                raise Exception(self.tbs.get(clpy))
            elif xp is numpy:
                raise ValueError(self.tbs.get(numpy))

        # `Exception` is not derived from `ValueError`, therefore expect an
        # error
        pattern = re.compile(
            self.tbs.get(clpy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_numpy_derived_unaccept_error(self)

    def test_forbidden_error(self):
        @testing.helper.numpy_clpy_raises(accept_error=False)
        def dummy_forbidden_error(self, xp):
            raise Exception(self.tbs.get(xp))

        pattern = re.compile(
            self.tbs.get(clpy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_forbidden_error(self)


class NumPyCuPyDecoratorBase(object):

    def test_valid(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(type(self).valid_func)
        decorated_func(self)

    def test_invalid(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(type(self).invalid_func)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_name(self):
        decorator = getattr(testing, self.decorator)(name='foo')
        decorated_func = decorator(type(self).strange_kw_func)
        decorated_func(self)


def numpy_error(_, xp):
    if xp == numpy:
        raise ValueError()
    elif xp == clpy:
        return clpy.array(1)


def clpy_error(_, xp):
    if xp == numpy:
        return numpy.array(1)
    elif xp == clpy:
        raise ValueError()


@testing.gpu
class NumPyCuPyDecoratorBase2(object):

    def test_accept_error_numpy(self):
        decorator = getattr(testing, self.decorator)(accept_error=False)
        decorated_func = decorator(numpy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_accept_error_clpy(self):
        decorator = getattr(testing, self.decorator)(accept_error=False)
        decorated_func = decorator(clpy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)


def make_result(xp, np_result, cp_result):
    if xp == numpy:
        return np_result
    elif xp == clpy:
        return cp_result


@testing.parameterize(
    {'decorator': 'numpy_clpy_allclose'},
    {'decorator': 'numpy_clpy_array_almost_equal'},
    {'decorator': 'numpy_clpy_array_almost_equal_nulp'},
    {'decorator': 'numpy_clpy_array_max_ulp'},
    {'decorator': 'numpy_clpy_array_equal'}
)
class TestNumPyCuPyEqual(unittest.TestCase, NumPyCuPyDecoratorBase,
                         NumPyCuPyDecoratorBase2):

    def valid_func(self, xp):
        return make_result(xp, numpy.array(1), clpy.array(1))

    def invalid_func(self, xp):
        return make_result(xp, numpy.array(1), clpy.array(2))

    def strange_kw_func(self, foo):
        return make_result(foo, numpy.array(1), clpy.array(1))


@testing.parameterize(
    {'decorator': 'numpy_clpy_array_list_equal'}
)
@testing.gpu
class TestNumPyCuPyListEqual(unittest.TestCase, NumPyCuPyDecoratorBase):

    def valid_func(self, xp):
        return make_result(xp, [numpy.array(1)], [clpy.array(1)])

    def invalid_func(self, xp):
        return make_result(xp, [numpy.array(1)], [clpy.array(2)])

    def strange_kw_func(self, foo):
        return make_result(foo, [numpy.array(1)], [clpy.array(1)])


@testing.parameterize(
    {'decorator': 'numpy_clpy_array_less'}
)
class TestNumPyCuPyLess(unittest.TestCase, NumPyCuPyDecoratorBase,
                        NumPyCuPyDecoratorBase2):

    def valid_func(self, xp):
        return make_result(xp, numpy.array(2), clpy.array(1))

    def invalid_func(self, xp):
        return make_result(xp, numpy.array(1), clpy.array(2))

    def strange_kw_func(self, foo):
        return make_result(foo, numpy.array(2), clpy.array(1))


@testing.parameterize(
    {'decorator': 'numpy_clpy_raises'}
)
class TestNumPyCuPyRaise(unittest.TestCase, NumPyCuPyDecoratorBase):

    def valid_func(self, xp):
        raise ValueError()

    def invalid_func(self, xp):
        return make_result(xp, numpy.array(1), clpy.array(1))

    def strange_kw_func(self, foo):
        raise ValueError()

    def test_accept_error_numpy(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(numpy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_accept_error_clpy(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(clpy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)


class TestIgnoreOfNegativeValueDifferenceOnCpuAndGpu(unittest.TestCase):

    @helper.for_unsigned_dtypes('dtype1')
    @helper.for_signed_dtypes('dtype2')
    @helper.numpy_clpy_allclose()
    def correct_failure(self, xp, dtype1, dtype2):
        if xp == numpy:
            return xp.array(-1, dtype=numpy.float32)
        else:
            return xp.array(-2, dtype=numpy.float32)

    def test_correct_failure(self):
        numpy.testing.assert_raises_regex(
            AssertionError, 'mismatch 100.0%', self.correct_failure)

    @helper.for_unsigned_dtypes('dtype1')
    @helper.for_signed_dtypes('dtype2')
    @helper.numpy_clpy_allclose()
    def test_correct_success(self, xp, dtype1, dtype2):
        # Behavior of assigning a negative value to an unsigned integer
        # variable is undefined.
        # nVidia GPUs and Intel CPUs behave differently.
        # To avoid this difference, we need to ignore dimensions whose
        # values are negative.
        if xp == numpy:
            return xp.array(-1, dtype=dtype1)
        else:
            return xp.array(-2, dtype=dtype1)
