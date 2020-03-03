import atexit
import binascii
import functools
import operator
import os
import time

import numpy
import six

import clpy
from clpy import backend
import clpy.backend.opencl.random as clrand
from clpy import core


_gumbel_kernel = None


def _get_gumbel_kernel():
    global _gumbel_kernel
    if _gumbel_kernel is None:
        _gumbel_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = loc - log(-log(1 - x)) * scale',
            'gumbel_kernel'
        )
    return _gumbel_kernel


class RandomState(object):

    """Portable container of a pseudo-random number generator.

    An instance of this class holds the state of a random number generator. The
    state is available only on the device which has been current at the
    initialization of the instance.

    Functions of :mod:`clpy.random` use global instances of this class.
    Different instances are used for different devices. The global state for
    the current device can be obtained by the
    :func:`clpy.random.get_random_state` function.

    Args:
        seed (None or int): Seed of the random number generator. See the
            :meth:`~clpy.random.RandomState.seed` method for detail.
        method (int): Method of the random number generator. Following values
            are available::

               clpy.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT
               clpy.cuda.curand.CURAND_RNG_XORWOW
               clpy.cuda.curand.CURAND_RNG_MRG32K3A
               clpy.cuda.curand.CURAND_RNG_MTGP32
               clpy.cuda.curand.CURAND_RNG_MT19937
               clpy.cuda.curand.CURAND_RNG_PHILOX4_32_10

    """

    def __init__(self, seed=None, method=None):
        if method is None:
            method = clrand.CLPY_RNG_PSEUDO_DEFAULT
        if method not in [
            clrand.CLPY_RNG_PSEUDO_DEFAULT,
            clrand.CLPY_RNG_XORWOW
        ]:
            raise NotImplementedError(
                "ClPy supports no other methods than XORWOW."
            )
        self._generator = clrand.createGenerator()
        self.seed(seed)

    def __del__(self):
        pass

    def set_stream(self, stream=None):
        raise NotImplementedError("ClPy doesn't support streams yet.")

    # NumPy compatible functions

    def lognormal(self, mean=0.0, sigma=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a log normal distribution.

        .. seealso::
            :func:`clpy.random.lognormal` for full documentation,
            :meth:`numpy.random.RandomState.lognormal`

        """
        # Note(nsakabe-fixstars):
        # https://github.com/numpy/numpy/blob/2a488fe76a0f732dc418d03b452caace161673da/numpy/random/src/distributions/distributions.c#L510
        # says lognormal(mean, sigma) equals to exp(random_normal(mean, sigma))

        return clpy.exp(
            self.normal(loc=mean, scale=sigma, size=size, dtype=dtype))

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of normally distributed samples.

        .. seealso::
            :func:`clpy.random.normal` for full documentation,
            :meth:`numpy.random.RandomState.normal`

        """
        dtype = _check_and_get_dtype(dtype)
        out = clpy.empty(core.get_size(size), dtype=dtype)
        if dtype.char == 'f':
            clrand.generateNormal(self._generator, out, loc, scale)
        else:
            clrand.generateNormalDouble(self._generator, out, loc, scale)
        return out

    def rand(self, *size, **kwarg):
        """Returns uniform random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`clpy.random.rand` for full documentation,
            :meth:`numpy.random.RandomState.rand`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('rand() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.random_sample(size=size, dtype=dtype)

    def randn(self, *size, **kwarg):
        """Returns an array of standard normal random values.

        .. seealso::
            :func:`clpy.random.randn` for full documentation,
            :meth:`numpy.random.RandomState.randn`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('randn() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.normal(size=size, dtype=dtype)

    _1m_kernel = core.ElementwiseKernel(
        '', 'T x', 'x = 1 - x', 'clpy_random_1_minus_x')

    def random_sample(self, size=None, dtype=float):
        """Returns an array of random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`clpy.random.random_sample` for full documentation,
            :meth:`numpy.random.RandomState.random_sample`

        """
        dtype = _check_and_get_dtype(dtype)
        out = clpy.empty(size, dtype=dtype)
        if dtype.char == 'f':
            func = clrand.generateUniform
        else:
            func = clrand.generateUniformDouble
        func(self._generator, out)
        RandomState._1m_kernel(out)
        return out

    def interval(self, mx, size):
        """Generate multiple integers independently sampled uniformly from ``[0, mx]``.

        Args:
            mx (int): Upper bound of the interval
            size (None or int or tuple): Shape of the array or the scalar
                returned.
        Returns:
            int or clpy.ndarray: If ``None``, an :class:`clpy.ndarray` with
            shape ``()`` is returned.
            If ``int``, 1-D array of length size is returned.
            If ``tuple``, multi-dimensional array with shape
            ``size`` is returned.
            Currently, only 32 bit integers can be sampled.
            If 0 :math:`\\leq` ``mx`` :math:`\\leq` 0x7fffffff,
            a ``numpy.int32`` array is returned.
            If 0x80000000 :math:`\\leq` ``mx`` :math:`\\leq` 0xffffffff,
            a ``numpy.uint32`` array is returned.
        """
        if size is None:
            return self.interval(mx, 1).reshape(())
        elif isinstance(size, int):
            size = (size, )

        if mx == 0:
            return clpy.zeros(size, dtype=numpy.int32)

        if mx < 0:
            raise ValueError(
                'mx must be non-negative (actual: {})'.format(mx))
        elif mx <= 0x7fffffff:
            dtype = numpy.int32
        elif mx <= 0xffffffff:
            dtype = numpy.uint32
        else:
            raise ValueError(
                'mx must be within uint32 range (actual: {})'.format(mx))

        mask = (1 << mx.bit_length()) - 1
        mask = clpy.array(mask, dtype=dtype)

        n = functools.reduce(operator.mul, size, 1)

        sample = clpy.empty((n,), dtype=dtype)
        n_rem = n  # The number of remaining elements to sample
        ret = None
        while n_rem > 0:
            clrand.generate(self._generator, sample)
            # Drop the samples that exceed the upper limit
            sample &= mask
            success = sample <= mx

            if ret is None:
                # If the sampling has finished in the first iteration,
                # just return the sample.
                if success.all():
                    n_rem = 0
                    ret = sample
                    break

                # Allocate the return array.
                ret = clpy.empty((n,), dtype=dtype)

            n_succ = min(n_rem, int(success.sum()))
            ret[n - n_rem:n - n_rem + n_succ] = sample[success][:n_succ]
            n_rem -= n_succ

        assert n_rem == 0
        return ret.reshape(size)

    def seed(self, seed=None):
        """Resets the state of the random number generator with a seed.

        .. seealso::
            :func:`clpy.random.seed` for full documentation,
            :meth:`numpy.random.RandomState.seed`

        """
        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.asarray(seed).astype(numpy.uint64, casting='safe')

        clrand.setPseudoRandomGeneratorSeed(self._generator, seed)
        # curand.setGeneratorOffset(self._generator, 0)

    def standard_normal(self, size=None, dtype=float):
        """Returns samples drawn from the standard normal distribution.

        .. seealso::
            :func:`clpy.random.standard_normal` for full documentation,
            :meth:`numpy.random.RandomState.standard_normal`

        """
        return self.normal(size=size, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float):
        """Returns an array of uniformly-distributed samples over an interval.

        .. seealso::
            :func:`clpy.random.uniform` for full documentation,
            :meth:`numpy.random.RandomState.uniform`

        """
        dtype = clpy.dtype(dtype)
        rand = self.random_sample(size=size, dtype=dtype)
        return dtype.type(low) + rand * dtype.type(high - low)

    def choice(self, a, size=None, replace=True, p=None):
        """Returns an array of random values from a given 1-D array.

        .. seealso::
            :func:`clpy.random.choice` for full document,
            :meth:`numpy.random.choice`

        """
        if a is None:
            raise ValueError('a must be 1-dimensional or an integer')
        if isinstance(a, clpy.ndarray) and a.ndim == 0:
            raise NotImplementedError
        if isinstance(a, six.integer_types):
            a_size = a
            if a_size <= 0:
                raise ValueError('a must be greater than 0')
        else:
            a = clpy.array(a, copy=False)
            if a.ndim != 1:
                raise ValueError('a must be 1-dimensional or an integer')
            else:
                a_size = len(a)
                if a_size == 0:
                    raise ValueError('a must be non-empty')

        if p is not None:
            p = clpy.array(p)
            if p.ndim != 1:
                raise ValueError('p must be 1-dimensional')
            if len(p) != a_size:
                raise ValueError('a and p must have same size')
            if not (p >= 0).all():
                raise ValueError('probabilities are not non-negative')
            p_sum = clpy.sum(p).get()
            if not numpy.allclose(p_sum, 1):
                raise ValueError('probabilities do not sum to 1')

        if size is None:
            raise NotImplementedError
        shape = size
        size = numpy.prod(shape)

        if not replace and p is None:
            if a_size < size:
                raise ValueError(
                    'Cannot take a larger sample than population when '
                    '\'replace=False\'')
            if isinstance(a, six.integer_types):
                indices = clpy.arange(a, dtype='l')
            else:
                indices = a.copy()
            self.shuffle(indices)
            return indices[:size].reshape(shape)

        if not replace:
            raise NotImplementedError

        if p is not None:
            p = clpy.broadcast_to(p, (size, a_size))
            index = clpy.argmax(clpy.log(p) +
                                self.gumbel(size=(size, a_size)),
                                axis=1)
            if not isinstance(shape, six.integer_types):
                index = clpy.reshape(index, shape)
        else:
            index = self.randint(0, a_size, size=shape)
            # Align the dtype with NumPy
            index = index.astype(clpy.int64, copy=False)

        if isinstance(a, six.integer_types):
            return index

        if index.ndim == 0:
            return clpy.array(a[index], dtype=a.dtype)

        return a[index]

    def shuffle(self, a):
        """Returns a shuffled array.

        .. seealso::
            :func:`clpy.random.shuffle` for full document,
            :meth:`numpy.random.shuffle`

        """
        if not isinstance(a, clpy.ndarray):
            raise TypeError('The array must be clpy.ndarray')

        if a.ndim == 0:
            raise TypeError('An array whose ndim is 0 is not supported')

        sample = clpy.zeros((len(a)), dtype=numpy.int32)
        clrand.generate(self._generator, sample)
        a[:] = a[clpy.argsort(sample)]

    def gumbel(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a Gumbel distribution.

        .. seealso::
            :func:`clpy.random.gumbel` for full documentation,
            :meth:`numpy.random.RandomState.gumbel`
        """
        x = self.uniform(size=size, dtype=dtype)
        # We use `1 - x` as input of `log` method to prevent overflow.
        # It obeys numpy implementation.
        _get_gumbel_kernel()(x, loc, scale, x)
        return x

    def randint(self, low, high=None, size=None, dtype='l'):
        """Returns a scalar or an array of integer values over ``[low, high)``.

        .. seealso::
            :func:`clpy.random.randint` for full documentation,
            :meth:`numpy.random.RandomState.randint`
        """
        if high is None:
            lo = 0
            hi = low
        else:
            lo = low
            hi = high

        if lo >= hi:
            raise ValueError('low >= high')
        if lo < clpy.iinfo(dtype).min:
            raise ValueError(
                'low is out of bounds for {}'.format(clpy.dtype(dtype).name))
        if hi > clpy.iinfo(dtype).max + 1:
            raise ValueError(
                'high is out of bounds for {}'.format(clpy.dtype(dtype).name))

        diff = hi - lo - 1
        if diff > clpy.iinfo(clpy.int32).max - clpy.iinfo(clpy.int32).min + 1:
            raise NotImplementedError(
                'Sampling from a range whose extent is larger than int32 '
                'range is currently not supported')
        x = self.interval(diff, size).astype(dtype, copy=False)
        clpy.add(x, lo, out=x)
        return x


def seed(seed=None):
    """Resets the state of the random number generator with a seed.

    This function resets the state of the global random number generator for
    the current device. Be careful that generators for other devices are not
    affected.

    Args:
        seed (None or int): Seed for the random number generator. If ``None``,
            it uses :func:`os.urandom` if available or :func:`time.clock`
            otherwise. Note that this function does not support seeding by an
            integer array.

    """
    get_random_state().seed(seed)


# ClPy specific functions

_random_states = {}


@atexit.register
def reset_states():
    global _random_states
    _random_states = {}


def get_random_state():
    """Gets the state of the random number generator for the current device.

    If the state for the current device is not created yet, this function
    creates a new one, initializes it, and stores it as the state for the
    current device.

    Returns:
        RandomState: The state of the random number generator for the
        device.

    """
    dev = backend.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        seed = os.getenv('CLPY_SEED')
        if seed is None:
            seed = os.getenv('CHAINER_SEED')
        if seed is not None:
            seed = numpy.uint64(int(seed))
        rs = RandomState(seed)
        rs = _random_states.setdefault(dev.id, rs)
    return rs


def _check_and_get_dtype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype.char not in ('f', 'd'):
        raise TypeError('clpy.random only supports float32 and float64')
    return dtype
