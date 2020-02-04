from __future__ import division
import os
import sys

import numpy
import six

from clpy import _version

from importlib.abc import MetaPathFinder
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.machinery import SourceFileLoader


try:
    from clpy import core  # NOQA
except ImportError:
    # core is a c-extension module.
    # When a user cannot import core, it represents that CuPy is not correctly
    # built.
    exc_info = sys.exc_info()
    msg = ('ClPy is not correctly installed. Please check your environment, '
           'uninstall ClPy and reinstall it with `pip install clpy '
           '--no-cache-dir -vvvv`.\n\n'
           'original error: {}'.format(exc_info[1]))

    six.reraise(ImportError, ImportError(msg), exc_info[2])

from clpy import backend


class CupyAliasMetaPathFinder(MetaPathFinder):
    def find_spec(fullname, path, target=None):
        split_name = fullname.split('.', maxsplit=1)
        if split_name[0] == 'cupy':
            alias_name = 'cupy_alias'
            if len(split_name) == 2:
                alias_name += '.' + split_name[1]
                import_module(fullname.rsplit('.', maxsplit=1)[0])
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                alias_name.replace('.', os.sep))
            if os.path.isdir(path):
                path = os.path.join(path, '__init__.py')
                return ModuleSpec(
                    name=fullname,
                    loader=SourceFileLoader(fullname, path),
                    origin=path,
                    is_package=True
                )
            else:
                path = path+'.py'
                return ModuleSpec(
                    name=fullname,
                    loader=SourceFileLoader(fullname, path),
                    origin=path
                )


if os.getenv('CLPY_NOT_HOOK_CUPY') != '1':
    CupyAliasMetaPathFinder_hooked = False
    for meta_path_item in sys.meta_path:
        if hasattr(meta_path_item, '__name__'):
            if meta_path_item.__name__ == 'CupyAliasMetaPathFinder':
                CupyAliasMetaPathFinder_hooked = True
    if CupyAliasMetaPathFinder_hooked is False:
        sys.meta_path.insert(0, CupyAliasMetaPathFinder)


def is_available():
    return backend.is_available()


__version__ = _version.__version__


from clpy import binary  # NOQA
from clpy.core import fusion  # NOQA
from clpy import creation  # NOQA
from clpy import indexing  # NOQA
from clpy import io  # NOQA
from clpy import linalg  # NOQA
from clpy import manipulation  # NOQA
from clpy import padding  # NOQA
from clpy import random  # NOQA
from clpy import sorting  # NOQA
from clpy import sparse  # NOQA
from clpy import statistics  # NOQA
from clpy import testing  # NOQA  # NOQA
from clpy import util  # NOQA


# import class and function
from clpy.core import ndarray  # NOQA

# dtype short cuts
from numpy import floating  # NOQA
from numpy import inexact  # NOQA
from numpy import integer  # NOQA
from numpy import number  # NOQA
from numpy import signedinteger  # NOQA
from numpy import unsignedinteger  # NOQA


from numpy import bool_  # NOQA

from numpy import byte  # NOQA

from numpy import short  # NOQA

from numpy import intc  # NOQA

from numpy import int_  # NOQA

from numpy import longlong  # NOQA

from numpy import ubyte  # NOQA

from numpy import ushort  # NOQA

from numpy import uintc  # NOQA

from numpy import uint  # NOQA

from numpy import ulonglong  # NOQA


# from numpy import half  # NOQA

from numpy import single  # NOQA

from numpy import float_  # NOQA

from numpy import longfloat  # NOQA


from numpy import int8  # NOQA

from numpy import int16  # NOQA

from numpy import int32  # NOQA

from numpy import int64  # NOQA

from numpy import uint8  # NOQA

from numpy import uint16  # NOQA

from numpy import uint32  # NOQA

from numpy import uint64  # NOQA


from numpy import float16  # NOQA

from numpy import float32  # NOQA

from numpy import float64  # NOQA


from clpy.core import ufunc  # NOQA

from numpy import newaxis  # == None  # NOQA


# =============================================================================
# Routines
#
# The order of these declarations are borrowed from the NumPy document:
# http://docs.scipy.org/doc/numpy/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
from clpy.creation.basic import empty  # NOQA
from clpy.creation.basic import empty_like  # NOQA
from clpy.creation.basic import eye  # NOQA
from clpy.creation.basic import full  # NOQA
from clpy.creation.basic import full_like  # NOQA
from clpy.creation.basic import identity  # NOQA
from clpy.creation.basic import ones  # NOQA
from clpy.creation.basic import ones_like  # NOQA
from clpy.creation.basic import zeros  # NOQA
from clpy.creation.basic import zeros_like  # NOQA

from clpy.core.fusion import copy  # NOQA
from clpy.creation.from_data import array  # NOQA
from clpy.creation.from_data import asanyarray  # NOQA
from clpy.creation.from_data import asarray  # NOQA
from clpy.creation.from_data import ascontiguousarray  # NOQA

from clpy.creation.ranges import arange  # NOQA
from clpy.creation.ranges import linspace  # NOQA
from clpy.creation.ranges import logspace  # NOQA
from clpy.creation.ranges import meshgrid  # NOQA
from clpy.creation.ranges import mgrid  # NOQA
from clpy.creation.ranges import ogrid  # NOQA

from clpy.creation.matrix import diag  # NOQA
from clpy.creation.matrix import diagflat  # NOQA

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
from clpy.manipulation.basic import copyto  # NOQA

from clpy.manipulation.shape import ravel  # NOQA
from clpy.manipulation.shape import reshape  # NOQA

from clpy.manipulation.transpose import rollaxis  # NOQA
from clpy.manipulation.transpose import swapaxes  # NOQA
from clpy.manipulation.transpose import transpose  # NOQA

from clpy.manipulation.dims import atleast_1d  # NOQA
from clpy.manipulation.dims import atleast_2d  # NOQA
from clpy.manipulation.dims import atleast_3d  # NOQA
from clpy.manipulation.dims import broadcast  # NOQA
from clpy.manipulation.dims import broadcast_arrays  # NOQA
from clpy.manipulation.dims import broadcast_to  # NOQA
from clpy.manipulation.dims import expand_dims  # NOQA
from clpy.manipulation.dims import squeeze  # NOQA

from clpy.manipulation.join import column_stack  # NOQA
from clpy.manipulation.join import concatenate  # NOQA
from clpy.manipulation.join import dstack  # NOQA
from clpy.manipulation.join import hstack  # NOQA
from clpy.manipulation.join import stack  # NOQA
from clpy.manipulation.join import vstack  # NOQA

from clpy.manipulation.kind import asfortranarray  # NOQA

from clpy.manipulation.split import array_split  # NOQA
from clpy.manipulation.split import dsplit  # NOQA
from clpy.manipulation.split import hsplit  # NOQA
from clpy.manipulation.split import split  # NOQA
from clpy.manipulation.split import vsplit  # NOQA

from clpy.manipulation.tiling import repeat  # NOQA
from clpy.manipulation.tiling import tile  # NOQA

from clpy.manipulation.rearrange import flip  # NOQA
from clpy.manipulation.rearrange import fliplr  # NOQA
from clpy.manipulation.rearrange import flipud  # NOQA
from clpy.manipulation.rearrange import roll  # NOQA
from clpy.manipulation.rearrange import rot90  # NOQA

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from clpy.core.fusion import bitwise_and  # NOQA
from clpy.core.fusion import bitwise_or  # NOQA
from clpy.core.fusion import bitwise_xor  # NOQA
from clpy.core.fusion import invert  # NOQA
from clpy.core.fusion import left_shift  # NOQA
from clpy.core.fusion import right_shift  # NOQA

from clpy.binary.packing import packbits  # NOQA
from clpy.binary.packing import unpackbits  # NOQA


def binary_repr(num, width=None):
    """Return the binary representation of the input number as a string.

    .. seealso:: :func:`numpy.binary_repr`
    """
    return numpy.binary_repr(num, width)


# -----------------------------------------------------------------------------
# Data type routines (borrowed from NumPy)
# -----------------------------------------------------------------------------
from numpy import can_cast  # NOQA
from numpy import common_type  # NOQA
from numpy import min_scalar_type  # NOQA
from numpy import obj2sctype  # NOQA
from numpy import promote_types  # NOQA
from numpy import result_type  # NOQA

from numpy import dtype  # NOQA
from numpy import format_parser  # NOQA

from numpy import finfo  # NOQA
from numpy import iinfo  # NOQA
from numpy import MachAr  # NOQA

from numpy import find_common_type  # NOQA
from numpy import issctype  # NOQA
from numpy import issubclass_  # NOQA
from numpy import issubdtype  # NOQA
from numpy import issubsctype  # NOQA

from numpy import mintypecode  # NOQA
from numpy import sctype2char  # NOQA
from numpy import typename  # NOQA

# -----------------------------------------------------------------------------
# Optionally Scipy-accelerated routines
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Discrete Fourier Transform
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Indexing routines
# -----------------------------------------------------------------------------
from clpy.indexing.generate import c_  # NOQA
from clpy.indexing.generate import indices  # NOQA
from clpy.indexing.generate import ix_  # NOQA
from clpy.indexing.generate import r_  # NOQA

# from clpy.indexing.indexing import choose  # NOQA
from clpy.indexing.indexing import diagonal  # NOQA
from clpy.indexing.indexing import take  # NOQA

from clpy.indexing.insert import fill_diagonal  # NOQA
# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
from clpy.io.npz import load  # NOQA
from clpy.io.npz import save  # NOQA
from clpy.io.npz import savez  # NOQA
from clpy.io.npz import savez_compressed  # NOQA

from clpy.io.formatting import array_repr  # NOQA
from clpy.io.formatting import array_str  # NOQA


def base_repr(number, base=2, padding=0):  # NOQA (needed to avoid redefinition of `number`)
    """Return a string representation of a number in the given base system.

    .. seealso:: :func:`numpy.base_repr`
    """
    return numpy.base_repr(number, base, padding)


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
# from clpy.linalg.einsum import einsum  # NOQA

from clpy.linalg.product import dot  # NOQA
from clpy.linalg.product import inner  # NOQA
from clpy.linalg.product import kron  # NOQA
from clpy.linalg.product import matmul  # NOQA
from clpy.linalg.product import outer  # NOQA
from clpy.linalg.product import tensordot  # NOQA
from clpy.linalg.product import vdot  # NOQA

# from clpy.linalg.norms import trace  # NOQA

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
from clpy.core.fusion import isfinite  # NOQA
from clpy.core.fusion import isinf  # NOQA
from clpy.core.fusion import isnan  # NOQA


def isscalar(num):
    """Returns True if the type of num is a scalar type.

    .. seealso:: :func:`numpy.isscalar`
    """
    return numpy.isscalar(num)


from clpy.core.fusion import logical_and  # NOQA
from clpy.core.fusion import logical_not  # NOQA
from clpy.core.fusion import logical_or  # NOQA
from clpy.core.fusion import logical_xor  # NOQA

from clpy.core.fusion import equal  # NOQA
from clpy.core.fusion import greater  # NOQA
from clpy.core.fusion import greater_equal  # NOQA
from clpy.core.fusion import less  # NOQA
from clpy.core.fusion import less_equal  # NOQA
from clpy.core.fusion import not_equal  # NOQA

from clpy.core.fusion import all  # NOQA
from clpy.core.fusion import any  # NOQA

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
from clpy.core.fusion import arccos  # NOQA
from clpy.core.fusion import arcsin  # NOQA
from clpy.core.fusion import arctan  # NOQA
from clpy.core.fusion import arctan2  # NOQA
from clpy.core.fusion import cos  # NOQA
from clpy.core.fusion import deg2rad  # NOQA
from clpy.core.fusion import degrees  # NOQA
from clpy.core.fusion import hypot  # NOQA
from clpy.core.fusion import rad2deg  # NOQA
from clpy.core.fusion import radians  # NOQA
from clpy.core.fusion import sin  # NOQA
from clpy.core.fusion import tan  # NOQA

from clpy.core.fusion import arccosh  # NOQA
from clpy.core.fusion import arcsinh  # NOQA
from clpy.core.fusion import arctanh  # NOQA
from clpy.core.fusion import cosh  # NOQA
from clpy.core.fusion import sinh  # NOQA
from clpy.core.fusion import tanh  # NOQA

from clpy.core.fusion import ceil  # NOQA
from clpy.core.fusion import fix  # NOQA
from clpy.core.fusion import floor  # NOQA
from clpy.core.fusion import rint  # NOQA
from clpy.core.fusion import trunc  # NOQA

from clpy.core.fusion import prod  # NOQA
from clpy.core.fusion import sum  # NOQA
from clpy.math.sumprod import cumprod  # NOQA
from clpy.math.sumprod import cumsum  # NOQA
from clpy.math.window import blackman  # NOQA
from clpy.math.window import hamming  # NOQA
from clpy.math.window import hanning  # NOQA


from clpy.core.fusion import exp  # NOQA
from clpy.core.fusion import exp2  # NOQA
from clpy.core.fusion import expm1  # NOQA
from clpy.core.fusion import log  # NOQA
from clpy.core.fusion import log10  # NOQA
from clpy.core.fusion import log1p  # NOQA
from clpy.core.fusion import log2  # NOQA
from clpy.core.fusion import logaddexp  # NOQA
from clpy.core.fusion import logaddexp2  # NOQA

from clpy.core.fusion import copysign  # NOQA
from clpy.core.fusion import frexp  # NOQA
from clpy.core.fusion import ldexp  # NOQA
from clpy.core.fusion import nextafter  # NOQA
from clpy.core.fusion import signbit  # NOQA

from clpy.core.fusion import add  # NOQA
from clpy.core.fusion import divide  # NOQA
from clpy.core.fusion import floor_divide  # NOQA
from clpy.core.fusion import fmod  # NOQA
from clpy.core.fusion import modf  # NOQA
from clpy.core.fusion import multiply  # NOQA
from clpy.core.fusion import negative  # NOQA
from clpy.core.fusion import power  # NOQA
from clpy.core.fusion import reciprocal  # NOQA
from clpy.core.fusion import remainder  # NOQA
from clpy.core.fusion import remainder as mod  # NOQA
from clpy.core.fusion import subtract  # NOQA
from clpy.core.fusion import true_divide  # NOQA

# TODO(okuta): implement fusion function
from clpy.core import angle  # NOQA
from clpy.core import conj  # NOQA
from clpy.core import imag  # NOQA
from clpy.core import real  # NOQA

from clpy.core.fusion import abs  # NOQA
from clpy.core.fusion import absolute  # NOQA
from clpy.core.fusion import clip  # NOQA
from clpy.core.fusion import fmax  # NOQA
from clpy.core.fusion import fmin  # NOQA
from clpy.core.fusion import maximum  # NOQA
from clpy.core.fusion import minimum  # NOQA
from clpy.core.fusion import sign  # NOQA
from clpy.core.fusion import sqrt  # NOQA
from clpy.core.fusion import square  # NOQA

# -----------------------------------------------------------------------------
# Padding
# -----------------------------------------------------------------------------
pad = padding.pad.pad


# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
from clpy.sorting.count import count_nonzero  # NOQA
from clpy.sorting.search import flatnonzero  # NOQA
from clpy.sorting.search import nonzero  # NOQA

from clpy.core.fusion import where  # NOQA
from clpy.sorting.search import argmax  # NOQA
from clpy.sorting.search import argmin  # NOQA

# from clpy.sorting.sort import argpartition  # NOQA
from clpy.sorting.sort import argsort  # NOQA
# from clpy.sorting.sort import lexsort  # NOQA
from clpy.sorting.sort import msort  # NOQA
# from clpy.sorting.sort import partition  # NOQA
from clpy.sorting.sort import sort  # NOQA

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
from clpy.core.fusion import amax  # NOQA
from clpy.core.fusion import amax as max  # NOQA
from clpy.core.fusion import amin  # NOQA
from clpy.core.fusion import amin as min  # NOQA
from clpy.statistics.order import nanmax  # NOQA
from clpy.statistics.order import nanmin  # NOQA

from clpy.statistics.meanvar import mean  # NOQA
from clpy.statistics.meanvar import std  # NOQA
from clpy.statistics.meanvar import var  # NOQA

from clpy.statistics.histogram import bincount  # NOQA

# -----------------------------------------------------------------------------
# CuPy specific functions
# -----------------------------------------------------------------------------

from clpy.util import clear_memo  # NOQA
from clpy.util import memoize  # NOQA

from clpy.core import ElementwiseKernel  # NOQA
from clpy.core import ReductionKernel  # NOQA

from clpy.ext.scatter import scatter_add  # NOQA


def asnumpy(a, stream=None):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (clpy.cuda.Stream): CUDA stream object. If it is specified, then
            the device-to-host copy runs asynchronously. Otherwise, the copy is
            synchronous. Note that if ``a`` is not a :class:`clpy.ndarray`
            object, then this argument has no effect.

    Returns:
        numpy.ndarray: Converted array on the host memory.

    """
    if isinstance(a, ndarray):
        return a.get(stream=stream)
    else:
        return numpy.asarray(a)


_clpy = sys.modules[__name__]


def get_array_module(*args):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If at least one of
    the arguments is a :class:`clpy.ndarray` object, the :mod:`clpy` module is
    returned.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`clpy` or :mod:`numpy` is returned based on the types of
        the arguments.

    .. admonition:: Example

       A NumPy/CuPy generic function can be written as follows

       >>> def softplus(x):
       ...     xp = clpy.get_array_module(x)
       ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

    """
    for arg in args:
        if isinstance(arg, (ndarray, sparse.spmatrix)):
            return _clpy
    return numpy


fuse = fusion.fuse

disable_experimental_feature_warning = False
