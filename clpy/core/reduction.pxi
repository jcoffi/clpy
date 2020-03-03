import collections
import string

import numpy

# from clpy.backend import compiler
from clpy import util

cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.utility


cpdef _get_simple_reduction_kernel(
        name, local_size, reduce_type, params, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, output_store, preamble,
        options, clpy_variables_declaration):
    if identity is None:
        identity = '0'

    module_code = string.Template(string.Template('''
    ${type_preamble}
    ${preamble}
    #define REDUCE(a, b) (${reduce_expr})
    #define POST_MAP(a) (${post_map_expr})
    #define _REDUCE(_offset) if (lid < _offset) { \
      _type_reduce _a = _sdata[lid], _b = _sdata[(lid + _offset)]; \
      _sdata[lid] = REDUCE(_a, _b); \
    }

    typedef ${reduce_type} _type_reduce;
    __kernel void ${name}(${params}) {
      const size_t lid = get_local_id(0);
      ${clpy_variables_declaration}
      __attribute__((annotate("clpy_reduction_tag"))) \
        void __clpy_reduction_preprocess();

      const size_t _J_offset = lid / _local_stride;
      const size_t  _j_offset = _J_offset * _out_ind.size();
      const size_t  _J_stride = ${local_size};
      const size_t  _j_stride = ${local_size} * _out_ind.size();

      for (size_t _i_base = get_group_id(0) * _local_stride;
           _i_base < _out_ind.size();
           _i_base += get_num_groups(0) * _local_stride) {
        _type_reduce _s = ${identity};
        const size_t  _i = _i_base + lid % _local_stride;
        size_t  _J = _J_offset;
        for (size_t _j = _i + _j_offset; _j < _in_ind.size();
             _j += _j_stride, _J += _J_stride) {
          __attribute__((annotate("clpy_reduction_tag"))) \
            void __clpy_reduction_set_cindex_in();
          ${input_expr}
          _type_reduce _a = ${pre_map_expr};
          _s = REDUCE(_s, _a);
        }
        if (_local_stride < ${local_size}) {
          _sdata[lid] = _s;
          barrier(CLK_LOCAL_MEM_FENCE);
          for (size_t offset = (${local_size} >> 1);
               offset >= _local_stride;
               offset >>= 1) {
            _REDUCE(offset);
            barrier(CLK_LOCAL_MEM_FENCE);
          }
          _s = _sdata[lid];
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (_J_offset == 0 && _i < _out_ind.size()) {
          __attribute__((annotate("clpy_reduction_tag"))) \
            void __clpy_reduction_set_cindex_out();
          ${output_expr}
          POST_MAP(_s);
          ${output_store}
        }
      }
    }''').safe_substitute(identity=identity)).substitute(
        name=name,
        local_size=local_size,
        reduce_type=reduce_type,
        params=params,
        reduce_expr=reduce_expr,
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,
        type_preamble=type_preamble,
        input_expr=input_expr,
        output_expr=output_expr,
        output_store=output_store,
        preamble=preamble,
        clpy_variables_declaration=clpy_variables_declaration)
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


cpdef tuple _get_axis(object axis, Py_ssize_t ndim):
    cdef Py_ssize_t dim
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, collections.Sequence):
        axis = tuple(axis)
    else:
        axis = axis,

    for dim in axis:
        if dim < -ndim or dim >= ndim:
            raise ValueError('Axis overrun')
    axis = tuple(sorted([dim % ndim for dim in axis]))
    raxis = tuple([dim for dim in range(ndim) if dim not in axis])
    return axis, raxis


cpdef tuple _get_out_shape(
        tuple shape, tuple axis, tuple raxis, bint keepdims):
    if keepdims:
        out_shape = list(shape)
        for i in axis:
            out_shape[i] = 1
        return tuple(out_shape)
    return tuple([shape[i] for i in raxis])


cpdef tuple _get_trans_args(list args, tuple trans, tuple shape, tuple params):
    cdef ParameterInfo p
    if trans == tuple(range(len(shape))):
        return args, shape
    if params is not None:
        for p in params:
            if p.raw:
                raise NotImplementedError('Illegal conditions')
    args = [a.transpose(trans) if isinstance(a, ndarray) else a
            for a in args]
    shape = tuple([shape[i] for i in trans])
    return args, shape


cpdef list _get_inout_args(
        list in_args, list out_args, Indexer in_indexer, Indexer out_indexer,
        object out_clp2_size, tuple params, bint reduce_dims):
    if reduce_dims:
        in_args, in_shape = _reduce_dims(
            in_args, params, in_indexer.shape)
        out_args, out_shape = _reduce_dims(
            out_args, params[len(in_args):], out_indexer.shape)
        in_indexer.shape = in_shape
        out_indexer.shape = out_shape
    args = in_args + out_args + [in_indexer, out_indexer,
                                 Size_t(out_clp2_size), LocalMem()]
    return args


@util.memoize(for_each_device=True)
def _get_simple_reduction_function(
        routine, params, args_info, in_arg_dtype, out_arg_dtype, out_types,
        name, local_size, identity, input_expr, output_expr, output_store,
        _preamble, options, clpy_variables_declaration=''):
    reduce_type = routine[3]
    if reduce_type is None:
        reduce_type = _get_typename(out_types[0])

    t = (_get_typename(in_arg_dtype), _get_typename(out_arg_dtype))
    type_preamble = 'typedef %s type_in0_data; typedef %s type_out0_data;' % t

    params, ndims = _get_kernel_params(params, args_info)
    ndim_in = ndims['_in_ind']
    ndim_out = ndims['_out_ind']
    input_expr = input_expr
    output_expr = output_expr
    output_store = output_store
    clpy_variables_declaration = clpy_variables_declaration.format(
        ndim_in=ndim_in, ndim_out=ndim_out)
    return _get_simple_reduction_kernel(
        name, local_size, reduce_type, params, identity,
        routine[0], routine[1], routine[2],
        type_preamble, input_expr, output_expr, output_store, _preamble,
        options, clpy_variables_declaration)


class simple_reduction_function(object):

    _local_size = 256  # TODO(LWisteria): GetDeviceInfo
    _block_size = _local_size  # to keep compatibility with clpy

    def __init__(self, name, ops, identity, preamble, default=False):
        # TODO(tomoya.sakai): raw array may be possible
        #                     for simple_reduction_function
        self.name = name
        self._ops = ops
        self.identity = identity
        self._preamble = preamble
        self.nin = 1
        self.nout = 1
        in_params = _get_param_info('T in0', True)
        out_params = _get_param_info('T out0', False)
        self._params = (
            in_params + out_params +
            _get_param_info(
                'CIndexer _in_ind, CIndexer _out_ind', False) +
            _get_param_info('__kernel_arg_size_t _local_stride', True) +
            _get_param_info('LocalMem _sdata', True))
        self._input_expr = \
            '__attribute__((annotate("clpy_simple_reduction_tag:in"))) ' \
            'type_in0_data in0;'
        self._output_expr = \
            '__attribute__((annotate("clpy_simple_reduction_tag:out"))) ' \
            'type_out0_data out0;'
        self._clpy_variables_declaration = \
            '__attribute__((annotate("clpy_ignore"))) type_in0_data* in0_data;' \
            '__attribute__((annotate("clpy_ignore"))) CArray_{ndim_in} in0_info;' \
            '__attribute__((annotate("clpy_ignore"))) type_out0_data* out0_data;' \
            '__attribute__((annotate("clpy_ignore"))) CArray_{ndim_out} ' \
            'out0_info;' \
            'type_in0_data in0;'
        self._output_store = '__attribute__((annotate("clpy_simple_reduction_tag")))' \
                             'void __clpy_reduction_postprocess();'
        self._routine_cache = {}
        # default is True when identity for the kernel is None in clpy
        self.default = default

    def __call__(self, ndarray a, axis=None, dtype=None, ndarray out=None,
                 bint keepdims=False):
        cdef list in_args, out_args
        cdef tuple in_sahpe, laxis, raxis
        if dtype is not None:
            dtype = numpy.dtype(dtype).type

        in_args = [a]
        a_shape = a.shape
        if out is None:
            _preprocess_args((a,))
            out_args = []
        else:
            _preprocess_args((a, out))
            out_args = [out]

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        laxis, raxis = _get_axis(axis, a._shape.size())
        del axis  # to avoid bug
        out_shape = _get_out_shape(a_shape, laxis, raxis, keepdims)
        out_args = _get_out_args(out_args, out_types, out_shape, 'unsafe')
        if out_args[0].size == 0:
            if len(out_args) == 1:
                return out_args[0]
            return tuple(out_args)
        if a.size == 0 and (self.identity is None or self.default):
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % self.name)

        # TODO(tomoharu.kitawaki): debug 0 size reduction and remove this
        if a.size == 0:
            if self.identity in ['true', 'false']:
                ret_identity = True if self.identity=='true' else False
            else:
                ret_identity = self.identity
            ret = clpy.empty(out_shape, out_types[0])
            ret.fill(ret_identity)
            if out is not None:
                out.fill(ret_identity)
            return ret

        in_args, in_shape = _get_trans_args(
            in_args, laxis + raxis, a_shape, None)

        local_size = self._local_size
        in_indexer = Indexer(in_shape)
        out_indexer = Indexer(out_shape)
        # Rounding Up to the Next Power of 2
        # clp2_count >= in_indexer.size // out_indexer.size
        clp2_count = 1 << int.bit_length(
            int(in_indexer.size // out_indexer.size - 1))
        local_stride = max(1, local_size // clp2_count)

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, local_stride,
            self._params, True)
        args_info = _get_args_info(inout_args)

        kern = _get_simple_reduction_function(
            routine, self._params, args_info,
            in_args[0].dtype.type, out_args[0].dtype.type, out_types,
            self.name, local_size, self.identity,
            self._input_expr, self._output_expr, self._output_store,
            self._preamble, (), self._clpy_variables_declaration)

        # TODO(okuta) set actual size
        shared_mem = 32 * local_size

        kern.linear_launch(
            (out_indexer.size + local_stride - 1) // local_stride * local_size,
            inout_args, shared_mem, local_size)

        if len(out_args) == 1:
            return out_args[0]
        return tuple(out_args)


@util.memoize(for_each_device=True)
def _get_reduction_kernel(
        params, args_info, types,
        name, local_size, reduce_type, identity, map_expr, reduce_expr,
        post_map_expr, preamble, options):
    kernel_params, ndims = _get_kernel_params(params, args_info)
    ndim_in = ndims['_in_ind']
    ndim_out = ndims['_out_ind']
    arrays = [p for p, a in zip(params, args_info)
              if not p.raw and a[0] is ndarray]
    type_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k)
        for k, v in types)
    input_expr = '\n'.join(
        ['__attribute__((annotate("clpy_standard_reduction_tag:j"))) '
         '{type} {name};'.format(type=p.ctype, name=p.name)
         for p in arrays if p.is_const])
    output_expr = '\n'.join(
        ['__attribute__((annotate("clpy_standard_reduction_tag:i"))) '
         '{type} {name};'.format(type=p.ctype, name=p.name)
         for p in arrays if not p.is_const])
    output_store = '__attribute__((annotate("clpy_standard_reduction_tag"))) ' \
                   'void __clpy_reduction_postprocess();'
    clpy_variables_declaration = '\n'.join(
        ['__attribute__((annotate("clpy_ignore"))) {type}* {name}_data;'
         '__attribute__((annotate("clpy_ignore"))) CArray_{ndim} {name}_info;'
         '{type} {name};'
         .format(type=p.ctype, name=p.name, ndim=a[2])
         for p, a in zip(params, args_info) if a[0] is ndarray and not p.raw] +
        ['__attribute__((annotate("clpy_ignore"))) {type}* {name}_data;'
         '__attribute__((annotate("clpy_ignore"))) CArray_{ndim} {name}_info;'
         .format(type=p.ctype, name=p.name, ndim=a[2])
         for p, a in zip(params, args_info) if a[0] is ndarray and p.raw])
    return _get_simple_reduction_kernel(
        name, local_size, reduce_type, kernel_params, identity,
        map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, output_store, preamble,
        options, clpy_variables_declaration)


class ReductionKernel(object):

    """User-defined reduction kernel.

    This class can be used to define a reduction kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ReductionKernel.__call__` method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.clpy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        map_expr (str): Mapping expression for input values.
        reduce_expr (str): Reduction expression.
        post_map_expr (str): Mapping expression for reduced values.
        identity (str): Identity value for starting the reduction.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_type (str): Type of values to be used for reduction. This type
            is used to store the special variables ``a``.
        reduce_dims (bool): If ``True``, input arrays are reshaped without copy
            to smaller dimensions for efficiency.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        options (tuple of str): Additional compilation options.

    """
    def __init__(self, in_params, out_params,
                 map_expr, reduce_expr, post_map_expr,
                 identity, name='reduce_kernel', reduce_type=None,
                 reduce_dims=True, preamble='', options=()):
        if not clpy.backend.opencl.utility.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        self.params = (
            self.in_params + self.out_params +
            _get_param_info('CIndexer _in_ind, CIndexer _out_ind', False) +
            _get_param_info('__kernel_arg_size_t _local_stride', True) +
            _get_param_info('LocalMem _sdata', True))
        self.identity = identity
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.name = name
        self.options = options
        self.reduce_dims = reduce_dims
        self.post_map_expr = post_map_expr
        if reduce_type is None:
            self.reduce_type = self.out_params[0].ctype
        else:
            self.reduce_type = reduce_type
        self.preamble = preamble

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the reduction kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes, ndims, or axis are not
        compatible. It means that single ReductionKernel object may be compiled
        into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        out = kwargs.pop('out', None)
        axis = kwargs.pop('axis', None)
        keepdims = kwargs.pop('keepdims', False)
        stream = kwargs.pop('stream', None)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        out_args = list(args[self.nin:])
        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if len(out_args) != 0:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")
            out_args = [out]

        in_args = _preprocess_args(args[:self.nin])
        out_args = _preprocess_args(out_args)
        in_args, broad_shape = _broadcast(in_args, self.in_params, False)

        if self.identity is None and 0 in broad_shape:
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % self.name)

        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in out_args])
        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)

        axis, raxis = _get_axis(axis, len(broad_shape))
        out_shape = _get_out_shape(broad_shape, axis, raxis, keepdims)
        out_args = _get_out_args_with_params(
            out_args, out_types, out_shape, self.out_params, False)
        if 0 in out_shape:
            return out_args[0]

        in_args = [x if isinstance(x, ndarray) else t(x)
                   for x, t in zip(in_args, in_types)]
        in_args, in_shape = _get_trans_args(
            in_args, axis + raxis, broad_shape, self.in_params)

        local_size = 256  # TODO(LWisteria): GetDeviceInfo
        in_indexer = Indexer(in_shape)
        out_indexer = Indexer(out_shape)
        # Rounding Up to the Next Power of 2
        # clp2_count >= in_indexer.size // out_indexer.size
        clp2_count = 1 << int.bit_length(
            int(in_indexer.size // out_indexer.size - 1))
        local_stride = max(1, local_size // clp2_count)

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, local_stride,
            self.params, self.reduce_dims)

        args_info = _get_args_info(inout_args)

        kern = _get_reduction_kernel(
            self.params, args_info, types,
            self.name, local_size, self.reduce_type, self.identity,
            self.map_expr, self.reduce_expr, self.post_map_expr,
            self.preamble, self.options)

        # TODO(okuta) set actual size
        shared_mem = 32 * local_size

        kern.linear_launch(
            (out_indexer.size + local_stride - 1) // local_stride * local_size,
            inout_args, shared_mem, local_size)
        return out_args[0]


cpdef create_reduction_func(name, ops, routine=None, identity=None,
                            preamble='', default=False):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t
            rt = tuple([i or j for i, j in zip(rt, routine)])

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return simple_reduction_function(name, _ops, identity, preamble, default)
