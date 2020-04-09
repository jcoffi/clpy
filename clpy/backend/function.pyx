# distutils: language = c++

import numpy
import six

cimport cpython
from libcpp cimport vector
import cython

from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS
from cpython.buffer cimport PyBUF_SIMPLE
from cpython.buffer cimport PyBuffer_Release
from cpython.buffer cimport PyObject_GetBuffer

# from clpy.cuda cimport driver
from clpy.core cimport core
import clpy.backend.opencl
cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.utility
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
import clpy.core

include "clpy/backend/opencl/types.pxi"

cdef inline size_t _get_stream(strm) except *:
    return 0 if strm is None else strm.ptr


DEF MAX_NDIM = 25

cdef struct _CIndexer:
    Py_ssize_t size
    Py_ssize_t shape_and_index[MAX_NDIM * 2]

cdef struct _CArray:
    Py_ssize_t offset
    Py_ssize_t size
    Py_ssize_t shape_and_index[MAX_NDIM * 2]

cdef struct _CArray0:
    char unused

cdef void _launch(clpy.backend.opencl.types.cl_kernel kernel, global_work_size,
                  local_work_size, args, Py_ssize_t local_mem) except *:
    global_dim = len(global_work_size)
    local_dim = len(local_work_size)
    if global_dim < 1 or 3 < global_dim:
        raise ValueError("Global workitem dimension should be 1,2,3"
                         " but {0} was given".format(global_dim))
    elif local_dim < 0 or 3 < local_dim:
        raise ValueError("Local workitem dimension should be 0,1,2,3"
                         " but {0} was given".format(local_dim))
    elif local_dim > 0 and global_dim != local_dim:
        raise ValueError("global_work_size dim is {0} but local is {1}"
                         .format(global_dim, local_dim))

    cdef size_t i = 0
    cdef _CIndexer indexer  # to keep lifetime until SetKernelArg
    cdef _CArray arrayInfo  # to keep lifetime until SetKernelArg
    cdef core.ndarray a_ndarray
    cdef size_t ndim = 0
    cdef size_t d = 0
    cdef size_t ptr = 0
    cdef size_t size = 0
    cdef size_t buffer_object = 0
    cdef cl_ulong imm_ulong
    cdef cl_uint imm_uint
    cdef cl_long imm_long
    cdef cl_double imm_double
    cdef cl_char imm_char
    cdef Py_buffer py_buffer

    for a in args:
        if isinstance(a, core.ndarray):
            # Note(y1r):
            # We give a type hint to Cython to optimize property access.
            # Without a type hint, Cython deal `a` as Python Object so
            # dynamic property access (__Pyx_PyObject_GetAttrStr) is required.
            # By giving type hint, Cython can use static property access for
            # `cdef` property. More detail, refer to PR#285.
            a_ndarray = <core.ndarray>a

            buffer_object = a_ndarray.data.buf.get()
            clpy.backend.opencl.api.SetKernelArg(kernel, i, sizeof(void*),
                                                 <void*>&buffer_object)
            i+=1

            ndim = a_ndarray._strides.size()
            for d in range(ndim):
                if a_ndarray._strides[d] % a_ndarray.itemsize != 0:
                    raise ValueError("Stride of dim {0} = {1},"
                                     " but item size is {2}"
                                     .format(d, a_ndarray._strides[d],
                                             a_ndarray.itemsize))
                arrayInfo.shape_and_index[d] = a_ndarray._shape[d]
                arrayInfo.shape_and_index[d + ndim] = a_ndarray._strides[d]
            arrayInfo.offset = a_ndarray.data.cl_mem_offset()
            arrayInfo.size = a_ndarray.size

            clpy.backend.opencl.api.SetKernelArg(
                kernel, i, cython.sizeof(Py_ssize_t)*(1+1+2*ndim),
                <void*>&arrayInfo)
        elif isinstance(a, clpy.core.core.LocalMem):
            clpy.backend.opencl.utility.SetKernelArgLocalMemory(kernel, i,
                                                                local_mem)
        else:
            if isinstance(a, core.Indexer):
                for d in range(a.ndim):
                    indexer.shape_and_index[d] = a.shape[d]
                ptr = <size_t>&indexer
                indexer.size = a.size
                size = a.get_size()
            else:
                if isinstance(a, clpy.core.core.Size_t):
                    if clpy.backend.opencl.utility.typeof_size() \
                            == 'uint':
                        imm_uint = <cl_uint>a.val
                        ptr = <size_t>&(imm_uint)
                        size = cython.sizeof(cl_uint)
                    elif clpy.backend.opencl.utility.typeof_size() \
                            == 'ulong':
                        imm_ulong = <cl_ulong>a.val
                        ptr = <size_t>&(imm_ulong)
                        size = cython.sizeof(cl_ulong)
                    else:
                        raise "api_sizeof_size is illegal"
                elif isinstance(a, int):
                    imm_long = <cl_long>a
                    ptr = <size_t>&(imm_long)
                    size = cython.sizeof(cl_long)
                elif isinstance(a, float):
                    imm_double = <cl_double>a
                    ptr = <size_t>&(imm_double)
                    size = cython.sizeof(cl_double)
                elif isinstance(a, bool):
                    imm_char = <cl_char>a
                    ptr = <size_t>&(imm_char)
                    size = cython.sizeof(cl_char)
                elif type(a) in core.numpy_scalar_type_set():
                    PyObject_GetBuffer(a,
                                       &py_buffer,
                                       PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
                    ptr = <size_t>py_buffer.buf
                    size = py_buffer.len
                    PyBuffer_Release(&py_buffer)
                else:
                    raise TypeError('Unsupported type %s' % type(a))
            clpy.backend.opencl.api.SetKernelArg(kernel, i, size, <void*>ptr)
        i+=1

    cdef size_t gws[3]
    for i in range(global_dim):
        gws[i] = global_work_size[i]

    cdef size_t lws[3]
    cdef size_t* lws_ptr
    if local_dim > 0:
        for i in range(local_dim):
            lws[i] = local_work_size[i]
        lws_ptr = &lws[0]
    else:
        lws_ptr = <size_t*>NULL

    clpy.backend.opencl.api.EnqueueNDRangeKernel(
        command_queue=clpy.backend.opencl.env.get_command_queue(),
        kernel=kernel,
        work_dim=global_dim,  # asserted to be equal to local_dim
        global_work_offset=<size_t*>NULL,
        global_work_size=&gws[0],
        local_work_size=lws_ptr)


cdef class Function:

    """CUDA kernel function."""

    def __init__(self, Module module, str funcname):
        self.module = module  # to keep module loaded
        self.kernel = clpy.backend.opencl.api.CreateKernel(
            module.program, funcname.encode('utf-8'))

    def __call__(self,
                 tuple global_work_size,
                 tuple local_work_size,
                 args,
                 size_t local_mem=0,
                 stream=None):
        global_work_size = (global_work_size + (1, 1))[:3]
        local_work_size = (local_work_size + (1, 1))[:3]
        # s = _get_stream(stream)
        if stream is not None:
            raise NotImplementedError("clpy does not support CUDA stream")
        gws_3d = [
            max(1, global_work_size[0]),
            max(1, global_work_size[1]),
            max(1, global_work_size[2])
        ]
        lws_3d = [
            max(1, local_work_size[0]),
            max(1, local_work_size[1]),
            max(1, local_work_size[2])
        ]
        _launch(self.kernel, gws_3d, lws_3d, args, local_mem)

    cpdef linear_launch(self, size_t size, args, size_t local_mem=0,
                        size_t local_size=0):
        # TODO(beam2d): Tune it
        if local_size == 0:
            local_work_size = []
        else:
            local_work_size = [local_size, ]
        _launch(self.kernel, [size, ], local_work_size, args, local_mem)


cdef class Module:

    """CUDA kernel module."""

    def __init__(self):
        pass

    def __del__(self):
        pass

    cpdef load_file(self, str filename):
        raise NotImplementedError("clpy does not support this")
#        self.ptr = driver.moduleLoad(filename)

    cpdef load(self, bytes cubin):
        raise NotImplementedError("clpy does not support this")
#        self.ptr = driver.moduleLoadData(cubin)

    cpdef get_global_var(self, str name):
        raise NotImplementedError("clpy does not support this")
#        return driver.moduleGetGlobal(self.ptr, name)

    cdef set(self, clpy.backend.opencl.types.cl_program program):
        self.program = program

    cpdef get_function(self, str name):
        return Function(self, name)


cdef class LinkState:

    """CUDA link state."""

    def __init__(self):
        raise NotImplementedError("clpy does not support this")
#        self.ptr = driver.linkCreate()

    def __del__(self):
        if self.ptr:
            raise NotImplementedError("clpy does not support this")
#            driver.linkDestroy(self.ptr)
            self.ptr = 0

    cpdef add_ptr_data(self, unicode data, unicode name):
        cdef bytes data_byte = data.encode()
        raise NotImplementedError("clpy does not support this")
#        driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data_byte, name)

    cpdef bytes complete(self):
        raise NotImplementedError("clpy does not support this")
#        cubin = driver.linkComplete(self.ptr)
        return cubin
