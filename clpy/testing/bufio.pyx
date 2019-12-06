# cython: profile=True
cimport clpy.backend.opencl.api
import clpy.backend.opencl.api
cimport clpy.backend.opencl.env
from clpy.backend.memory cimport Buf

cpdef writebuf(Buf buffer_to_write, n_bytes, host_ptr, offset=0):
    cdef size_t host_ptr_sizet = host_ptr
    clpy.backend.opencl.api.EnqueueWriteBuffer(
        clpy.backend.opencl.env.get_command_queue(),
        buffer_to_write.ptr,
        clpy.backend.opencl.api.BLOCKING,
        offset,
        n_bytes,
        <void*>host_ptr_sizet)

cpdef readbuf(Buf buffer_to_read, n_bytes, host_ptr, offset=0):
    cdef size_t host_ptr_sizet = host_ptr
    clpy.backend.opencl.api.EnqueueReadBuffer(
        clpy.backend.opencl.env.get_command_queue(),
        buffer_to_read.ptr,
        clpy.backend.opencl.api.BLOCKING,
        offset,
        n_bytes,
        <void*>host_ptr_sizet)
