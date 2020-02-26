# distutils: language = c++

import collections
import ctypes
import gc
import threading
import warnings
import weakref

from fastrlock cimport rlock

# from clpy.backend import driver
# from clpy.backend import runtime

from clpy.backend cimport device
# from clpy.backend cimport memory_hook
# from clpy.backend cimport runtime

import clpy.backend.opencl
cimport clpy.backend.opencl.utility
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
from clpy.backend.opencl.types cimport cl_mem

cimport libc.stdlib

thread_local = threading.local()

cdef inline _ensure_context(int device_id):

    """Ensure that CUcontext bound to the calling host thread exists.

    See discussion on https://github.com/clpy/clpy/issues/72 for details.
    """

    if not hasattr(thread_local, 'seen_devices'):
        thread_local.seen_devices = set()
    if device_id not in thread_local.seen_devices:
        raise NotImplementedError("clpy does not support this")
#        try:
#            driver.ctxGetCurrent()
#        except driver.CUDADriverError:
#            # Call Runtime API to establish context on this host thread.
#            runtime.memGetInfo()
        thread_local.seen_devices.add(device_id)


class OutOfMemoryError(MemoryError):

    def __init__(self, size, total):
        msg = 'out of memory to allocate %d bytes ' \
              '(total %d bytes)' % (size, total)
        super(OutOfMemoryError, self).__init__(msg)


cdef void ReleaseMemObjectHelper(Buf buf):
    clpy.backend.opencl.api.ReleaseMemObject(buf.ptr)


class Memory(object):

    """Memory allocation on a CUDA device.

    This class provides an RAII interface of the CUDA memory allocation.

    Args:
        ~Memory.device (clpy.cuda.Device): Device whose memory the pointer
            refers to.
        ~Memory.size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device = None
        self.buf = Buf()
        if size > 0:
            self.device = device.Device()
            self.buf = Buf(<size_t>clpy.backend.opencl.api.CreateBuffer(
                clpy.backend.opencl.env.get_context(),
                clpy.backend.opencl.api.MEM_READ_WRITE,
                size,
                <void*>NULL))

    def __del__(self):
        if not self.buf.isNull():
            ReleaseMemObjectHelper(self.buf)

    def __int__(self):
        """Returns the pointer value to the head of the allocation."""
        raise NotImplementedError("clpy does not support this")
        # return self.ptr


class ManagedMemory(Memory):

    """Managed memory (Unified memory) allocation on a CUDA device.

    This class provides an RAII interface of the CUDA managed memory
    allocation.

    Args:
        device (clpy.cuda.Device): Device whose memory the pointer refers to.
        size (int): Size of the memory allocation in bytes.

    """

    def __init__(self, Py_ssize_t size):
        self.size = size
        self.device = None
        self.ptr = 0
        if size > 0:
            self.device = device.Device()
            raise NotImplementedError("clpy does not support this")
            # self.ptr = runtime.mallocManaged(size)

    def prefetch(self, stream):
        """(experimental) Prefetch memory.

        Args:
            stream (clpy.cuda.Stream): CUDA stream.
        """
        raise NotImplementedError("clpy does not support this")
        # runtime.memPrefetchAsync(self.ptr, self.size, self.device.id,
        #                          stream.ptr)

    def advise(self, int advise, device.Device device):
        """(experimental) Advise about the usage of this memory.

        Args:
            advics (int): Advise to be applied for this memory.
            device (clpy.cuda.Device): Device to apply the advice for.

        """
        raise NotImplementedError("clpy does not support this")
        # runtime.memAdvise(self.ptr, self.size, advise, device.id)


cdef set _peer_access_checked = set()


cpdef _set_peer_access(int device, int peer):
    device_pair = device, peer

    if device_pair in _peer_access_checked:
        return
    raise NotImplementedError("clpy does not support this")
    # cdef int can_access = runtime.deviceCanAccessPeer(device, peer)
    # _peer_access_checked.add(device_pair)
    # if not can_access:
    #     return
    #
    # cdef int current = runtime.getDevice()
    # runtime.setDevice(device)
    # try:
    #     runtime.deviceEnablePeerAccess(peer)
    # finally:
    #     runtime.setDevice(current)

cdef class Buf:
    def __init__(self, size_t ptr=0):
        if ptr == 0:
            self.ptr = NULL
        else:
            self.ptr = <cl_mem>ptr

    cpdef size_t get(self):
        return <size_t>self.ptr

    def isNull(self):
        return self.ptr == NULL

cdef class Chunk:

    """A chunk points to a device memory.

    A chunk might be a splitted memory block from a larger allocation.
    The prev/next pointers contruct a doubly-linked list of memory addresses
    sorted by base address that must be contiguous.

    Args:
        mem (Memory): The device memory buffer.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.

    Attributes:
        device (clpy.cuda.Device): Device whose memory the pointer refers to.
        mem (Memory): The device memory buffer.
        ptr (int): Memory address.
        offset (int): An offset bytes from the head of the buffer.
        size (int): Chunk size in bytes.
        prev (Chunk): prev memory pointer if split from a larger allocation
        next (Chunk): next memory pointer if split from a larger allocation
    """

    def __init__(self, mem, Py_ssize_t offset, Py_ssize_t size):
        assert not mem.buf.isNull() or offset == 0
        self.mem = mem
        self.device = mem.device
        self.buf = mem.buf
        self.offset = offset
        self.size = size
        self.prev = None
        self.next = None

cdef class MemoryPointer:

    """Pointer to a point on a device memory.

    An instance of this class holds a reference to the original memory buffer
    and a pointer to a place within this buffer.

    Args:
        mem (Memory): The device memory buffer.
        offset (int): An offset from the head of the buffer to the place this
            pointer refers.

    Attributes:
        ~MemoryPointer.device (clpy.cuda.Device): Device whose memory the
            pointer refers to.
        mem (Memory): The device memory buffer.
        ptr (int): Pointer to the place within the buffer.
    """

    def __init__(self, mem, Py_ssize_t offset):
        assert (not mem.buf.isNull()) or offset == 0
        self.mem = mem
        self.offset = offset
        self.device = mem.device
        if offset == mem.size:
            self.buf = Buf()  # Not to make any error without access
        else:
            self.buf = mem.buf

    def __int__(self):
        """Returns the pointer value."""
        raise NotImplementedError("clpy does not support this")
        # return self.ptr

    def __add__(x, y):
        """Adds an offset to the pointer."""
        cdef MemoryPointer self
        cdef Py_ssize_t offset
        if isinstance(x, MemoryPointer):
            self = x
            offset = <Py_ssize_t?>y
        else:
            self = <MemoryPointer?>y
            offset = <Py_ssize_t?>x

        return MemoryPointer(self.mem, self.offset + offset)

    def __iadd__(self, Py_ssize_t offset):
        """Adds an offset to the pointer in place."""
        raise NotImplementedError("clpy does not support this")
#        assert self.ptr > 0 or offset == 0
#        self.ptr += offset
#        return self

    def __sub__(self, offset):
        """Subtracts an offset from the pointer."""
        return self + -offset

    def __isub__(self, Py_ssize_t offset):
        """Subtracts an offset from the pointer in place."""
        return self.__iadd__(-offset)

    cpdef copy_from_device(self, MemoryPointer src, Py_ssize_t size):
        """Copies a memory sequence from a (possibly different) device.

        Args:
            src (clpy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        cdef void* tmp
        if size > 0:
            if src.device == self.device:
                with self.device:
                    queue = clpy.backend.opencl.env.get_command_queue()
                    clpy.backend.opencl.api.EnqueueCopyBuffer(
                        command_queue=queue,
                        src_buffer=src.buf.ptr,
                        dst_buffer=self.buf.ptr,
                        src_offset=src.cl_mem_offset(),
                        dst_offset=self.cl_mem_offset(),
                        cb=size)
                    clpy.backend.opencl.api.Finish(queue)
            else:
                tmp = libc.stdlib.malloc(size)
                with src.device:
                    queue = clpy.backend.opencl.env.get_command_queue()
                    clpy.backend.opencl.api.EnqueueReadBuffer(
                        command_queue=queue,
                        buffer=src.buf.ptr,
                        blocking_read=clpy.backend.opencl.api.BLOCKING,
                        offset=src.cl_mem_offset(),
                        cb=size,
                        host_ptr=tmp)
                    clpy.backend.opencl.api.Finish(queue)
                with self.device:
                    queue = clpy.backend.opencl.env.get_command_queue()
                    clpy.backend.opencl.api.EnqueueWriteBuffer(
                        command_queue=queue,
                        buffer=self.buf.ptr,
                        blocking_write=clpy.backend.opencl.api.BLOCKING,
                        offset=self.cl_mem_offset(),
                        cb=size,
                        host_ptr=tmp)
                    clpy.backend.opencl.api.Finish(queue)
                libc.stdlib.free(tmp)

    cpdef copy_from_device_async(self, MemoryPointer src, size_t size, stream):
        """Copies a memory from a (possibly different) device asynchronously.

        Args:
            src (clpy.cuda.MemoryPointer): Source memory pointer.
            size (int): Size of the sequence in bytes.
            stream (clpy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            # TODO(LWisteria): Impelment async
            self.copy_from_device(src, size)

    cpdef copy_from_host(self, mem, size_t size):
        """Copies a memory sequence from the host memory.

        Args:
            mem (ctypes.c_void_p): Source memory pointer.
            size (int): Size of the sequence in bytes.

        """
        cdef size_t host_ptr = mem.value
        if size > 0:
            with self.device:
                queue = clpy.backend.opencl.env.get_command_queue()
                clpy.backend.opencl.api.EnqueueWriteBuffer(
                    command_queue=queue,
                    buffer=self.buf.ptr,
                    blocking_write=clpy.backend.opencl.api.BLOCKING,
                    offset=self.cl_mem_offset(),
                    cb=size,
                    host_ptr=<void*>host_ptr)
                clpy.backend.opencl.api.Finish(queue)

    cpdef copy_from_host_async(self, mem, size_t size, stream):
        """Copies a memory sequence from the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Source memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (clpy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            # TODO(LWisteria): Impelment async
            self.copy_from_host(mem, size)

    cpdef copy_from(self, mem, size_t size):
        """Copies a memory sequence from a (possibly different) device or host.

        This function is a useful interface that selects appropriate one from
        :meth:`~clpy.cuda.MemoryPointer.copy_from_device` and
        :meth:`~clpy.cuda.MemoryPointer.copy_from_host`.

        Args:
            mem (ctypes.c_void_p or clpy.cuda.MemoryPointer): Source memory
                pointer.
            size (int): Size of the sequence in bytes.

        """
        if isinstance(mem, MemoryPointer):
            self.copy_from_device(mem, size)
        else:
            self.copy_from_host(mem, size)

    cpdef copy_from_async(self, mem, size_t size, stream):
        """Copies a memory sequence from an arbitrary place asynchronously.

        This function is a useful interface that selects appropriate one from
        :meth:`~clpy.cuda.MemoryPointer.copy_from_device_async` and
        :meth:`~clpy.cuda.MemoryPointer.copy_from_host_async`.

        Args:
            mem (ctypes.c_void_p or clpy.cuda.MemoryPointer): Source memory
                pointer.
            size (int): Size of the sequence in bytes.
            stream (clpy.cuda.Stream): CUDA stream.

        """
        if isinstance(mem, MemoryPointer):
            self.copy_from_device_async(mem, size, stream)
        else:
            self.copy_from_host_async(mem, size, stream)

    cpdef copy_to_host(self, mem, size_t size):
        """Copies a memory sequence to the host memory.

        Args:
            mem (ctypes.c_void_p): Target memory pointer.
            size (int): Size of the sequence in bytes.

        """
        cdef size_t host_ptr = mem.value
        if size > 0:
            with self.device:
                queue = clpy.backend.opencl.env.get_command_queue()
                clpy.backend.opencl.api.EnqueueReadBuffer(
                    command_queue=queue,
                    buffer=self.buf.ptr,
                    blocking_read=clpy.backend.opencl.api.BLOCKING,
                    offset=self.cl_mem_offset(),
                    cb=size,
                    host_ptr=<void*>host_ptr)
                clpy.backend.opencl.api.Finish(queue)

    cpdef copy_to_host_async(self, mem, size_t size, stream):
        """Copies a memory sequence to the host memory asynchronously.

        Args:
            mem (ctypes.c_void_p): Target memory pointer. It must be a pinned
                memory.
            size (int): Size of the sequence in bytes.
            stream (clpy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            # TODO(LWisteria): Impelment async
            self.copy_to_host(mem, size)

    cpdef memset(self, int value, size_t size):
        """Fills a memory sequence by constant byte value.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.

        """
        if size > 0:
            queue = clpy.backend.opencl.env.get_command_queue()
            clpy.backend.opencl.api.EnqueueFillBuffer(
                command_queue=queue,
                buffer=self.buf.ptr,
                pattern=&value,
                pattern_size=sizeof(value),
                offset=0,
                size=size)
            clpy.backend.opencl.api.Finish(queue)

    cpdef memset_async(self, int value, size_t size, stream):
        """Fills a memory sequence by constant byte value asynchronously.

        Args:
            value (int): Value to fill.
            size (int): Size of the sequence in bytes.
            stream (clpy.cuda.Stream): CUDA stream.

        """
        if size > 0:
            # TODO(LWisteria): Impelment async
            self.memset(value, size)

    cpdef Py_ssize_t cl_mem_offset(self):
        cdef Py_ssize_t ret = self.offset
        if (isinstance(self.mem, PooledMemory)):
            ret = ret + self.mem.offset
        return ret

cpdef MemoryPointer _malloc(Py_ssize_t size):
    mem = Memory(size)
    return MemoryPointer(mem, 0)


cpdef MemoryPointer malloc_managed(Py_ssize_t size):
    """Allocate managed memory (unified memory).

    This method can be used as a CuPy memory allocator. The simplest way to
    use a managed memory as the default allocator is the following code::

        set_allocator(malloc_managed)

    The advantage using managed memory in CuPy is that device memory
    oversubscription is possible for GPUs that have a non-zero value for the
    device attribute cudaDevAttrConcurrentManagedAccess.
    CUDA >= 8.0 with GPUs later than or equal to Pascal is preferrable.

    Read more at: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#axzz4qygc1Ry1  # NOQA

    Args:
        size (int): Size of the memory allocation in bytes.

    Returns:
        ~clpy.cuda.MemoryPointer: Pointer to the allocated buffer.
    """
    mem = ManagedMemory(size)
    return MemoryPointer(mem, 0)


cdef object _current_allocator = _malloc


cpdef MemoryPointer alloc(Py_ssize_t size):
    """Calls the current allocator.

    Use :func:`~clpy.cuda.set_allocator` to change the current allocator.

    Args:
        size (int): Size of the memory allocation.

    Returns:
        ~clpy.cuda.MemoryPointer: Pointer to the allocated buffer.

    """
    return _current_allocator(size)


cpdef set_allocator(allocator=_malloc):
    """Sets the current allocator.

    Args:
        allocator (function): CuPy memory allocator. It must have the same
            interface as the :func:`clpy.cuda.alloc` function, which takes the
            buffer size as an argument and returns the device buffer of that
            size.

    """
    global _current_allocator
    _current_allocator = allocator


class PooledMemory(Memory):

    """Memory allocation for a memory pool.

    The instance of this class is created by memory pool allocator, so user
    should not instantiate it by hand.

    """

    def __init__(self, Chunk chunk, pool):
        self.device = chunk.device
        self.buf = chunk.buf
        self.size = chunk.size
        self.pool = pool
        self.offset = chunk.offset

    def free(self):
        """Frees the memory buffer and returns it to the memory pool.

        This function actually does not free the buffer. It just returns the
        buffer to the memory pool for reuse.

        """
        buf = self.buf
        if buf.isNull():
            return
        pool = self.pool()
        if pool is None:
            return

        pool.free(buf, self.size, self.offset)
#        hooks = None
#        # to avoid error at exit
#        hooks = memory_hook.get_memory_hooks()
#        size = self.size
#        if hooks:
#            device_id = self.device.id
#            pmem_id = id(self)
#            hooks_values = hooks.values()  # avoid six for performance
#            for hook in hooks_values:
#                hook.free_preprocess(device_id=device_id,
#                                     mem_size=size,
#                                     mem_ptr=ptr,
#                                     pmem_id=pmem_id)
#            try:
#                pool.free(ptr, size)
#            finally:
#                for hook in hooks_values:
#                    hook.free_postprocess(device_id=device_id,
#                                          mem_size=size,
#                                          mem_ptr=ptr,
#                                          pmem_id=pmem_id)
#        else:
#            pool.free(ptr, size)

    __del__ = free


cdef class SingleDeviceMemoryPool:
    """Memory pool implementation for single device.

    - The allocator attempts to find the smallest cached block that will fit
      the requested size. If the block is larger than the requested size,
      it may be split. If no block is found, the allocator will delegate to
      cudaMalloc.
    - If the cudaMalloc fails, the allocator will free all cached blocks that
      are not split and retry the allocation.
    """

    def __init__(self, allocator=None):
        if allocator is None:
            allocator = _malloc
        self._allocation_unit_size = \
            clpy.backend.opencl.utility.GetDeviceMemBaseAddrAlign(
                clpy.backend.opencl.env.get_device()) // 8
        self._initial_bins_size = 1024
        self._in_use = {}
        self._free = [None] * self._initial_bins_size
        self._allocator = allocator
        self._weakref = weakref.ref(self)
        self._device_id = device.get_device_id()
        self._free_lock = rlock.create_fastrlock()
        self._in_use_lock = rlock.create_fastrlock()

    cpdef Py_ssize_t _round_size(self, Py_ssize_t size):
        """Round up the memory size to fit memory alignment of cudaMalloc."""
        unit = self._allocation_unit_size
        return (((size + unit - 1) // unit) * unit)

    cpdef Py_ssize_t _bin_index_from_size(self, Py_ssize_t size):
        """Get appropriate bins (_free) index from the memory size"""
        unit = self._allocation_unit_size
        return (size - 1) // unit

    cpdef void _grow_free_if_necessary(self, Py_ssize_t size):
        """Extend bins (_free) size if necessary"""
        current_size = len(self._free)
        if current_size >= size:
            return
        growth_size = size - current_size
        growth = [None] * growth_size
        self._free.extend(growth)

    cpdef tuple _split(self, Chunk chunk, Py_ssize_t size):
        """Split contiguous block of a larger allocation"""
        assert chunk.size >= size
        if chunk.size == size:
            return (chunk, None)
        cdef Chunk head
        cdef Chunk remaining
        head = Chunk(chunk.mem, chunk.offset, size)
        remaining = Chunk(chunk.mem, chunk.offset + size, chunk.size - size)
        if chunk.prev is not None:
            head.prev = chunk.prev
            chunk.prev.next = head
        if chunk.next is not None:
            remaining.next = chunk.next
            chunk.next.prev = remaining
        head.next = remaining
        remaining.prev = head
        return head, remaining

    cpdef Chunk _merge(self, Chunk head, Chunk remaining):
        """Merge previously splitted block (chunk)"""
        cdef Chunk merged
        size = head.size + remaining.size
        merged = Chunk(head.mem, head.offset, size)
        if head.prev is not None:
            merged.prev = head.prev
            merged.prev.next = merged
        if remaining.next is not None:
            merged.next = remaining.next
            merged.next.prev = merged
        return merged

    cpdef MemoryPointer _alloc(self, Py_ssize_t rounded_size):
        return self._allocator(rounded_size)
        # hooks = memory_hook.get_memory_hooks()
        # if hooks:
        #     memptr = None
        #     device_id = self._device_id
        #     hooks_values = hooks.values()  # avoid six for performance
        #     for hook in hooks_values:
        #         hook.alloc_preprocess(device_id=device_id,
        #                               mem_size=rounded_size)
        #     try:
        #         memptr = self._allocator(rounded_size)
        #     finally:
        #         for hook in hooks_values:
        #             mem_ptr = memptr.ptr if memptr is not None else 0
        #             hook.alloc_postprocess(device_id=device_id,
        #                                    mem_size=rounded_size,
        #                                    mem_ptr=mem_ptr)
        #     return memptr
        # else:
        #     return self._allocator(rounded_size)

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        rounded_size = self._round_size(size)
        return self._malloc(rounded_size)
        # hooks = memory_hook.get_memory_hooks()
        # if hooks:
        #     memptr = None
        #     device_id = self._device_id
        #     hooks_values = hooks.values()  # avoid six for performance
        #     for hook in hooks_values:
        #         hook.malloc_preprocess(device_id=device_id,
        #                                size=size,
        #                                mem_size=rounded_size)
        #     try:
        #         memptr = self._malloc(rounded_size)
        #     finally:
        #         if memptr is None:
        #             mem_ptr = 0
        #             pmem_id = 0
        #         else:
        #             mem_ptr = memptr.ptr
        #             pmem_id = id(memptr.mem)
        #         for hook in hooks_values:
        #             hook.malloc_postprocess(device_id=device_id,
        #                                     size=size,
        #                                     mem_size=rounded_size,
        #                                     mem_ptr=mem_ptr,
        #                                     pmem_id=pmem_id)
        #     return memptr
        # else:
        #     return self._malloc(rounded_size)

    cpdef MemoryPointer _malloc(self, Py_ssize_t size):
        cdef set free_list = None
        cdef Chunk chunk = None
        cdef Chunk remaining = None

        if size == 0:
            return MemoryPointer(Memory(0), 0)

        index = self._bin_index_from_size(size)
        # find best-fit, or a smallest larger allocation
        length = len(self._free)
        for i in range(index, length):
            free_list = self._free[i]
            if free_list:
                try:
                    rlock.lock_fastrlock(self._free_lock, -1, True)
                    free_list = self._free[i]
                    if free_list:
                        chunk = free_list.pop()
                        break
                finally:
                    rlock.unlock_fastrlock(self._free_lock)

        if chunk is not None:
            # # TODO(LWisteria): need on OpenCL?
            # _ensure_context(self._device_id)
            chunk, remaining = self._split(chunk, size)
        else:
            # cudaMalloc if a cache is not found
            mem = self._alloc(size).mem
            # try:
            #     mem = self._alloc(size).mem
            # except runtime.CUDARuntimeError as e:
            #     if e.status != runtime.errorMemoryAllocation:
            #         raise
            #     self.free_all_blocks()
            #     try:
            #         mem = self._alloc(size).mem
            #     except runtime.CUDARuntimeError as e:
            #         if e.status != runtime.errorMemoryAllocation:
            #             raise
            #         gc.collect()
            #         try:
            #             mem = self._alloc(size).mem
            #         except runtime.CUDARuntimeError as e:
            #             if e.status != runtime.errorMemoryAllocation:
            #                 raise
            #             else:
            #                 total = size + self.total_bytes()
            #                 raise OutOfMemoryError(size, total)
            chunk = Chunk(mem, 0, size)

        try:
            rlock.lock_fastrlock(self._in_use_lock, -1, True)
            self._in_use[(chunk.buf, chunk.offset)] = chunk
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        if remaining:
            remaining_index = self._bin_index_from_size(remaining.size)
            try:
                rlock.lock_fastrlock(self._free_lock, -1, True)
                free_list = self._free[remaining_index]
                if free_list is None:
                    self._free[remaining_index] = free_list = set()
                free_list.add(remaining)
            finally:
                rlock.unlock_fastrlock(self._free_lock)
        pmem = PooledMemory(chunk, self._weakref)
        return MemoryPointer(pmem, 0)

    cpdef free(self, Buf buf, Py_ssize_t size, Py_ssize_t offset):
        cdef set free_list = None
        cdef Chunk chunk
        cdef int index

        try:
            rlock.lock_fastrlock(self._in_use_lock, -1, True)
            chunk = self._in_use.pop((buf, offset), None)
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        if chunk is None:
            raise RuntimeError('Cannot free out-of-pool memory')

        if chunk.next:
            chunk_next = None
            index = self._bin_index_from_size(chunk.next.size)
            try:
                rlock.lock_fastrlock(self._free_lock, -1, True)
                free_list = self._free[index]
                if free_list is not None and chunk.next in free_list:
                    free_list.remove(chunk.next)
                    chunk_next = chunk.next
            finally:
                rlock.unlock_fastrlock(self._free_lock)
            if chunk_next:
                chunk = self._merge(chunk, chunk_next)

        if chunk.prev:
            chunk_prev = None
            index = self._bin_index_from_size(chunk.prev.size)
            try:
                rlock.lock_fastrlock(self._free_lock, -1, True)
                free_list = self._free[index]
                if free_list is not None and chunk.prev in free_list:
                    free_list.remove(chunk.prev)
                    chunk_prev = chunk.prev
            finally:
                rlock.unlock_fastrlock(self._free_lock)
            if chunk_prev:
                chunk = self._merge(chunk_prev, chunk)

        index = self._bin_index_from_size(chunk.size)
        self._grow_free_if_necessary(index + 1)
        try:
            rlock.lock_fastrlock(self._free_lock, -1, True)
            free_list = self._free[index]
            if free_list is None:
                self._free[index] = free_list = set()
            free_list.add(chunk)
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef free_all_blocks(self):
        cdef set free_list = None
        cdef set keep_list = None
        # Free all **non-split** chunks
        try:
            rlock.lock_fastrlock(self._free_lock, -1, True)
            for i in range(len(self._free)):
                free_list = self._free[i]
                if free_list is None:
                    continue
                keep_list = set()
                for chunk in free_list:
                    if chunk.prev or chunk.next:
                        keep_list.add(chunk)
                self._free[i] = keep_list
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        cdef Py_ssize_t n = 0
        cdef set free_list
        try:
            rlock.lock_fastrlock(self._free_lock, -1, True)
            for free_list in self._free:
                if free_list is not None:
                    n += len(free_list)
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return n

    cpdef used_bytes(self):
        cdef Py_ssize_t size = 0
        try:
            rlock.lock_fastrlock(self._in_use_lock, -1, True)
            for chunk in self._in_use.itervalues():
                size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        return size

    cpdef free_bytes(self):
        cdef Py_ssize_t size = 0
        cdef set free_list = None
        try:
            rlock.lock_fastrlock(self._free_lock, -1, True)
            for free_list in self._free:
                if free_list is not None:
                    for chunk in free_list:
                        size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return size

    cpdef total_bytes(self):
        return self.used_bytes() + self.free_bytes()


cdef class MemoryPool(object):

    """Memory pool for all devices on the machine.

    A memory pool preserves any allocations even if they are freed by the user.
    Freed memory buffers are held by the memory pool as *free blocks*, and they
    are reused for further memory allocations of the same sizes. The allocated
    blocks are managed for each device, so one instance of this class can be
    used for multiple devices.

    .. note::
       When the allocation is skipped by reusing the pre-allocated block, it
       does not call ``cudaMalloc`` and therefore CPU-GPU synchronization does
       not occur. It makes interleaves of memory allocations and kernel
       invocations very fast.

    .. note::
       The memory pool holds allocated blocks without freeing as much as
       possible. It makes the program hold most of the device memory, which may
       make other CUDA programs running in parallel out-of-memory situation.

    Args:
        allocator (function): The base CuPy memory allocator. It is used for
            allocating new blocks when the blocks of the required size are all
            in use.

    """

    def __init__(self, allocator=_malloc):
        self._pools = collections.defaultdict(
            lambda: SingleDeviceMemoryPool(allocator))

    cpdef MemoryPointer malloc(self, Py_ssize_t size):
        """Allocates the memory, from the pool if possible.

        This method can be used as a CuPy memory allocator. The simplest way to
        use a memory pool as the default allocator is the following code::

            set_allocator(MemoryPool().malloc)

        Also, the way to use a memory pool of Managed memory (Unified memory)
        as the default allocator is the following code::

            set_allocator(MemoryPool(malloc_managed).malloc)

        Args:
            size (int): Size of the memory buffer to allocate in bytes.

        Returns:
            ~clpy.cuda.MemoryPointer: Pointer to the allocated buffer.

        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.malloc(size)

    cpdef free_all_blocks(self):
        """Release free blocks."""
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        mp.free_all_blocks()

    cpdef free_all_free(self):
        """Release free blocks."""
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        """Count the total number of free blocks.

        Returns:
            int: The total number of free blocks.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.n_free_blocks()

    cpdef used_bytes(self):
        """Get the total number of bytes used.

        Returns:
            int: The total number of bytes used.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.used_bytes()

    cpdef free_bytes(self):
        """Get the total number of bytes acquired but not used in the pool.

        Returns:
            int: The total number of bytes acquired but not used in the pool.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.free_bytes()

    cpdef total_bytes(self):
        """Get the total number of bytes acquired in the pool.

        Returns:
            int: The total number of bytes acquired in the pool.
        """
        mp = <SingleDeviceMemoryPool>self._pools[device.get_device_id()]
        return mp.total_bytes()
