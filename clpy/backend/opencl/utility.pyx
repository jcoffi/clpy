# cython: profile=True
import env
import os
import re
import tempfile
import time

cimport api
cimport env
import cython
from cpython cimport array
from exceptions cimport check_status
from libc.stdlib cimport malloc
from libc.string cimport memcpy

###############################################################################
# helpers

cdef cl_uint GetDeviceMemBaseAddrAlign(cl_device_id device):
    cdef cl_uint[1] valptrs
    cdef size_t[1] retptrs
    cdef cl_int status = api.clGetDeviceInfo(
        device,
        <cl_device_info>CL_DEVICE_MEM_BASE_ADDR_ALIGN,
        <size_t>sizeof(cl_uint),
        <void *>&valptrs[0],
        <size_t *>&retptrs[0])
    check_status(status)

    ret = valptrs[0]
    return ret

cdef GetDeviceAddressBits(cl_device_id device):
    cdef cl_uint[1] valptrs
    cdef size_t[1] retptrs
    cdef cl_int status = api.clGetDeviceInfo(
        device,
        <cl_device_info>CL_DEVICE_ADDRESS_BITS,
        <size_t>sizeof(cl_uint),
        <void *>&valptrs[0],
        <size_t *>&retptrs[0])
    check_status(status)

    ret = valptrs[0]
    return ret

cpdef GetDeviceMaxMemoryAllocation(int device_id):
    cdef cl_ulong ret
    cdef cl_int status = api.clGetDeviceInfo(
        env.get_devices()[device_id],
        <cl_device_info>CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(ret),
        <void *>&ret,
        <size_t*>NULL)
    check_status(status)

    return ret

cdef GetProgramDevice(cl_program program):
    cdef size_t max_length = sizeof(cl_device_id)*env.num_devices
    cdef cl_device_id* val = <cl_device_id*>malloc(max_length)
    cdef size_t length
    cdef cl_int status = api.clGetProgramInfo(
        program,
        <cl_program_info>CL_PROGRAM_DEVICES,
        max_length,
        <void *>&val[0],
        <size_t *>&length)
    check_status(status)
    cdef count = length/sizeof(cl_device_id)

    # Get Number
    devices = [-1] * count
    cdef cl_device_id device_i
    cdef cl_device_id device_j
    for i in range(env.num_devices):
        device_i = env.get_devices()[i]
        for j in range(count):
            device_j = val[j]
            if device_j == device_i:
                devices[j] = i
                break
    return devices

###############################################################################
# utility

cdef void SetKernelArgLocalMemory(cl_kernel kernel, arg_index, size_t size):
    api.SetKernelArg(kernel, arg_index, size, <void*>NULL)

cdef is_valid_kernel_name(name):
    return re.match('^[a-zA-Z_][a-zA-Z_0-9]*$', name) is not None

cdef cl_program CreateProgram(sources, cl_context context, num_devices,
                              cl_device_id* devices_ptrs,
                              options=b"") except *:
    cdef size_t length = len(sources)
    cdef char** src
    cdef size_t* src_size
    src = <char**>malloc(sizeof(const char*)*length)
    src_size = <size_t*>malloc(sizeof(size_t)*length)
    for i, s in enumerate(sources):
        src_size[i] = len(s)
        src[i] = <char*>malloc(sizeof(char)*src_size[i])
        memcpy(src[i], <char*>s, src_size[i])

    cdef bytes py_string
    if os.getenv("CLPY_SAVE_CL_KERNEL_SOURCE") == "1":
        for i in range(length):
            with open(tempfile.gettempdir() + "/" +
                      str(time.monotonic()) + ".cl", 'w') as f:
                py_string = sources[i]
                f.write(py_string.decode('utf-8'))

    program = api.CreateProgramWithSource(context=context, count=length,
                                          strings=src, lengths=src_size)
    options = options + b'\0'
    cdef char* options_cstr = options

    from exceptions import OpenCLProgramBuildError
    from exceptions import OpenCLRuntimeError
    try:
        api.BuildProgram(program, num_devices, devices_ptrs, options_cstr,
                         <void*>NULL, <void*>NULL)
    except OpenCLRuntimeError as err:
        if err.status == CL_BUILD_PROGRAM_FAILURE:
            devices = GetProgramDevice(program)
            log = str()
            for id in devices:
                l = GetProgramBuildLog(program, env.get_devices()[id])
                log += "Device#{0}: {1}".format(id, l)
            err = OpenCLProgramBuildError(err, log)
        raise err

    return program

cdef str GetProgramBuildLog(cl_program program, cl_device_id device):
    cdef size_t length
    cdef cl_int status = api.clGetProgramBuildInfo(
        program,
        device,
        CL_PROGRAM_BUILD_LOG,
        0,
        NULL,
        &length)
    check_status(status)

    cdef array.array info = array.array('b')
    array.resize(info, length)
    status = api.clGetProgramBuildInfo(
        program,
        device,
        CL_PROGRAM_BUILD_LOG,
        length,
        info.data.as_voidptr,
        NULL)
    check_status(status)
    return info.tobytes().decode('utf8')

__typesof_size = ['ulong'] * env.num_devices
for id in range(env.num_devices):
    host_size_t_bits = cython.sizeof(Py_ssize_t)*8
    device_address_bits = GetDeviceAddressBits(
        env.get_devices()[id])
    if host_size_t_bits != device_address_bits:
        raise "Host's size_t is different from device's size_t."

    if device_address_bits == 32:
        __typesof_size[id] = 'uint'
    elif device_address_bits != 64:
        raise "There is no type of size_t."


def typeof_size():
    return __typesof_size[env.get_device_id()]


cpdef size_t eventRecord() except *:
    cdef cl_event event
    api.EnqueueMarker(
        env.get_command_queue(),
        &event)
    api.WaitForEvents(1, &event)
    cpdef size_t time
    api.GetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(time),
        &time,
        NULL)
    return time

cpdef void eventSynchronize() except *:
    api.Finish(env.get_command_queue())
