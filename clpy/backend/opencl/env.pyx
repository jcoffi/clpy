# -*- coding: utf-8 -*-
import atexit
import locale
import logging
import numpy
import re

from clpy.backend.opencl import api
from clpy.backend.opencl cimport api
from clpy.backend.opencl import exceptions
from cython.view cimport array as cython_array

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.string cimport memset
from libc.string cimport memcpy

cdef interpret_versionstr(versionstr):
    version_detector = re.compile('''OpenCL (\d+)\.(\d+)''')
    match = version_detector.match(versionstr)
    if not match:
        raise RuntimeError("Invalid platform's OpenCL version string")
    major_version = int(match.group(1))
    minor_version = int(match.group(2))
    return (major_version, minor_version)

cdef void check_platform_version(
        cl_platform_id platform,
        required_version) except *:
    cdef size_t param_value_size
    api.GetPlatformInfo(
        platform,
        CL_PLATFORM_VERSION,
        0,
        NULL,
        &param_value_size)

    cdef cython_array versionstr_buffer =\
        cython_array(shape=(param_value_size,),
                     itemsize=sizeof(char),
                     format='b')
    api.GetPlatformInfo(
        platform,
        CL_PLATFORM_VERSION,
        param_value_size,
        <void*>versionstr_buffer.data,
        &param_value_size)

    versionstr =\
        versionstr_buffer.data[:param_value_size]\
        .decode(locale.getpreferredencoding())
    if not interpret_versionstr(versionstr) >= required_version:
        raise RuntimeError("Platform's OpenCL version must be >= 1.2")

cdef void check_device_version(cl_device_id device, required_version) except *:
    cdef size_t param_value_size
    api.GetDeviceInfo(
        device,
        CL_DEVICE_VERSION,
        0,
        NULL,
        &param_value_size)

    cdef cython_array versionstr_buffer =\
        cython_array(shape=(param_value_size,),
                     itemsize=sizeof(char),
                     format='b')
    api.GetDeviceInfo(
        device,
        CL_DEVICE_VERSION,
        param_value_size,
        <void*>versionstr_buffer.data,
        &param_value_size)

    versionstr =\
        versionstr_buffer.data[:param_value_size]\
        .decode(locale.getpreferredencoding())
    if not interpret_versionstr(versionstr) >= required_version:
        raise RuntimeError("Device's OpenCL version must be >= 1.2")


##########################################
# Initialization
##########################################

logging.info("Get num_platforms...", end='')
cdef cl_uint num_platforms = api.GetPlatformIDs(0, <cl_platform_id*>NULL)
logging.info("SUCCESS")
logging.info("%d platform(s) found" % num_platforms)

logging.info("Get the first platform...", end='')
cdef cl_platform_id[1] __platforms_ptr
num_platforms = api.GetPlatformIDs(1, &__platforms_ptr[0])
cdef cl_platform_id primary_platform = __platforms_ptr[0]
logging.info("SUCCESS")

check_platform_version(primary_platform, required_version=(1, 2))

logging.info("Get num_devices...", end='')
cdef cl_uint __num_devices = api.GetDeviceIDs(
    primary_platform,
    CL_DEVICE_TYPE_ALL,
    0,
    <cl_device_id*>NULL)
logging.info("SUCCESS")
logging.info("%d device(s) found" % __num_devices)

logging.info("Get all devices...", end='')
cdef cl_device_id* __devices = \
    <cl_device_id*>malloc(sizeof(cl_device_id)*__num_devices)
api.GetDeviceIDs(
    primary_platform,
    CL_DEVICE_TYPE_ALL,
    __num_devices,
    &__devices[0])
num_devices = __num_devices     # provide as pure python interface
logging.info("SUCCESS")

for id in range(__num_devices):
    check_device_version(__devices[id], required_version=(1, 2))

cdef int __current_device_id=0

logging.info("Create context...", end='')
cdef cl_context* __contexts = \
    <cl_context*>malloc(sizeof(cl_context)*__num_devices)
for id in range(__num_devices):
    __contexts[id] = api.CreateContext(
        properties=<cl_context_properties*>NULL,
        num_devices=1,
        devices=&__devices[id],
        pfn_notify=<void*>NULL,
        user_data=<void*>NULL)
logging.info("SUCCESS")

logging.info("Create command_queues...", end='')
cdef cl_command_queue* __command_queues = \
    <cl_command_queue*>malloc(sizeof(cl_command_queue)*__num_devices)
for id in range(__num_devices):
    __command_queues[id] = \
        api.CreateCommandQueue(__contexts[id], __devices[id], 0)
logging.info("SUCCESS")

cdef cl_char* __supports_cl_khr_fp16 = \
    <cl_char*>malloc(sizeof(cl_char)*__num_devices)
memset(__supports_cl_khr_fp16, -1, sizeof(cl_char)*__num_devices)

##########################################
# Functions
##########################################

cdef cl_context get_context():
    global __current_device_id
    return __contexts[__current_device_id]

cdef cl_command_queue get_command_queue():
    global __current_device_id
    return __command_queues[__current_device_id]

cpdef int get_device_id():
    global __current_device_id
    return __current_device_id

cpdef set_device_id(int id):
    global __current_device_id
    __current_device_id = id

cdef cl_device_id* get_devices():
    return &__devices[0]

cdef cl_device_id get_device():
    global __current_device_id
    return __devices[__current_device_id]

cdef cl_char check_cl_khr_fp16():
    code = b'''
        #pragma OPENCL EXTENSION cl_khr_fp16: enable
        __kernel void check_cl_khr_fp16(__global half* v){
          ushort x = 0x5140;
          *v = *(const half*)&x;
        }
        '''
    cdef size_t length
    cdef char* src
    cdef char* options
    cdef cl_mem buf=NULL
    cdef size_t global_work_size[3]
    cdef size_t ptr
    try:
        device = __devices[__current_device_id]
        context = __contexts[__current_device_id]
        queue = __command_queues[__current_device_id]

        length = len(code)
        src = <char*>malloc(sizeof(char)*length)
        memcpy(src, <char*>code, length)

        program = api.CreateProgramWithSource(
            context=context,
            count=<size_t>1,
            strings=&src,
            lengths=&length)
        options = b'-cl-fp32-correctly-rounded-divide-sqrt'
        api.BuildProgram(
            program,
            1,
            &device,
            options,
            <void*>NULL,
            <void*>NULL)
        kernel = api.CreateKernel(
            program, b'check_cl_khr_fp16')
        global_work_size[0] = 1
        global_work_size[1] = 0
        global_work_size[2] = 0
        v = numpy.array([0], dtype=numpy.float16)
        ptr = <size_t>v.ctypes.get_as_parameter().value
        buf = api.CreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            <size_t>v.nbytes,
            <void*>NULL)
        api.SetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &buf)
        api.EnqueueNDRangeKernel(
            command_queue=queue,
            kernel=kernel,
            work_dim=1,
            global_work_offset=<size_t*>NULL,
            global_work_size=&global_work_size[0],
            local_work_size=<size_t*>NULL)
        api.EnqueueReadBuffer(
            command_queue=queue,
            buffer=buf,
            blocking_read=api.BLOCKING,
            offset=0,
            cb=<size_t>v.nbytes,
            host_ptr=<void*>ptr)
        result = 1 if v[0] == numpy.float16(42) else 0
    except exceptions.OpenCLRuntimeError:
        result = 0
    if buf != NULL:
        api.ReleaseMemObject(buf)
    free(src)
    return result

cpdef cl_bool supports_cl_khr_fp16():
    global __current_device_id
    if __supports_cl_khr_fp16[__current_device_id] == -1:
        __supports_cl_khr_fp16[__current_device_id] = check_cl_khr_fp16()
    return CL_TRUE if __supports_cl_khr_fp16[__current_device_id] == 1 \
        else CL_FALSE


def release():
    """Release command_queue and context automatically."""
    logging.info("Flush/Finish/Release command queues...", end='')
    for id in range(__num_devices):
        api.Flush(__command_queues[id])
        api.Finish(__command_queues[id])
        api.ReleaseCommandQueue(__command_queues[id])
        api.ReleaseContext(__contexts[id])
    logging.info("SUCCESS")

    # Release kernels, programs here if needed.

atexit.register(release)
