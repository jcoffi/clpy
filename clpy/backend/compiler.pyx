import functools
import operator
cimport function

import clpy.backend.opencl.api
cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.env
cimport clpy.backend.opencl.utility

cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cache_dir=None,
        extra_source=None):
    options += (' -cl-fp32-correctly-rounded-divide-sqrt', )
    if clpy.backend.opencl.env.supports_cl_khr_fp16() == \
            clpy.backend.opencl.api.TRUE:
        options += (' -D__CLPY_ENABLE_CL_KHR_FP16', )
    optionStr = functools.reduce(operator.add, options)

    device = clpy.backend.opencl.env.get_device()
    program = clpy.backend.opencl.utility.CreateProgram(
        [source.encode('utf-8')],
        clpy.backend.opencl.env.get_context(),
        1,
        &device,
        optionStr.encode('utf-8'))
    cdef function.Module module = function.Module()
    module.set(program)
    return module
