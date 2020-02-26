import functools
import operator
cimport function

cimport clpy.backend.opencl.env
cimport clpy.backend.opencl.utility

cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cache_dir=None,
        extra_source=None):
    options += (' -cl-fp32-correctly-rounded-divide-sqrt', )
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
