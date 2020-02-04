import clpy as cp


import os


def include_path():
    return os.path.join(cp.__path__[0], "..", "clpy", "core", "include")


@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, options=()):
    assert isinstance(options, tuple)
    kernel_code = cp.backend.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)


def read_code(code_filename, params):
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code


# TODO(shusukeueda):
# ClPy does not support cp.backend.Event (clpy/backend/stream.py)
def benchmark(func, args, n_run):
    times = []
    for _ in range(n_run):
        start = cp.backend.Event()
        end = cp.backend.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.backend.get_elapsed_time(start, end))  # milliseconds
    return times
