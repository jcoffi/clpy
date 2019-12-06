# cython: profile=True
import clpy
from clpy.core.core cimport ndarray
import numpy

import math

xorwow_src = """
ulong xorwow(
    __global ulong* a,
    __global ulong* b,
    __global ulong* c,
    __global ulong* d,
    __global ulong* counter
){

    ulong t = *d;
    ulong const s = *a;
    *d = *c;
    *c = *b;
    *b = s;

    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ ( s << 4 );
    *a = t;

    *counter += 362437;

    return t + *counter;
}

"""

expand_kernel_src = xorwow_src + """
__kernel void clpy_expand_inner_state_array(
    CArray<ulong, 1> a,
    CArray<ulong, 1> b,
    CArray<ulong, 1> c,
    CArray<ulong, 1> d,
    CArray<ulong, 1> counter,
    ulong stride
){
    size_t const id = get_global_id(0);

    size_t const base = id * stride * 2;
    size_t const target = base + stride;

    counter[target] =
        xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    a[target] =
        xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    c[target] =
        xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    b[target] =
        xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    d[target] =
        xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
}
"""
expand_function = None


roll_kernel_src = xorwow_src + """
__kernel void clpy_rng_roll(
    CArray<ulong, 1> a,
    CArray<ulong, 1> b,
    CArray<ulong, 1> c,
    CArray<ulong, 1> d,
    CArray<ulong, 1> counter,
    CArray<ulong, 1> output
){
    size_t const id = get_global_id(0);
    output[id] = xorwow( &a[id], &b[id], &c[id], &d[id], &counter[id] );
}
"""
roll_function = None

cdef class clrandGenerator:

    def _issue_by_np(self):
        # note(nsakabe-fixstars):
        # numpy.random.random_integers accepts a number
        # up to the maximum bound of *signed* int64.
        return clpy.asarray(
            [numpy.random.randint(numpy.iinfo(numpy.int64).max)],
            dtype=numpy.uint64
        )

    def __init__(self):
        global expand_function
        global roll_function
        if expand_function is None:
            expand_module =\
                clpy.core.core.compile_with_cache(expand_kernel_src)
            expand_function =\
                expand_module.get_function("clpy_expand_inner_state_array")
        if roll_function is None:
            roll_module = clpy.core.core.compile_with_cache(roll_kernel_src)
            roll_function = roll_module.get_function("clpy_rng_roll")

        numpy.random.seed(0)
        self.a = self._issue_by_np()
        self.c = self._issue_by_np()
        self.b = self._issue_by_np()
        self.d = self._issue_by_np()
        self.counter = self._issue_by_np()
        self.inner_state_size = 1

    def seed(self, seed):
        # note(nsakabe-fixstars):
        # seed for numpy must be between 0 and 2**32-1.
        numpy.random.seed(seed % (2**32-1))
        self.a = self._issue_by_np()
        self.c = self._issue_by_np()
        self.b = self._issue_by_np()
        self.d = self._issue_by_np()
        self.counter = self._issue_by_np()
        self.inner_state_size = 1

    def expand(self, size):
        if self.inner_state_size >= size:
            return
        old_inner_state_size = self.inner_state_size
        new_inner_state_size = 2 ** math.ceil(math.log2(size))  # 2べきに繰り上げ

        new_a = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_a[0] = self.a[0]
        new_b = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_b[0] = self.b[0]
        new_c = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_c[0] = self.c[0]
        new_d = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_d[0] = self.d[0]
        new_counter = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_counter[0] = self.counter[0]

        self.a = new_a
        self.b = new_b
        self.c = new_c
        self.d = new_d
        self.counter = new_counter

        stride = new_inner_state_size // 2
        while stride >= 1:
            expand_function(
                global_work_size=(new_inner_state_size // (stride*2),),
                local_work_size=(1,),
                args=(self.a, self.b, self.c, self.d, self.counter, stride),
                local_mem=0)
            stride = stride // 2

        self.inner_state_size = new_inner_state_size

    def roll(self):
        output = clpy.empty_like(self.a)
        roll_function(
            global_work_size=(output.size,),
            local_work_size=(1,),
            args=(self.a, self.b, self.c, self.d, self.counter, output),
            local_mem=0)
        return output

    def reveal(self):
        return self.a, self.b, self.c, self.d


cpdef clrandGenerator createGenerator():
    return clrandGenerator()

cpdef setPseudoRandomGeneratorSeed(
    clrandGenerator generator, unsigned long long seed
):
    generator.seed(seed)


def is_acceptable_int(dtype):
    return dtype.char in 'qlihbQLIHB'


def safe_cast_to_ints(array, dtype):
    if not is_acceptable_int(dtype):
        raise ValueError("array's type must be integer")
    if array.dtype == dtype:
        return array
    mask = (1 << dtype.itemsize*8) - 1
    mask = clpy.array(mask, dtype=numpy.uint64)

    masked_array = array & mask

    return masked_array.astype(dtype)


cpdef generate(clrandGenerator generator, ndarray array):
    if not is_acceptable_int(array.dtype):
        raise ValueError("array's type must be integer")
    generator.expand(array.size)
    state = generator.roll()
    array[:] = safe_cast_to_ints(
        state[0:array.size].reshape(array.shape), dtype=array.dtype
    )

u64_shrinkto_fp = clpy.core.core.ElementwiseKernel(
    '', 'T in, U out',
    '''
    out = (U)(in)/(U)(0xFFFFFFFFFFFFFFFF);
    ''',
    'clpy_u64_shrinkto_fp'
)

cpdef generateUniform(clrandGenerator generator, ndarray array):
    if array.dtype.name != "float32":
        raise ValueError("array's type must be float32")
    size = array.size
    generator.expand(size)
    state = generator.roll()
    state_view = state[0:size].reshape(array.shape)
    u64_shrinkto_fp(state_view, array)

cpdef generateUniformDouble(clrandGenerator generator, ndarray array):
    if array.dtype.name != "float64":
        raise ValueError("array's type must be float64")
    size = array.size
    generator.expand(size)
    state = generator.roll()
    state_view = state[0:size].reshape(array.shape)
    u64_shrinkto_fp(state_view, array)

box_muller = clpy.core.core.ElementwiseKernel(
    '', 'T u1, T u2, U out',
    '''
    out = sqrt( -2.0 * log ( u1 ) ) * cos ( 2.0 * M_PI * u2 );
    ''',
    'clpy_box_muller'
)

cpdef generateNormal(
    clrandGenerator generator, ndarray array, float loc, float scale
):
    # Box-Muller method
    if array.dtype.name != "float32":
        raise ValueError("array's type must be float32")
    size = array.size
    generator.expand(size)
    u1 = clpy.empty_like(array)
    generateUniform(generator, u1)
    u2 = clpy.empty_like(array)
    generateUniform(generator, u2)
    box_muller(u1, u2, array)
    array += loc
    array *= scale

cpdef generateNormalDouble(
    clrandGenerator generator, ndarray array, float loc, float scale
):
    # Box-Muller method
    if array.dtype.name != "float64":
        raise ValueError("array's type must be float64")
    size = array.size
    generator.expand(size)
    u1 = clpy.empty_like(array)
    generateUniformDouble(generator, u1)
    u2 = clpy.empty_like(array)
    generateUniformDouble(generator, u2)
    box_muller(u1, u2, array)
    array += loc
    array *= scale


CLPY_RNG_PSEUDO_DEFAULT = 0
CLPY_RNG_XORWOW = 1
