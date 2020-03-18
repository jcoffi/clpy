from clpy.core.core cimport ndarray

cdef class clrandGenerator:
    cdef ndarray a
    cdef ndarray b
    cdef ndarray c
    cdef ndarray d
    cdef ndarray counter
    cdef int inner_state_size
    cpdef expand(self, int size)


cpdef clrandGenerator createGenerator()
cpdef setPseudoRandomGeneratorSeed(
    clrandGenerator generator, unsigned long long seed
)

cpdef generate(clrandGenerator generator, ndarray array)

cpdef generateUniform(clrandGenerator generator, ndarray array)
cpdef generateUniformDouble(clrandGenerator generator, ndarray array)

cpdef generateNormal(
    clrandGenerator generator, ndarray array, float loc, float scale
)
cpdef generateNormalDouble(
    clrandGenerator generator, ndarray array, float loc, float scale
)
