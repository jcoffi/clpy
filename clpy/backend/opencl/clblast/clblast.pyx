cimport clpy.backend.opencl.api as api
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
from clpy.backend.opencl.types cimport cl_command_queue
from clpy.backend.opencl.types cimport cl_event
from clpy.backend.opencl.types cimport cl_mem
from clpy.backend.opencl.exceptions import OpenCLRuntimeError

import numpy


def getCLBlastErrorName(statuscode):
    if statuscode in CLBLAST_STATUS_CODE:
        return CLBLAST_STATUS_CODE[statuscode]
    else:
        return "Unknown CLBlastStatusCode:" + str(statuscode)


class CLBlastRuntimeError(RuntimeError):
    def __init__(self, statuscode, detail=''):
        self.statuscode = statuscode
        name = getCLBlastErrorName(statuscode)
        super(CLBlastRuntimeError, self).__init__(
            '%s %s' % (name, detail))

cdef CLBlastLayout translate_layout(str_layout) except *:
    if (str_layout == 'R'):
        return CLBlastLayoutRowMajor
    elif (str_layout == 'C'):
        return CLBlastLayoutColMajor
    else:
        raise ValueError("layout should be \'R\' or \'C\'")

cdef CLBlastTranspose translate_transpose(str_transpose) except *:
    if str_transpose in ['N', 'n', 0]:
        return CLBlastTransposeNo
    elif str_transpose in ['T', 't', 1]:
        return CLBlastTransposeYes
    else:
        raise ValueError("transpose should be 'N'(0) or 'T'(1)")

cdef CLBlastSide translate_side(str_side) except *:
    if str_side in ['L', 'l']:
        return CLBlastSideLeft
    elif str_side in ['R', 'r']:
        return CLBlastSideRight
    else:
        raise ValueError("side should be \'L\' or \'R\'")

cdef CLBlastDiagonal translate_diagonal(str_diagonal) except *:
    if str_diagonal in ['U', 'u']:
        return CLBlastDiagonalUnit
    elif str_diagonal in ['N', 'n']:
        return CLBlastDiagonalNonUnit
    else:
        raise ValueError("diagonal should be 'U' or 'N'")

cdef CLBlastTriangle translate_triangle(str_triangle) except *:
    if str_triangle in ['U', 'u']:
        return CLBlastTriangleUpper
    elif str_triangle in ['L', 'l']:
        return CLBlastTriangleLower
    else:
        raise ValueError("triangle should be 'U' or 'L'")

cdef void clblast_sgemm(
        CLBlastLayout layout,
        CLBlastTranspose a_transpose,
        CLBlastTranspose b_transpose,
        size_t m, size_t n, size_t k,
        float alpha,
        cl_mem a_buffer, size_t a_offset, size_t a_ld,
        cl_mem b_buffer, size_t b_offset, size_t b_ld,
        float beta,
        cl_mem c_buffer, size_t c_offset, size_t c_ld) except *:
    cdef cl_command_queue\
        command_queue=clpy.backend.opencl.env.get_command_queue()

    cdef CLBlastStatusCode status = CLBlastSgemm(
        layout, a_transpose, b_transpose,
        m, n, k,
        alpha,
        a_buffer, a_offset, a_ld,
        b_buffer, b_offset, b_ld,
        beta,
        c_buffer, c_offset, c_ld,
        &command_queue,
        <cl_event*>NULL)
    if (status != CLBlastSuccess):
        raise CLBlastRuntimeError(statuscode=status)
    return

cpdef sgemm(str_layout, transa, transb,
            m, n, k, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc):
    cdef CLBlastLayout layout = translate_layout(str_layout)
    cdef CLBlastTranspose a_transpose = translate_transpose(transa)
    cdef CLBlastTranspose b_transpose = translate_transpose(transb)

    cdef size_t a_buffer = A.data.buf.get()
    cdef size_t b_buffer = B.data.buf.get()
    cdef size_t c_buffer = C.data.buf.get()

    clblast_sgemm(
        layout, a_transpose, b_transpose,
        m, n, k, alpha,
        <cl_mem>a_buffer, A.data.cl_mem_offset() // A.itemsize, lda,
        <cl_mem>b_buffer, B.data.cl_mem_offset() // B.itemsize, ldb,
        beta,
        <cl_mem>c_buffer, C.data.cl_mem_offset() // C.itemsize, ldc)


cdef void clblast_dgemm(
        CLBlastLayout layout,
        CLBlastTranspose a_transpose,
        CLBlastTranspose b_transpose,
        size_t m, size_t n, size_t k,
        double alpha,
        cl_mem a_buffer, size_t a_offset, size_t a_ld,
        cl_mem b_buffer, size_t b_offset, size_t b_ld,
        double beta,
        cl_mem c_buffer, size_t c_offset, size_t c_ld) except *:
    cdef cl_command_queue\
        command_queue=clpy.backend.opencl.env.get_command_queue()

    cdef CLBlastStatusCode status = CLBlastDgemm(
        layout, a_transpose, b_transpose,
        m, n, k,
        alpha,
        a_buffer, a_offset, a_ld,
        b_buffer, b_offset, b_ld,
        beta,
        c_buffer, c_offset, c_ld,
        &command_queue,
        <cl_event*>NULL)
    if (status != CLBlastSuccess):
        raise CLBlastRuntimeError(statuscode=status)
    return

cpdef dgemm(str_layout, transa, transb,
            m, n, k, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc):
    cdef CLBlastLayout layout = translate_layout(str_layout)
    cdef CLBlastTranspose a_transpose = translate_transpose(transa)
    cdef CLBlastTranspose b_transpose = translate_transpose(transb)

    cdef size_t a_buffer = A.data.buf.get()
    cdef size_t b_buffer = B.data.buf.get()
    cdef size_t c_buffer = C.data.buf.get()

    clblast_dgemm(
        layout, a_transpose, b_transpose,
        m, n, k, alpha,
        <cl_mem>a_buffer, A.data.cl_mem_offset() // A.itemsize, lda,
        <cl_mem>b_buffer, B.data.cl_mem_offset() // B.itemsize, ldb,
        beta,
        <cl_mem>c_buffer, C.data.cl_mem_offset() // C.itemsize, ldc)


cdef void clblast_strsm(
        CLBlastLayout layout,
        CLBlastSide side,
        CLBlastTriangle triangle,
        CLBlastTranspose a_transpose,
        CLBlastDiagonal diagonal,
        size_t m, size_t n, float alpha,
        cl_mem a_buffer, size_t a_offset, size_t a_ld,
        cl_mem b_buffer, size_t b_offset, size_t b_ld) except *:
    cdef cl_command_queue\
        command_queue=clpy.backend.opencl.env.get_command_queue()

    cdef CLBlastStatusCode status = CLBlastStrsm(
        layout,
        side,
        triangle,
        a_transpose,
        diagonal,
        m, n, alpha,
        a_buffer, a_offset, a_ld,
        b_buffer, b_offset, b_ld,
        &command_queue,
        <cl_event*>NULL)
    if (status != CLBlastSuccess):
        raise CLBlastRuntimeError(statuscode=status)
    return

cpdef strsm(str_layout,
            str_side,
            str_triangle,
            str_a_transpose,
            str_diagonal,
            m, n, alpha,
            A, lda,
            B, ldb):
    cdef CLBlastLayout layout = translate_layout(str_layout)
    cdef CLBlastSide side = translate_side(str_side)
    cdef CLBlastTriangle triangle = translate_triangle(str_triangle)
    cdef CLBlastTranspose a_transpose = translate_transpose(str_a_transpose)
    cdef CLBlastDiagonal diagonal = translate_diagonal(str_diagonal)

    cdef size_t a_buffer = A.data.buf.get()
    cdef size_t b_buffer = B.data.buf.get()

    clblast_strsm(
        layout, side, triangle, a_transpose, diagonal,
        m, n, alpha,
        <cl_mem>a_buffer, A.data.cl_mem_offset() // A.itemsize, lda,
        <cl_mem>b_buffer, B.data.cl_mem_offset() // B.itemsize, ldb)

cdef void clblast_sgemm_batched(
        CLBlastLayout layout,
        CLBlastTranspose a_transpose,
        CLBlastTranspose b_transpose,
        size_t m,
        size_t n,
        size_t k,
        float *alphas,
        cl_mem a_buffer,
        size_t *a_offsets,
        size_t a_ld,
        cl_mem b_buffer,
        size_t *b_offsets,
        size_t b_ld,
        float *betas,
        cl_mem c_buffer,
        size_t *c_offsets,
        size_t c_ld,
        size_t batch_count) except *:
    cdef cl_command_queue\
        command_queue=clpy.backend.opencl.env.get_command_queue()

    cdef CLBlastStatusCode status = CLBlastSgemmBatched(
        layout, a_transpose, b_transpose,
        m, n, k,
        alphas,
        a_buffer, a_offsets, a_ld,
        b_buffer, b_offsets, b_ld,
        betas,
        c_buffer, c_offsets, c_ld,
        batch_count,
        &command_queue,
        <cl_event*>NULL)
    if (status != CLBlastSuccess):
        raise CLBlastRuntimeError(statuscode=status)
    return

cpdef sgemm_batched(str_layout, transa, transb,
                    m, n, k,
                    alpha,
                    A, offsets_a, lda,
                    B, offsets_b, ldb,
                    beta,
                    C, offsets_c, ldc,
                    batch_count):
    cdef CLBlastLayout layout = translate_layout(str_layout)
    cdef CLBlastTranspose a_transpose = translate_transpose(transa)
    cdef CLBlastTranspose b_transpose = translate_transpose(transb)

    cdef size_t a_buffer = A.data.buf.get()
    cdef size_t offsets_a_ptr = offsets_a.ctypes.data
    cdef size_t b_buffer = B.data.buf.get()
    cdef size_t offsets_b_ptr = offsets_b.ctypes.data
    cdef size_t c_buffer = C.data.buf.get()
    cdef size_t offsets_c_ptr = offsets_c.ctypes.data

    np_alphas = numpy.full((batch_count,), alpha, dtype='float32')
    cdef size_t alphas = np_alphas.ctypes.data
    np_betas = numpy.full((batch_count,), beta, dtype='float32')
    cdef size_t betas = np_betas.ctypes.data

    clblast_sgemm_batched(
        layout, a_transpose, b_transpose,
        m, n, k, <float*>alphas,
        <cl_mem>a_buffer, <size_t*>offsets_a_ptr, lda,
        <cl_mem>b_buffer, <size_t*>offsets_b_ptr, ldb,
        <float*>betas,
        <cl_mem>c_buffer, <size_t*>offsets_c_ptr, ldc,
        batch_count)

cdef void clblast_dgemm_batched(
        CLBlastLayout layout,
        CLBlastTranspose a_transpose,
        CLBlastTranspose b_transpose,
        size_t m,
        size_t n,
        size_t k,
        double *alphas,
        cl_mem a_buffer,
        size_t *a_offsets,
        size_t a_ld,
        cl_mem b_buffer,
        size_t *b_offsets,
        size_t b_ld,
        double *betas,
        cl_mem c_buffer,
        size_t *c_offsets,
        size_t c_ld,
        size_t batch_count) except *:
    cdef cl_command_queue\
        command_queue=clpy.backend.opencl.env.get_command_queue()

    cdef CLBlastStatusCode status = CLBlastDgemmBatched(
        layout, a_transpose, b_transpose,
        m, n, k,
        alphas,
        a_buffer, a_offsets, a_ld,
        b_buffer, b_offsets, b_ld,
        betas,
        c_buffer, c_offsets, c_ld,
        batch_count,
        &command_queue,
        <cl_event*>NULL)
    if (status != CLBlastSuccess):
        raise CLBlastRuntimeError(statuscode=status)
    return

cpdef dgemm_batched(str_layout, transa, transb,
                    m, n, k,
                    alpha,
                    A, offsets_a, lda,
                    B, offsets_b, ldb,
                    beta,
                    C, offsets_c, ldc,
                    batch_count):
    cdef CLBlastLayout layout = translate_layout(str_layout)
    cdef CLBlastTranspose a_transpose = translate_transpose(transa)
    cdef CLBlastTranspose b_transpose = translate_transpose(transb)

    cdef size_t a_buffer = A.data.buf.get()
    cdef size_t offsets_a_ptr = offsets_a.ctypes.data
    cdef size_t b_buffer = B.data.buf.get()
    cdef size_t offsets_b_ptr = offsets_b.ctypes.data
    cdef size_t c_buffer = C.data.buf.get()
    cdef size_t offsets_c_ptr = offsets_c.ctypes.data

    np_alphas = numpy.full((batch_count,), alpha, dtype='float64')
    cdef size_t alphas = np_alphas.ctypes.data
    np_betas = numpy.full((batch_count,), beta, dtype='float64')
    cdef size_t betas = np_betas.ctypes.data

    clblast_dgemm_batched(
        layout, a_transpose, b_transpose,
        m, n, k, <double*>alphas,
        <cl_mem>a_buffer, <size_t*>offsets_a_ptr, lda,
        <cl_mem>b_buffer, <size_t*>offsets_b_ptr, ldb,
        <double*>betas,
        <cl_mem>c_buffer, <size_t*>offsets_c_ptr, ldc,
        batch_count)

CLBLAST_STATUS_CODE = {
    CLBlastSuccess: "CLBlastSuccess",
    CLBlastOpenCLCompilerNotAvailable: "CLBlastOpenCLCompilerNotAvailable",
    CLBlastTempBufferAllocFailure: "CLBlastTempBufferAllocFailure",
    CLBlastOpenCLOutOfResources: "CLBlastOpenCLOutOfResources",
    CLBlastOpenCLOutOfHostMemory: "CLBlastOpenCLOutOfHostMemory",
    CLBlastOpenCLBuildProgramFailure: "CLBlastOpenCLBuildProgramFailure",
    CLBlastInvalidValue: "CLBlastInvalidValue",
    CLBlastInvalidCommandQueue: "CLBlastInvalidCommandQueue",
    CLBlastInvalidMemObject: "CLBlastInvalidMemObject",
    CLBlastInvalidBinary: "CLBlastInvalidBinary",
    CLBlastInvalidBuildOptions: "CLBlastInvalidBuildOptions",
    CLBlastInvalidProgram: "CLBlastInvalidProgram",
    CLBlastInvalidProgramExecutable: "CLBlastInvalidProgramExecutable",
    CLBlastInvalidKernelName: "CLBlastInvalidKernelName",
    CLBlastInvalidKernelDefinition: "CLBlastInvalidKernelDefinition",
    CLBlastInvalidKernel: "CLBlastInvalidKernel",
    CLBlastInvalidArgIndex: "CLBlastInvalidArgIndex",
    CLBlastInvalidArgValue: "CLBlastInvalidArgValue",
    CLBlastInvalidArgSize: "CLBlastInvalidArgSize",
    CLBlastInvalidKernelArgs: "CLBlastInvalidKernelArgs",
    CLBlastInvalidLocalNumDimensions: "CLBlastInvalidLocalNumDimensions",
    CLBlastInvalidLocalThreadsTotal: "CLBlastInvalidLocalThreadsTotal",
    CLBlastInvalidLocalThreadsDim: "CLBlastInvalidLocalThreadsDim",
    CLBlastInvalidGlobalOffset: "CLBlastInvalidGlobalOffset",
    CLBlastInvalidEventWaitList: "CLBlastInvalidEventWaitList",
    CLBlastInvalidEvent: "CLBlastInvalidEvent",
    CLBlastInvalidOperation: "CLBlastInvalidOperation",
    CLBlastInvalidBufferSize: "CLBlastInvalidBufferSize",
    CLBlastInvalidGlobalWorkSize: "CLBlastInvalidGlobalWorkSize",
    CLBlastNotImplemented: "CLBlastNotImplemented",
    CLBlastInvalidMatrixA: "CLBlastInvalidMatrixA",
    CLBlastInvalidMatrixB: "CLBlastInvalidMatrixB",
    CLBlastInvalidMatrixC: "CLBlastInvalidMatrixC",
    CLBlastInvalidVectorX: "CLBlastInvalidVectorX",
    CLBlastInvalidVectorY: "CLBlastInvalidVectorY",
    CLBlastInvalidDimension: "CLBlastInvalidDimension",
    CLBlastInvalidLeadDimA: "CLBlastInvalidLeadDimA",
    CLBlastInvalidLeadDimB: "CLBlastInvalidLeadDimB",
    CLBlastInvalidLeadDimC: "CLBlastInvalidLeadDimC",
    CLBlastInvalidIncrementX: "CLBlastInvalidIncrementX",
    CLBlastInvalidIncrementY: "CLBlastInvalidIncrementY",
    CLBlastInsufficientMemoryA: "CLBlastInsufficientMemoryA",
    CLBlastInsufficientMemoryB: "CLBlastInsufficientMemoryB",
    CLBlastInsufficientMemoryC: "CLBlastInsufficientMemoryC",
    CLBlastInsufficientMemoryX: "CLBlastInsufficientMemoryX",
    CLBlastInsufficientMemoryY: "CLBlastInsufficientMemoryY",
    CLBlastInsufficientMemoryTemp: "CLBlastInsufficientMemoryTemp",
    CLBlastInvalidBatchCount: "CLBlastInvalidBatchCount",
    CLBlastInvalidOverrideKernel: "CLBlastInvalidOverrideKernel",
    CLBlastMissingOverrideParameter: "CLBlastMissingOverrideParameter",
    CLBlastInvalidLocalMemUsage: "CLBlastInvalidLocalMemUsage",
    CLBlastNoHalfPrecision: "CLBlastNoHalfPrecision",
    CLBlastNoDoublePrecision: "CLBlastNoDoublePrecision",
    CLBlastInvalidVectorScalar: "CLBlastInvalidVectorScalar",
    CLBlastInsufficientMemoryScalar: "CLBlastInsufficientMemoryScalar",
    CLBlastDatabaseError: "CLBlastDatabaseError",
    CLBlastUnknownError: "CLBlastUnknownError",
    CLBlastUnexpectedError: "CLBlastUnexpectedError",
}
