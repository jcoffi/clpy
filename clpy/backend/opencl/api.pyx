# cython: profile=True
cimport exceptions


###############################################################################
# thin wrappers
# function name: s/cl([A-Za-z]+)/$1/

cdef cl_uint GetPlatformIDs(size_t num_entries,
                            cl_platform_id* platforms) except *:
    cdef cl_uint num_platforms
    cdef cl_int status
    status = clGetPlatformIDs(
        <cl_uint>num_entries,
        platforms,
        &num_platforms)
    exceptions.check_status(status)
    return num_platforms

cdef void GetPlatformInfo(
        cl_platform_id platform,
        cl_platform_info param_name,
        size_t param_value_size,
        void *param_value,
        size_t *param_value_size_ret) except *:
    cdef cl_int status
    status = clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret)
    exceptions.check_status(status)

cdef cl_uint GetDeviceIDs(cl_platform_id platform,
                          size_t device_type,
                          size_t num_entries,
                          cl_device_id* devices) except *:
    cdef cl_uint num_devices
    cdef cl_int status
    status = clGetDeviceIDs(
        platform,
        <cl_device_type>device_type,
        <cl_uint>num_entries,
        devices,
        &num_devices)
    exceptions.check_status(status)
    return num_devices

cdef void GetDeviceInfo(
        cl_device_id device,
        cl_platform_info param_name,
        size_t param_value_size,
        void *param_value,
        size_t *param_value_size_ret) except *:
    cdef cl_int status
    status = clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret)
    exceptions.check_status(status)

cdef cl_context CreateContext(
        cl_context_properties* properties,
        size_t num_devices,
        cl_device_id* devices,
        void* pfn_notify,
        void* user_data):
    cdef cl_int status
    cdef cl_context context = clCreateContext(
        properties,
        <cl_uint>num_devices,
        <const cl_device_id*>devices,
        <void(*)(const char*, const void*, size_t, void*)>pfn_notify,
        <void*>user_data,
        &status)
    exceptions.check_status(status)
    return context

cdef cl_command_queue CreateCommandQueue(
        cl_context context,
        cl_device_id device,
        cl_command_queue_properties properties):

    cdef cl_int status
    cdef cl_command_queue command_queue = clCreateCommandQueue(
        context,
        device,
        properties,
        &status)
    exceptions.check_status(status)
    return command_queue

cdef cl_mem CreateBuffer(
        cl_context context,
        size_t flags,
        size_t size,
        void* host_ptr):

    cdef cl_int status
    cdef cl_mem memobj = clCreateBuffer(
        context,
        <cl_mem_flags>flags,
        <size_t>size,
        host_ptr,
        &status)
    exceptions.check_status(status)
    return memobj

cdef cl_program CreateProgramWithSource(
        cl_context context,
        cl_uint count,
        char** strings,
        size_t* lengths):

    cdef cl_int status
    cdef cl_program program = clCreateProgramWithSource(
        context,
        <cl_uint>count,
        <const char**>strings,
        <const size_t*>lengths,
        &status)
    exceptions.check_status(status)
    return program

cdef void GetProgramInfo(
        cl_program program,
        cl_program_info param_name,
        size_t param_value_size,
        void *param_value,
        size_t *param_value_size_ret) except *:

    cdef cl_int status
    status = clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret)
    exceptions.check_status(status)

cdef void BuildProgram(
        cl_program program,
        cl_uint num_devices,
        cl_device_id* device_list,
        char* options,
        void* pfn_notify,
        void* user_data) except *:

    cdef cl_int status = clBuildProgram(
        program,
        <cl_uint>num_devices,
        <const cl_device_id*>device_list,
        <const char*>options,
        <void(*)(cl_program, void*)>pfn_notify,
        <void*>user_data)
    exceptions.check_status(status)

cdef cl_kernel CreateKernel(cl_program program, char* kernel_name):
    cdef cl_int status
    cdef cl_kernel kernel = clCreateKernel(
        program,
        <const char*>kernel_name,
        &status)
    exceptions.check_status(status)
    return kernel

cdef void SetKernelArg(cl_kernel kernel, arg_index, arg_size,
                       void* arg_value) except *:
    cdef cl_int status = clSetKernelArg(
        kernel,
        <cl_uint>arg_index,
        <size_t>arg_size,
        <const void*>arg_value)
    exceptions.check_status(status)

cdef void EnqueueTask(
        cl_command_queue command_queue,
        cl_kernel kernel,
        cl_uint num_events_in_wait_list=0,
        cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:
    cdef cl_int status = clEnqueueTask(
        command_queue,
        kernel,
        <cl_uint>num_events_in_wait_list,
        <const cl_event*>event_wait_list,
        event)

cdef void EnqueueNDRangeKernel(
        cl_command_queue command_queue,
        cl_kernel kernel,
        size_t work_dim,
        size_t* global_work_offset,
        size_t* global_work_size,
        size_t* local_work_size,
        cl_uint num_events_in_wait_list=0,
        cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:

    cdef cl_int status = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        <cl_uint>work_dim,
        <const size_t *>global_work_offset,
        <const size_t *>global_work_size,
        <const size_t *>local_work_size,
        <cl_uint>num_events_in_wait_list,
        <const cl_event *>event_wait_list,
        event)
    exceptions.check_status(status)

cdef void EnqueueReadBuffer(
        cl_command_queue command_queue,
        cl_mem buffer,
        blocking_read,
        size_t offset,
        size_t cb,
        void* host_ptr,
        cl_uint num_events_in_wait_list=0,
        cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:
    cdef cl_int status = clEnqueueReadBuffer(
        command_queue,
        buffer,
        <cl_bool>blocking_read,
        <size_t>offset,
        <size_t>cb,
        host_ptr,
        <cl_uint>num_events_in_wait_list,
        <const cl_event *>event_wait_list,
        event)
    exceptions.check_status(status)

cdef void EnqueueWriteBuffer(
        cl_command_queue command_queue,
        cl_mem buffer,
        blocking_write,
        size_t offset,
        size_t cb,
        void* host_ptr,
        cl_uint num_events_in_wait_list=0,
        cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:
    cdef cl_int status = clEnqueueWriteBuffer(
        command_queue,
        buffer,
        <cl_bool>blocking_write,
        <size_t>offset,
        <size_t>cb,
        host_ptr,
        <cl_uint>num_events_in_wait_list,
        <const cl_event *>event_wait_list,
        event)
    exceptions.check_status(status)

cdef void EnqueueCopyBuffer(
        cl_command_queue command_queue,
        cl_mem src_buffer,
        cl_mem dst_buffer,
        size_t src_offset,
        size_t dst_offset,
        size_t cb,
        cl_uint num_events_in_wait_list=0,
        cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:
    cdef cl_int status = clEnqueueCopyBuffer(
        command_queue,
        src_buffer,
        dst_buffer,
        <size_t>src_offset,
        <size_t>dst_offset,
        <size_t>cb,
        <cl_uint>num_events_in_wait_list,
        <const cl_event *>event_wait_list,
        event)
    exceptions.check_status(status)

cdef void EnqueueFillBuffer(
        cl_command_queue command_queue,
        cl_mem buffer,
        void* pattern,
        size_t pattern_size,
        size_t offset,
        size_t size,
        cl_uint num_events_in_wait_list=0,
        cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:
    cdef cl_int status = clEnqueueFillBuffer(
        command_queue,
        buffer,
        pattern,
        pattern_size,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event)
    exceptions.check_status(status)

cdef void EnqueueMarker(
        cl_command_queue command_queue,
        cl_event* event) except *:
    cdef cl_int status = clEnqueueMarker(
        command_queue,
        event)
    exceptions.check_status(status)

cdef void EnqueueBarrierWithWaitList(
        cl_command_queue command_queue,
        cl_uint num_events_in_wait_list=0,
        const cl_event* event_wait_list=<cl_event*>NULL,
        cl_event* event=<cl_event*>NULL) except *:
    cdef cl_int status = clEnqueueBarrierWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event)
    exceptions.check_status(status)

cdef void GetEventProfilingInfo(
        cl_event event,
        cl_profiling_info param_name,
        size_t param_value_size,
        void* param_value,
        size_t* param_value_size_ret) except *:
    cdef cl_int status = clGetEventProfilingInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret)
    exceptions.check_status(status)

cdef void Flush(cl_command_queue command_queue) except *:
    exceptions.check_status(clFlush(command_queue))

cdef void Finish(cl_command_queue command_queue) except *:
    exceptions.check_status(clFinish(command_queue))

cdef void ReleaseKernel(cl_kernel kernel) except *:
    exceptions.check_status(clReleaseKernel(kernel))

cdef void ReleaseProgram(cl_program program) except *:
    exceptions.check_status(clReleaseProgram(program))

cdef void ReleaseEvent(cl_event memobj) except *:
    exceptions.check_status(clReleaseEvent(memobj))

cdef void ReleaseMemObject(cl_mem memobj) except *:
    exceptions.check_status(clReleaseMemObject(memobj))

cdef void ReleaseCommandQueue(cl_command_queue command_queue) except *:
    exceptions.check_status(clReleaseCommandQueue(command_queue))

cdef void ReleaseContext(cl_context context) except *:
    exceptions.check_status(clReleaseContext(context))

cdef void WaitForEvents(size_t num_events, cl_event* event_list) except *:
    exceptions.check_status(clWaitForEvents(<cl_uint>num_events, event_list))


TRUE = CL_TRUE
FALSE = CL_FALSE
BLOCKING = CL_BLOCKING
NON_BLOCKING = CL_NON_BLOCKING

MEM_READ_WRITE = CL_MEM_READ_WRITE

BUFFER_CREATE_TYPE_REGION = CL_BUFFER_CREATE_TYPE_REGION
