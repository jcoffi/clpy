include "common_decl.pxi"

include "func_decl.pxi"

###############################################################################
# thin wrappers
cdef cl_uint GetPlatformIDs(size_t num_entries,
                            cl_platform_id* platforms) except *
cdef void GetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) except *
cdef cl_uint GetDeviceIDs(
    cl_platform_id platform,
    size_t device_type,
    size_t num_entries,
    cl_device_id* devices) except *
cdef void GetDeviceInfo(
    cl_device_id device,
    cl_platform_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) except *
cdef cl_context CreateContext(
    cl_context_properties* properties,
    size_t num_devices,
    cl_device_id* devices,
    void* pfn_notify,
    void* user_data)
cdef cl_command_queue CreateCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties)
cdef cl_mem CreateBuffer(
    cl_context context,
    size_t flags,
    size_t size,
    void* host_ptr)
cdef cl_program CreateProgramWithSource(
    cl_context context,
    cl_uint count,
    char** strings,
    size_t* lengths)
cdef void GetProgramInfo(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) except *
cdef void BuildProgram(
    cl_program program,
    cl_uint num_devices,
    cl_device_id* device_list,
    char* options,
    void* pfn_notify,
    void* user_data) except *
cdef cl_kernel CreateKernel(cl_program program, char* kernel_name)
cdef void SetKernelArg(cl_kernel kernel,
                       arg_index,
                       arg_size,
                       void* arg_value) except *
cdef void EnqueueTask(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list=*,
    cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void EnqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel kernel,
    size_t work_dim,
    size_t* global_work_offset,
    size_t* global_work_size,
    size_t* local_work_size,
    cl_uint num_events_in_wait_list=*,
    cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void EnqueueReadBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    blocking_read,
    size_t offset,
    size_t cb,
    void* host_ptr,
    cl_uint num_events_in_wait_list=*,
    cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void EnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    blocking_write,
    size_t offset,
    size_t cb,
    void* host_ptr,
    cl_uint num_events_in_wait_list=*,
    cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void EnqueueCopyBuffer(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t cb,
    cl_uint num_events_in_wait_list=*,
    cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void EnqueueFillBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list=*,
    cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void EnqueueMarker(
    cl_command_queue command_queue,
    cl_event* event) except *
cdef void EnqueueBarrierWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list=*,
    const cl_event* event_wait_list=*,
    cl_event* event=*) except *
cdef void GetEventProfilingInfo(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) except *
cdef void Flush(cl_command_queue command_queue) except *
cdef void Finish(cl_command_queue command_queue) except *
cdef void ReleaseKernel(cl_kernel kernel) except *
cdef void ReleaseProgram(cl_program program) except *
cdef void ReleaseEvent(cl_event memobj) except *
cdef void ReleaseMemObject(cl_mem memobj) except *
cdef void ReleaseCommandQueue(cl_command_queue command_queue) except *
cdef void ReleaseContext(cl_context context) except *
cdef void WaitForEvents(size_t num_events, cl_event* event_list) except *
