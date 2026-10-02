#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char *source = 
"__kernel void test_half(__global const ushort *in, __global float *out) {\n"
"    int gid = get_global_id(0);\n"
"    // Test vload_half\n"
"    out[gid] = vload_half(gid, (const __global half *)in);\n"
"}\n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { printf("clGetPlatformIDs failed\n"); return 1; }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { printf("clGetDeviceIDs failed\n"); return 1; }

    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("Testing device: %s\n", name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &source, NULL, &err);
    err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build failed! err=%d\n", err);
        char log[4096];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build log:\n%s\n", log);
        return 1;
    }
    printf("SUCCESS! vload_half built cleanly on %s\n", name);
    clReleaseProgram(prog);
    clReleaseContext(ctx);
    return 0;
}
