#include "mex.h"
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;

    setenv("RUSTICL_ENABLE", "nouveau", 1);

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    mexPrintf("clGetPlatformIDs: found %u platform(s) (err: %d)\n", num_platforms, err);

    if (num_platforms == 0) {
        mexErrMsgIdAndTxt("OpenCL:NoPlatforms", "No OpenCL platforms discovered!");
        return;
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (cl_uint i = 0; i < num_platforms; i++) {
        char name[128], vendor[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        mexPrintf("Platform %u: %s (%s)\n", i, name, vendor);

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        mexPrintf("  Devices: %u\n", num_devices);

        if (num_devices > 0) {
            cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
            for (cl_uint j = 0; j < num_devices; j++) {
                char dev_name[128];
                cl_uint compute_units;
                cl_ulong global_mem;
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
                mexPrintf("  -> GPU Device %u: %s | SMX Cores: %u | VRAM: %lu MB\n", 
                          j, dev_name, compute_units, global_mem / (1024 * 1024));
            }
            free(devices);
        }
    }
    free(platforms);
}
