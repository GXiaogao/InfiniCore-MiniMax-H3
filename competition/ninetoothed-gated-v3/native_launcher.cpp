// Host-only HIP launch adapter. The GPU computation is NineToothed-generated HSACO.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdint>
#include <climits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

struct TensorArg {
    void* pointer;
    int64_t rows, width, stride0, stride1;
};
static_assert(sizeof(TensorArg) == 40 && sizeof(void*) == 8, "64-bit ABI required");
static thread_local std::string last_error;

template<class T> T symbol(void* lib, const char* name) {
    auto p = dlsym(lib, name);
    if (!p) throw std::runtime_error(std::string("Missing HIP symbol: ") + name);
    return reinterpret_cast<T>(p);
}

struct Kernel {
    void* lib = nullptr;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    decltype(&hipModuleLoad) load = nullptr;
    decltype(&hipModuleUnload) unload = nullptr;
    decltype(&hipModuleGetFunction) get_function = nullptr;
    decltype(&hipModuleLaunchKernel) launch = nullptr;
    decltype(&hipGetDevice) get_device = nullptr;
    decltype(&hipGetErrorString) error_string = nullptr;
    int device = -1;
    uint64_t calls = 0;
    uintptr_t last_stream = 0;
    void check(hipError_t rc) const {
        if (rc != hipSuccess) throw std::runtime_error(error_string(rc));
    }
    ~Kernel() {
        if (module && unload) unload(module);
        if (lib) dlclose(lib);
    }
};

extern "C" const char* h3_error() { return last_error.c_str(); }

extern "C" void* h3_open(const char* hip_path, const char* hsaco, const char* name) {
    try {
        last_error.clear();
        auto k = std::make_unique<Kernel>();
        k->lib = dlopen(hip_path, RTLD_NOW | RTLD_LOCAL);
        if (!k->lib) throw std::runtime_error(dlerror());
        k->load = symbol<decltype(k->load)>(k->lib, "hipModuleLoad");
        k->unload = symbol<decltype(k->unload)>(k->lib, "hipModuleUnload");
        k->get_function = symbol<decltype(k->get_function)>(k->lib, "hipModuleGetFunction");
        k->launch = symbol<decltype(k->launch)>(k->lib, "hipModuleLaunchKernel");
        k->get_device = symbol<decltype(k->get_device)>(k->lib, "hipGetDevice");
        k->error_string = symbol<decltype(k->error_string)>(k->lib, "hipGetErrorString");
        k->check(k->get_device(&k->device));
        k->check(k->load(&k->module, hsaco));
        k->check(k->get_function(&k->function, k->module, name));
        return k.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

extern "C" int h3_launch(void* handle, const TensorArg* tensors, void* stream) {
    try {
        last_error.clear();
        if (!handle || !tensors) throw std::runtime_error("Null launch arguments");
        auto& k = *static_cast<Kernel*>(handle);
        int device = -1;
        k.check(k.get_device(&device));
        if (device != k.device) throw std::runtime_error("HIP device changed since module load");
        const int64_t rows = tensors[0].rows, width = tensors[0].width;
        if (rows <= 0 || width <= 0 || rows > INT32_MAX || width > INT32_MAX)
            throw std::runtime_error("Unsupported tensor size");
        const uint64_t grid = uint64_t(rows) * ((uint64_t(width) + 127) / 128);
        if (grid > INT32_MAX) throw std::runtime_error("Grid is too large");
        // Local storage lives until the synchronous host launch has consumed its arguments.
        TensorArg args[4];
        void* params[22];
        for (int i = 0; i < 4; ++i) {
            args[i] = tensors[i];
            const auto& t = args[i];
            if (!t.pointer || t.rows != rows || t.width != width ||
                t.stride0 < 0 || t.stride1 < 0 || t.stride0 > INT32_MAX || t.stride1 > INT32_MAX)
                throw std::runtime_error("Invalid tensor pointer, shape or strides");
            const uint64_t span = uint64_t(rows-1)*t.stride0 + uint64_t(width-1)*t.stride1;
            if (span > uint64_t(INT64_MAX)/4) throw std::runtime_error("Tensor span overflow");
            params[5*i] = &args[i].pointer;
            params[5*i+1] = &args[i].rows;
            params[5*i+2] = &args[i].width;
            params[5*i+3] = &args[i].stride0;
            params[5*i+4] = &args[i].stride1;
        }
        if (args[3].stride1 < 1 || (rows > 1 && args[3].stride0 < width*args[3].stride1))
            throw std::runtime_error("Overlapping output layout is not supported");
        void* global_scratch = nullptr;
        void* profile_scratch = nullptr;
        params[20] = &global_scratch;
        params[21] = &profile_scratch;
        k.check(k.launch(k.function, static_cast<unsigned>(grid), 1, 1,
                         256, 1, 1, 0, reinterpret_cast<hipStream_t>(stream), params, nullptr));
        ++k.calls;
        k.last_stream = reinterpret_cast<uintptr_t>(stream);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

extern "C" uint64_t h3_calls(void* handle) { return static_cast<Kernel*>(handle)->calls; }
extern "C" uintptr_t h3_last_stream(void* handle) { return static_cast<Kernel*>(handle)->last_stream; }
// Caller must synchronize all outstanding work before closing a module.
extern "C" int h3_close(void* handle) {
    if (!handle) return 0;
    auto* k = static_cast<Kernel*>(handle);
    const auto rc = k->unload(k->module);
    if (rc != hipSuccess) { last_error = k->error_string(rc); return -1; }
    k->module = nullptr;
    delete k;
    return 0;
}
