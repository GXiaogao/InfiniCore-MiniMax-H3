// Host-only HIP module adapter; all GPU computation comes from NineToothed HSACO.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(void*)==8 && sizeof(uint64_t)==8,"64-bit host ABI required");
static thread_local std::string error;
template<class T> T symbol(void* lib,const char* name) {
    auto p=dlsym(lib,name);
    if(!p) throw std::runtime_error(std::string("Missing HIP symbol: ")+name);
    return reinterpret_cast<T>(p);
}
struct Kernel {
    void* lib=nullptr;
    hipModule_t module=nullptr;
    hipFunction_t function=nullptr;
    decltype(&hipModuleLoad) load=nullptr;
    decltype(&hipModuleUnload) unload=nullptr;
    decltype(&hipModuleGetFunction) get_function=nullptr;
    decltype(&hipModuleLaunchKernel) launch=nullptr;
    decltype(&hipGetDevice) get_device=nullptr;
    decltype(&hipGetErrorString) error_string=nullptr;
    int device=-1;
    unsigned shared=0;
    uint64_t calls=0,last_stream=0;
    void check(hipError_t rc)const {
        if(rc!=hipSuccess) throw std::runtime_error(error_string(rc));
    }
    ~Kernel(){if(module&&unload)unload(module);if(lib)dlclose(lib);}
};
extern "C" const char* h3_attention_error(){return error.c_str();}
extern "C" void* h3_attention_open(const char* hip,const char* hsaco,const char* name,unsigned shared) {
    try {
        error.clear();
        if(!hip||!hsaco||!name||shared>65536) throw std::runtime_error("Invalid open arguments");
        auto k=std::make_unique<Kernel>();
        k->lib=dlopen(hip,RTLD_NOW|RTLD_LOCAL);
        if(!k->lib)throw std::runtime_error(dlerror());
        k->load=symbol<decltype(k->load)>(k->lib,"hipModuleLoad");
        k->unload=symbol<decltype(k->unload)>(k->lib,"hipModuleUnload");
        k->get_function=symbol<decltype(k->get_function)>(k->lib,"hipModuleGetFunction");
        k->launch=symbol<decltype(k->launch)>(k->lib,"hipModuleLaunchKernel");
        k->get_device=symbol<decltype(k->get_device)>(k->lib,"hipGetDevice");
        k->error_string=symbol<decltype(k->error_string)>(k->lib,"hipGetErrorString");
        k->check(k->get_device(&k->device));
        k->check(k->load(&k->module,hsaco));
        k->check(k->get_function(&k->function,k->module,name));
        k->shared=shared;
        return k.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" int h3_attention_launch(void* handle,const uint64_t* slots,unsigned count,unsigned grid,void* stream){
    try {
        error.clear();
        if(!handle||!slots||count!=44||grid==0||grid>0x7fffffffU||!stream)
            throw std::runtime_error("Invalid launch arguments");
        auto& k=*static_cast<Kernel*>(handle);
        int device=-1;k.check(k.get_device(&device));
        if(device!=k.device)throw std::runtime_error("HIP device changed after module load");
        // Each validated pointer/i64/fp64 argument occupies one aligned eight-byte slot.
        // This vendor3.5 HIP launcher appends two null scratch pointers as well.
        std::vector<uint64_t> storage(slots,slots+count);
        storage.push_back(0);storage.push_back(0);
        std::vector<void*> params;
        for(auto& slot:storage)params.push_back(&slot);
        k.check(k.launch(k.function,grid,1,1,256,1,1,k.shared,
                         reinterpret_cast<hipStream_t>(stream),params.data(),nullptr));
        ++k.calls;k.last_stream=reinterpret_cast<uintptr_t>(stream);
        return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" uint64_t h3_attention_calls(void* handle){return handle?static_cast<Kernel*>(handle)->calls:0;}
extern "C" uint64_t h3_attention_last_stream(void* handle){return handle?static_cast<Kernel*>(handle)->last_stream:0;}
// Caller synchronizes before close, including error cleanup.
extern "C" int h3_attention_close(void* handle){
    if(!handle)return 0;
    try {
        auto* k=static_cast<Kernel*>(handle);
        k->check(k->unload(k->module));k->module=nullptr;delete k;return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
