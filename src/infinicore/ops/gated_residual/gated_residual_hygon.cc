#include "infinicore/ops/gated_residual.hpp"
#include <atomic>
#include <cstdlib>
#include <stdexcept>

#ifdef ENABLE_HYGON_API
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <memory>
#include <array>

namespace infinicore::op::gated_residual_impl {
struct TensorArg { void* pointer; int64_t rows,width,stride0,stride1; };
static_assert(sizeof(TensorArg)==40 && sizeof(void*)==8,"64-bit host ABI required");
std::atomic<uint64_t> launches{0};
template<class T> T resolve(void* lib, const char* name) {
    auto p=dlsym(lib,name);
    if (!p) throw std::runtime_error(std::string("gated_residual: missing symbol ")+name);
    return reinterpret_cast<T>(p);
}
std::string bundle_path() {
    auto value=std::getenv("INFINICORE_H3_GATED_BUNDLE");
    if (!value || !*value || !std::filesystem::path(value).is_absolute())
        throw std::runtime_error("Set INFINICORE_H3_GATED_BUNDLE to the verified absolute bundle path");
    return std::filesystem::canonical(value).string();
}
struct Api {
    std::string root,hip;
    void* library;
    void* (*open)(const char*,const char*,const char*);
    int (*launch)(void*,const TensorArg*,void*);
    const char* (*error)();
    std::mutex lock;
    std::map<std::pair<size_t,DataType>,void*> modules;
    explicit Api(const std::string& path):root(path),library(nullptr) {
        std::ifstream input(root+"/hip_library.txt");
        std::getline(input,hip);
        if (!input || !std::filesystem::path(hip).is_absolute() || !std::filesystem::is_regular_file(hip))
            throw std::runtime_error("gated_residual: invalid bundle HIP library");
        library=dlopen((root+"/libh3_gated_host.so").c_str(),RTLD_NOW|RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        try {
            open=resolve<decltype(open)>(library,"h3_open");
            launch=resolve<decltype(launch)>(library,"h3_launch");
            error=resolve<decltype(error)>(library,"h3_error");
        } catch (...) { dlclose(library); throw; }
    }
    void* module(const Tensor& t) {
        std::lock_guard<std::mutex> guard(lock);
        auto key=std::make_pair(t->device().getIndex(),t->dtype());
        auto found=modules.find(key);
        if(found!=modules.end()) return found->second;
        const char* dtype=t->dtype()==DataType::F32?"float32":t->dtype()==DataType::F16?"float16":"bfloat16";
        auto path=root+"/export_"+dtype+"/kernel.hsaco";
        auto name=std::string("h3_gated_residual_")+dtype;
        void* result=open(hip.c_str(),path.c_str(),name.c_str());
        if (!result) throw std::runtime_error(std::string("gated_residual load: ")+error());
        modules.emplace(key,result);
        return result;
    }
};
Api& api() {
    // Bounded process-lifetime cache: at most three modules/device. Never unload
    // code while asynchronous work or a recorded graph can still reference it.
    static Api* instance=new Api(bundle_path());
    if (instance->root!=bundle_path()) throw std::runtime_error("gated_residual: bundle changed within process");
    return *instance;
}
struct Plan {
    graph::GraphTensor out,residual,gate,branch;
    void* kernel;
    Api* api;
};
void* plan(Tensor out,const Tensor& residual,const Tensor& gate,const Tensor& branch) {
    auto& a=api();
    auto kernel=a.module(out);
    return new Plan{graph::GraphTensor(out),graph::GraphTensor(residual),graph::GraphTensor(gate),graph::GraphTensor(branch),kernel,&a};
}
void run(void* value) {
    auto& p=*static_cast<Plan*>(value);
    if (context::getDevice()!=p.out->device()) throw std::runtime_error("gated_residual: replay device mismatch");
    std::array<TensorArg,4> args;
    std::array<Tensor*,4> tensors{&p.residual,&p.gate,&p.branch,&p.out};
    for(size_t i=0;i<4;++i) {
        auto& t=*tensors[i];auto nd=t->ndim();
        args[i]={t->data(),int64_t(t->size(nd-2)),int64_t(t->size(nd-1)),t->stride(nd-2),t->stride(nd-1)};
    }
    // Serialize only the host adapter's small bookkeeping section. No device sync.
    std::lock_guard<std::mutex> guard(p.api->lock);
    if(p.api->launch(p.kernel,args.data(),context::getStream())!=0)
        throw std::runtime_error(std::string("gated_residual launch: ")+p.api->error());
    launches.fetch_add(1,std::memory_order_relaxed);
}
void cleanup(void** p) { delete static_cast<Plan*>(*p);*p=nullptr; }
static bool registered=[] {
    GatedResidual::plan_dispatcher().registerDevice(Device::Type::HYGON,&plan);
    GatedResidual::run_dispatcher().registerDevice(Device::Type::HYGON,&run);
    GatedResidual::cleanup_dispatcher().registerDevice(Device::Type::HYGON,&cleanup);
    return true;
}();
} // namespace infinicore::op::gated_residual_impl
#endif

namespace infinicore::op {
std::uint64_t gated_residual_hygon_launch_count() {
#ifdef ENABLE_HYGON_API
    return gated_residual_impl::launches.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
std::string gated_residual_hygon_backend_info() {
#ifdef ENABLE_HYGON_API
    return "NineToothed 0.26.0 / generic gfx906 HIP HSACO / host graph only; bundle="+gated_residual_impl::bundle_path();
#else
    return "HYGON backend was not built";
#endif
}
}
