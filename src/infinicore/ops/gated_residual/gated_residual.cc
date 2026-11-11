#include "infinicore/ops/gated_residual.hpp"
#include <climits>
#include <stdexcept>
#include <array>
#include <limits>

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GatedResidual);

namespace {
std::pair<uintptr_t, uintptr_t> bounds(const Tensor& t) {
    const auto n = t->ndim();
    const uint64_t span = uint64_t(t->size(n-2)-1)*t->stride(n-2)
                        + uint64_t(t->size(n-1)-1)*t->stride(n-1);
    if (span > uint64_t(INT64_MAX)/4-1) throw std::invalid_argument("gated_residual: span overflow");
    auto begin = reinterpret_cast<uintptr_t>(t->data());
    const uint64_t bytes = (span+1)*t->element_size();
    if (!begin || bytes > std::numeric_limits<uintptr_t>::max()-begin)
        throw std::invalid_argument("gated_residual: invalid address span");
    return {begin, begin+bytes};
}
bool overlaps(const Tensor& a, const Tensor& b) {
    auto x=bounds(a), y=bounds(b);
    return x.first<y.second && y.first<x.second;
}
void validate(const Tensor& out, const Tensor& residual, const Tensor& gate, const Tensor& branch) {
    if (!out || !residual || !gate || !branch) throw std::invalid_argument("gated_residual: null tensor");
    auto dtype = residual->dtype();
    if (dtype!=DataType::F32 && dtype!=DataType::F16 && dtype!=DataType::BF16)
        throw std::invalid_argument("gated_residual: only FP32/FP16/BF16 supported");
    if (residual->device().getType()!=Device::Type::HYGON)
        throw std::invalid_argument("gated_residual: only HYGON supported");
    if (context::getDevice()!=residual->device())
        throw std::invalid_argument("gated_residual: current device mismatch");
    for (const Tensor* ptr : {&out,&residual,&gate,&branch}) {
        const auto& t=*ptr;
        auto nd=t->ndim();
        if ((nd!=2 && nd!=3) || (nd==3 && t->size(0)!=1) || t->shape()!=residual->shape()
            || t->dtype()!=dtype || t->device()!=residual->device())
            throw std::invalid_argument("gated_residual: expected same shape/dtype/device [N,H] or [1,N,H]");
        for (size_t i=0; i<nd; ++i)
            if (t->size(i)==0 || t->size(i)>INT32_MAX || t->stride(i)<0 || t->stride(i)>INT32_MAX)
                throw std::invalid_argument("gated_residual: invalid size or stride");
        bounds(t);
    }
    auto nd=out->ndim();
    if (out->stride(nd-1)<1 || (out->size(nd-2)>1 && uint64_t(out->stride(nd-2)) < out->size(nd-1)*out->stride(nd-1)))
        throw std::invalid_argument("gated_residual: overlapping output layout");
    const bool exact_residual = out->data()==residual->data() && out->strides()==residual->strides();
    if ((overlaps(out,residual) && !exact_residual) || overlaps(out,gate) || overlaps(out,branch))
        throw std::invalid_argument("gated_residual: unsupported output/input alias");
}
}

GatedResidual::GatedResidual(Tensor out, const Tensor& residual, const Tensor& gate, const Tensor& branch) {
    validate(out,residual,gate,branch);
    const auto type=out->device().getType();
    if (!plan_dispatcher().lookup(type) || !run_dispatcher().lookup(type) || !cleanup_dispatcher().lookup(type))
        throw std::runtime_error("gated_residual: HYGON backend registration missing");
    INFINICORE_GRAPH_OP_DISPATCH(type,out,residual,gate,branch);
}
void GatedResidual::execute(Tensor out, const Tensor& residual, const Tensor& gate, const Tensor& branch) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GatedResidual,out,residual,gate,branch);
}
Tensor gated_residual(const Tensor& residual, const Tensor& gate, const Tensor& branch) {
    if (!residual) throw std::invalid_argument("gated_residual: null residual");
    auto out=Tensor::empty(residual->shape(),residual->dtype(),residual->device());
    gated_residual_(out,residual,gate,branch);
    return out;
}
void gated_residual_(Tensor out, const Tensor& residual, const Tensor& gate, const Tensor& branch) {
    GatedResidual::execute(out,residual,gate,branch);
}
} // namespace infinicore::op
