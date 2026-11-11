#include "infinicore/ops/mha.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MultiheadAttention);

MultiheadAttention::MultiheadAttention(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       std::optional<Tensor> alibi_slopes,
                                       float scale,
                                       bool is_causal) {
    if (!out || !q || !k || !v) {
        throw std::invalid_argument("MHA expects non-null output, q, k and v");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v);
    const auto device_type = out->device().getType();
    if (!plan_dispatcher().lookup(device_type)
        || !run_dispatcher().lookup(device_type)
        || !cleanup_dispatcher().lookup(device_type)) {
        throw std::runtime_error("MHA backend registration is incomplete for device: "
                                 + out->device().toString());
    }
    context::setDevice(out->device());
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k, v, alibi_slopes, scale, is_causal);
}

void MultiheadAttention::execute(Tensor out,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 std::optional<Tensor> alibi_slopes,
                                 float scale,
                                 bool is_causal) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MultiheadAttention,
        out, q, k, v, alibi_slopes, scale, is_causal);
}

Tensor mha(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    std::optional<Tensor> alibi_slopes,
    float scale,
    bool is_causal) {
    if (!q) {
        throw std::invalid_argument("MHA expects non-null q");
    }
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_(out, q, k, v, alibi_slopes, scale, is_causal);
    return out;
}

void mha_(Tensor out,
          const Tensor &q,
          const Tensor &k,
          const Tensor &v,
          std::optional<Tensor> alibi_slopes,
          float scale,
          bool is_causal) {
    MultiheadAttention::execute(out, q, k, v, alibi_slopes, scale, is_causal);
}

} // namespace infinicore::op
