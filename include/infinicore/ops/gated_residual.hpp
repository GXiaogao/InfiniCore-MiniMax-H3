#pragma once
#include "common/op.hpp"
#include <cstdint>
#include <string>

namespace infinicore::op {
// residual + cast_dtype(gate * branch), preserving the separate mul/add rounding.
// Shapes [N,H] or [1,N,H], matching dtype/device; HYGON FP32/FP16/BF16.
class GatedResidual : public graph::DispatchableGraphOperator {
public:
    using plan_schema = void *(*)(Tensor, const Tensor &, const Tensor &, const Tensor &);
    static common::OpDispatcher<plan_schema> &plan_dispatcher();
    static common::OpDispatcher<run_schema> &run_dispatcher();
    static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher();
    GatedResidual(Tensor out, const Tensor &residual, const Tensor &gate, const Tensor &branch);
    static void execute(Tensor out, const Tensor &residual, const Tensor &gate, const Tensor &branch);
    // Host graph replay is allowed; device graph capture is deliberately disabled
    // until the host module lifecycle and launch-counter semantics are validated.
    bool is_device_graph_capture_safe() const override { return false; }
};
Tensor gated_residual(const Tensor &residual, const Tensor &gate, const Tensor &branch);
void gated_residual_(Tensor out, const Tensor &residual, const Tensor &gate, const Tensor &branch);
std::uint64_t gated_residual_hygon_launch_count();
std::string gated_residual_hygon_backend_info();
} // namespace infinicore::op
