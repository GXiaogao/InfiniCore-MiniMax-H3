#pragma once
#include <pybind11/pybind11.h>
#include "infinicore/ops/gated_residual.hpp"
namespace infinicore::ops {
inline void bind_gated_residual(pybind11::module& m) {
    m.def("gated_residual",&op::gated_residual);
    m.def("gated_residual_",&op::gated_residual_);
    m.def("gated_residual_hygon_launch_count",&op::gated_residual_hygon_launch_count);
    m.def("gated_residual_hygon_backend_info",&op::gated_residual_hygon_backend_info);
}
}
