// Correctness fallback for Hygon builds without Flash Attention. Reuses only
// the repository's InfiniOp operators; no ATen, Python, or new GPU kernels.
#if defined(ENABLE_HYGON_API) && !defined(ENABLE_FLASH_ATTN)

#include "../../utils.hpp"
#include "infinicore/ops/mha.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace infinicore::op::mha_impl::hygon_infiniop {
namespace {

constexpr size_t QUERY_BLOCK = 128;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::invalid_argument(std::string("Hygon InfiniOp MHA: ") + message);
    }
}

// Return the conservative byte span, also rejecting overlapping/unsupported
// layouts and overflow before computing device pointer offsets.
size_t validate_layout(const Tensor &tensor) {
    require(tensor->ndim() == 4, "expected rank-4 BSND tensors");
    require(tensor->stride(3) == 1, "last-dimension stride must be 1");
    std::array<size_t, 4> dims{0, 1, 2, 3};
    for (auto dim : dims) {
        require(tensor->size(dim) > 0, "empty dimensions are unsupported");
        // InfiniCore unsqueeze inserts a size-1 dimension with stride 0.
        // It never advances the address; broadcast/negative strides remain unsupported.
        require(tensor->stride(dim) >= 0
                    && (tensor->size(dim) == 1 || tensor->stride(dim) > 0),
                "strides must be nonnegative and zero only for singleton dimensions");
    }
    std::sort(dims.begin(), dims.end(), [&](size_t a, size_t b) {
        return tensor->stride(a) < tensor->stride(b);
    });
    const auto limit = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    size_t span = 1;
    for (auto dim : dims) {
        if (tensor->size(dim) == 1) {
            continue;
        }
        const auto stride = static_cast<size_t>(tensor->stride(dim));
        require(stride >= span, "overlapping tensor layout is unsupported");
        const auto count = tensor->size(dim) - 1;
        require(count <= (limit - span) / stride, "tensor span overflows");
        span += count * stride;
    }
    require(span <= limit / tensor->element_size(), "tensor byte span overflows");
    return span * tensor->element_size();
}

Stride sequence_stride(const Tensor &tensor) {
    // A single sequence row may have stride 0, but BLAS still requires a valid
    // leading dimension. validate_layout has already ruled out broadcasting.
    return tensor->stride(1) == 0 ? static_cast<Stride>(tensor->size(3)) : tensor->stride(1);
}

void reject_output_alias(const Tensor &out, size_t out_span,
                         const Tensor &input, size_t input_span) {
    const auto a = reinterpret_cast<uintptr_t>(out->data());
    const auto b = reinterpret_cast<uintptr_t>(input->data());
    // Subtraction avoids overflow of address + span. Conservative: gaps inside
    // a strided tensor are included in its span.
    const bool overlaps = a <= b ? b - a < out_span : a - b < input_span;
    require(!overlaps, "output must not overlap q, k or v");
}

struct Chunk {
    size_t query_start = 0;
    infiniopGemmDescriptor_t qk = nullptr, pv = nullptr;
    infiniopSoftmaxDescriptor_t softmax = nullptr;
    infiniopCausalSoftmaxDescriptor_t causal = nullptr;
    Chunk() = default;
    Chunk(const Chunk &) = delete;
    Chunk &operator=(const Chunk &) = delete;
    ~Chunk() {
        if (qk) { infiniopDestroyGemmDescriptor(qk); }
        if (pv) { infiniopDestroyGemmDescriptor(pv); }
        if (softmax) { infiniopDestroySoftmaxDescriptor(softmax); }
        if (causal) { infiniopDestroyCausalSoftmaxDescriptor(causal); }
    }
};

struct Plan {
    graph::GraphTensor out, q, k, v;
    // Retain ownership for the entire plan lifetime; all heads/chunks reuse
    // these buffers serially on the same InfiniCore stream.
    Tensor scores, workspace;
    std::vector<std::unique_ptr<Chunk>> chunks;
    size_t workspace_size = 0;
    float scale;
    Plan(Tensor output, const Tensor &query, const Tensor &key, const Tensor &value, float s)
        : out(output), q(query), k(key), v(value), scale(s) {}
};

std::byte *offset(Tensor &tensor, size_t batch, size_t seq, size_t head) {
    const auto elements = batch * static_cast<size_t>(tensor->stride(0))
                        + seq * static_cast<size_t>(tensor->stride(1))
                        + head * static_cast<size_t>(tensor->stride(2));
    return tensor->data() + elements * tensor->element_size();
}

} // namespace

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi_slopes, float scale, bool is_causal) {
    require(!alibi_slopes.has_value(), "ALiBi is not implemented by this fallback");
    require(std::isfinite(scale), "scale must be finite");
    const auto dtype = q->dtype();
    require(dtype == DataType::F16 || dtype == DataType::BF16 || dtype == DataType::F32,
            "supported dtype values are FP16, BF16, FP32");
    const auto out_span = validate_layout(out);
    for (const auto &input : {q, k, v}) {
        require(input->dtype() == dtype && out->dtype() == dtype, "dtype mismatch");
        reject_output_alias(out, out_span, input, validate_layout(input));
    }
    require(out->shape() == q->shape(), "output shape must equal q shape");
    require(q->size(0) == k->size(0) && q->size(0) == v->size(0), "batch mismatch");
    require(k->shape() == v->shape(), "k and v shapes must match");
    require(q->size(3) == k->size(3), "head dimensions must match");
    require(q->size(2) % k->size(2) == 0, "query heads must be divisible by KV heads");
    const size_t nq = q->size(1), nk = k->size(1), d = q->size(3);
    require(!is_causal || nq <= nk, "causal attention currently requires Sq <= Sk");
    const size_t rows_max = std::min(QUERY_BLOCK, nq);
    require(nk <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())
                      / rows_max / q->element_size(), "score buffer size overflows");

    auto p = std::make_unique<Plan>(out, q, k, v, scale);
    p->scores = Tensor::empty({rows_max, nk}, dtype, out->device());
    const auto handle = context::getInfiniopHandle(out->device());
    for (size_t start = 0; start < nq; start += std::min(QUERY_BLOCK, nq - start)) {
        const auto rows = std::min(QUERY_BLOCK, nq - start);
        // Existing CausalSoftmax uses right-aligned masking. Truncating K/V to
        // this prefix makes its local diagonal equal the global diagonal.
        const auto keys = is_causal ? nk - nq + start + rows : nk;
        auto chunk = std::make_unique<Chunk>();
        chunk->query_start = start;
        TensorMetaData score_desc({rows, keys}, {static_cast<Stride>(nk), 1}, dtype);
        TensorMetaData query_desc({rows, d}, {sequence_stride(q), 1}, dtype);
        TensorMetaData key_transposed_desc({d, keys}, {1, sequence_stride(k)}, dtype);
        TensorMetaData value_desc({keys, d}, {sequence_stride(v), 1}, dtype);
        TensorMetaData output_desc({rows, d}, {sequence_stride(out), 1}, dtype);
        INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(
            handle, &chunk->qk, score_desc.desc, query_desc.desc, key_transposed_desc.desc));
        INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(
            handle, &chunk->pv, output_desc.desc, score_desc.desc, value_desc.desc));
        size_t bytes = 0;
        for (auto desc : {chunk->qk, chunk->pv}) {
            INFINICORE_CHECK_ERROR(infiniopGetGemmWorkspaceSize(desc, &bytes));
            p->workspace_size = std::max(p->workspace_size, bytes);
        }
        if (is_causal) {
            INFINICORE_CHECK_ERROR(infiniopCreateCausalSoftmaxDescriptor(
                handle, &chunk->causal, score_desc.desc, score_desc.desc));
            INFINICORE_CHECK_ERROR(infiniopGetCausalSoftmaxWorkspaceSize(chunk->causal, &bytes));
        } else {
            INFINICORE_CHECK_ERROR(infiniopCreateSoftmaxDescriptor(
                handle, &chunk->softmax, score_desc.desc, score_desc.desc, 1));
            INFINICORE_CHECK_ERROR(infiniopGetSoftmaxWorkspaceSize(chunk->softmax, &bytes));
        }
        p->workspace_size = std::max(p->workspace_size, bytes);
        p->chunks.push_back(std::move(chunk));
    }
    p->workspace = Tensor::empty({std::max(size_t{1}, p->workspace_size)}, DataType::U8, out->device());
    return p.release();
}

void run(void *meta) {
    auto &p = *static_cast<Plan *>(meta);
    auto stream = context::getStream();
    auto scratch = p.scores->data();
    auto workspace = p.workspace->data();
    const auto group_size = p.q->size(2) / p.k->size(2);
    for (size_t batch = 0; batch < p.q->size(0); ++batch) {
        for (size_t head = 0; head < p.q->size(2); ++head) {
            const auto kv_head = head / group_size;
            auto key = offset(p.k, batch, 0, kv_head);
            auto value = offset(p.v, batch, 0, kv_head);
            for (const auto &chunk : p.chunks) {
                auto query = offset(p.q, batch, chunk->query_start, head);
                auto output = offset(p.out, batch, chunk->query_start, head);
                INFINICORE_CHECK_ERROR(infiniopGemm(chunk->qk, workspace, p.workspace_size,
                                                   scratch, query, key, p.scale, 0.0f, stream));
                if (chunk->causal) {
                    INFINICORE_CHECK_ERROR(infiniopCausalSoftmax(chunk->causal, workspace,
                                                                p.workspace_size, scratch, scratch, stream));
                } else {
                    INFINICORE_CHECK_ERROR(infiniopSoftmax(chunk->softmax, workspace,
                                                          p.workspace_size, scratch, scratch, stream));
                }
                INFINICORE_CHECK_ERROR(infiniopGemm(chunk->pv, workspace, p.workspace_size,
                                                   output, scratch, value, 1.0f, 0.0f, stream));
            }
        }
    }
}

void cleanup(void **meta) {
    delete static_cast<Plan *>(*meta);
    *meta = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::hygon_infiniop
#endif
