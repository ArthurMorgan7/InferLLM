#include "op.h"
#include "kernel.h"
#include "cpu/kernel.h"

#include <fstream>
#include <iostream>
#include <string>


using namespace inferllm;


/* ----------------------------------- 抽象类 ---------------------------------- */

OpBase::OpBase(Device* device, const std::string& name, OpIOs inputs)
    : m_device(device)
    , m_name(name) 
    , m_inputs(inputs)  // 在构造函数里绑定了输入
{
    for (auto input : m_inputs) {
        input->add_user();
    }
}

void OpBase::set_weights(OpIOs weights) { 
    m_weights = weights; 
    for (auto weight : m_weights) {
        weight->set_owner_op(this);
    }
}

void OpBase::add_outputs(std::shared_ptr<Tensor> output) {
    output->set_owner_op(this);
    m_outputs.push_back(output);
}

/* -------------------------------- Embedding ------------------------------- */

Embedding::Embedding(OpIOs inputs, uint32_t embd, uint32_t vocab, DType compt_type,Device* device, const std::string& name)
    : OpBase(device, name, inputs)
    , m_embd(embd)
    , m_vocab(vocab)
    , m_comp_type(compt_type) 
{
    // 创建并管理自己的输出，
    add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    
    // 创建并管理 Embedding 权重
    auto embeddings = std::make_shared<Tensor>(device, name + ".weight");   // 权重 Tensor
    std::vector<size_t> shape = {(size_t)vocab, (size_t)embd};  // 设置大小
    embeddings->set_shape(shape);   
    set_weights({embeddings});  // 权重张量绑定到该算子
}
    

void Embedding::deduce_output_shape(){
    size_t len = inputs()[0]->shape()[0];
    outputs()[0]->set_shape({len, m_embd}, m_comp_type);
}



void Embedding::execute(WorkSpace*, uint32_t) {
    // 输入、输出、权重都只是一个Tensor
    auto weight = weights()[0];
    auto input = inputs()[0];
    auto output = outputs()[0];
    DType weight_type = weight->dtype();
    auto len = input->shape()[0];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        switch (weight_type) {
            case DType::Int4:
                kernel->operator()<KernelID::EmbeddingGetInt4Float>(
                        weight->ptr(), input->ptr<uint32_t>(), output->ptr<float>(),len, m_embd);
                break;
            case DType::Int8:
                kernel->operator()<KernelID::EmbeddingGetInt8Float>(
                        weight->ptr(), input->ptr<uint32_t>(), output->ptr<float>(),len, m_embd);
                break;
            case DType::Float32:
                kernel->operator()<KernelID::EmbeddingGetFloatFloat>(
                        weight->ptr<float>(), input->ptr<uint32_t>(),output->ptr<float>(), len, m_embd);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
    } 
    else {
        //! fp16
    }
}




/* -------------------------------- LayerNorm ------------------------------- */

LayerNorm::LayerNorm(Device* device, const std::string& name, OpIOs inputs, size_t embd,
        bool mul, bool bias, bool rms, float eps)
    : OpBase(device, name, inputs)
    , m_mul(mul)
    , m_bias(bias)
    , m_rms(rms)
    , m_norm_eps(eps) 
{
    // 创建输出 Tensor
    add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    
    // 创建权重 Tensor
    std::vector<std::shared_ptr<Tensor>> weights;
    if (m_mul) {
        weights.push_back(std::make_shared<Tensor>(device, name + ".weight"));
        weights.back()->set_shape({embd}, DType::Float32);
    }
    if (m_bias) {
        weights.push_back(std::make_shared<Tensor>(device, name + ".bias"));
        weights.back()->set_shape({embd}, DType::Float32);
    }
    set_weights(weights);
}


void LayerNorm::execute(WorkSpace* workspace, uint32_t nr_past) {
    std::shared_ptr<Tensor> weight = nullptr, bias = nullptr;
    int weight_idx = 0;
    
    // 如果有缩放参数
    if (m_mul) {
        weight = weights()[weight_idx++];
        DType weight_type = weight->dtype();
        INFER_ASSERT(weight_type == DType::Float32, "layer norm weights must be float32.");
    }

    // 如果有偏置参数
    if (m_bias) {
        bias = weights()[weight_idx++];
    }

    auto input = inputs()[0];
    auto output = outputs()[0];
    uint32_t seq_len = input->shape()[0];
    uint32_t embd = input->shape()[1];
    auto kernel = get_kernel();


    if (input->dtype() == DType::Float32) {
        const float* src = input->ptr<float>();
        float* dst = output->ptr<float>();
        float *weight_ptr = nullptr, *bias_ptr = nullptr;

        if (m_mul) {
            weight_ptr = weight->ptr<float>();
        }
        if (m_bias) {
            bias_ptr = bias->ptr<float>();
        }

        // 执行 RMSNorm 或者 LayerNorm
        if (m_rms) {
            // 运算符重载的本质是函数重载，函数是特殊的运算符
            kernel->operator()<KernelID::RmsNormFloat>(src, dst, seq_len, embd, m_norm_eps);
        } 
        else {
            kernel->operator()<KernelID::NormFloat>(src, dst, seq_len, embd, m_norm_eps);
        }

        // 如果有缩放
        if (weight_ptr) {
            kernel->operator()<KernelID::ElemwiseBroadcastDim0Src1Float>(dst, weight_ptr, dst, seq_len, embd, ElemMode::Mul);
        }

        // 如果有偏置
        if (bias_ptr) {
            kernel->operator()<KernelID::ElemwiseBroadcastDim0Src1Float>(dst, bias_ptr, dst, seq_len, embd, ElemMode::Add);
        }
    }
}


/* ------------------------------- Elementwise ------------------------------ */

void Elemwise::execute(WorkSpace*, uint32_t) {
    auto output = outputs()[0];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        if (m_scale == -INFINITY) {
            InData<float> in_datas;
            for (auto input : inputs()) {
                in_datas.push_back(input->ptr<float>());
            }
            float* dst = output->ptr<float>();
            size_t len = output->length();
            kernel->operator()<KernelID::ElemwiseFloat>(in_datas, dst, len, m_mode);
        } 
        else {
            float* dst = output->ptr<float>();
            size_t len = output->length();
            kernel->operator()<KernelID::ElemwiseFloatScale>(inputs()[0]->ptr<float>(), dst, len, m_scale);

            InData<float> in_datas;
            for (auto input : inputs()) {
                in_datas.push_back(input->ptr<float>());
            }
            in_datas[0] = dst;
            kernel->operator()<KernelID::ElemwiseFloat>(in_datas, dst, len, m_mode);
        }
    } else {
        //! fp16
    }
}

/* ----------------------------------- GLU ---------------------------------- */

void SpliteHalfActiveMul::execute(WorkSpace*, uint32_t) {
    auto input = inputs()[0];
    auto output = outputs()[0];
    auto out_dim = output->shape()[1];
    auto seqlen = input->shape()[0];
    auto dim = input->shape()[1];
    auto kernel = get_kernel();
    for (int i = 0; i < seqlen; i++) {
        if (input->dtype() == DType::Float32) {
            float* dst = output->ptr<float>() + i * out_dim;
            float* in_data = input->ptr<float>() + i * dim;
            auto len = dim / 2;
            kernel->operator()<KernelID::ElemwiseFloat>(InData<float>{in_data}, dst, len, m_mode);

            kernel->operator()<KernelID::ElemwiseFloat>(InData<float>{dst, in_data + len}, dst, len, ElemMode::Mul);
        } else {
            //! fp16
        }
    }
}

/* --------------------------------- MatMul 算子 --------------------------------- */

void MatMul::execute(WorkSpace* workspace, uint32_t) {
    auto N = weights()[0]->shape()[0];
    auto K = inputs()[0]->shape()[1];
    auto M = inputs()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto weight_dtype = weights()[0]->dtype();
    void* p_workspace = workspace->ptr();
    uint32_t p_workspace_size = workspace->length();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32) {
        float* dst = outputs()[0]->ptr<float>();
        const float* bias = nullptr;
        if (m_bias) {
            bias = weights()[1]->ptr<float>();
        }
        const float* src = inputs()[0]->ptr<float>();
        switch (weight_dtype) {
            case DType::Int4:
                if (!m_weight_packed) {
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            dst, weights()[0]->ptr(), bias, src, M, N, K, p_workspace,
                            p_workspace_size);
                } else {
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            dst, weights()[0]->ptr(), bias, src, M, N * PACK_SIZE, K,
                            p_workspace, p_workspace_size);
                }
                break;
            case DType::Int8:
                kernel->operator()<KernelID::MatmulInt8Float>(
                        dst, weights()[0]->ptr(), bias, src, M, N, K, p_workspace,
                        p_workspace_size);
                break;
            case DType::Float32:
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        dst, weights()[0]->ptr<float>(), bias, src, M, N, K,
                        p_workspace, p_workspace_size);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
    }
}

size_t MatMul::get_workspace_in_byte() {
    uint32_t M = inputs()[0]->shape()[0];
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto kernel = get_kernel();
    auto weight_dtype = weights()[0]->dtype();
    if (src_dtype == DType::Float32) {
        return kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), M, N, K);
    }
    return 0;
}

//! all the memory is the host memory
std::vector<size_t> MatMul::preprocess_weight(Tensor* tensor, void* src, void* dst) {
    INFER_ASSERT(tensor->dtype() == DType::Int4, "only support optimized int4 kernel");
    auto weight_shape = tensor->shape();
    size_t M = weight_shape[0];
    size_t N = weight_shape[1];
    INFER_ASSERT(N % QK40 == 0, "error of embd size.");
    INFER_ASSERT(M % PACK_SIZE == 0, "the M in matmul is not align to 8.");

    auto kernel = get_kernel();
    kernel->operator()<KernelID::MatmulInt4WeightReorder>(M, N, dst, src, PACK_SIZE);
    size_t block_m = M / PACK_SIZE;

    m_weight_packed = true;
    return {block_m, N * PACK_SIZE};
}


/* ------------------------------ MatMulLast 算子 ----------------------------- */

void MatMulLast::execute(WorkSpace* workspace, uint32_t) {
    auto N = weights()[0]->shape()[0];
    auto K = weights()[0]->shape()[1];
    //! only compute the last token
    auto M = 1;
    auto row = inputs()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto weight_dtype = weights()[0]->dtype();
    void* p_workspace = workspace->ptr();
    uint32_t p_workspace_size = workspace->length();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32) {
        float* dst = outputs()[0]->ptr<float>();
        const float* bias = nullptr;
        if (m_bias) {
            bias = weights()[1]->ptr<float>();
        }
        const float* src = inputs()[0]->ptr<float>() + (row - 1) * K;
        switch (weight_dtype) {
            case DType::Int4:
                if (!m_weight_packed) {
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            dst, weights()[0]->ptr(), bias, src, M, N, K, p_workspace,
                            p_workspace_size);
                } else {
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            dst, weights()[0]->ptr(), bias, src, M, N * PACK_SIZE, K,
                            p_workspace, p_workspace_size);
                }
                break;
            case DType::Int8:
                kernel->operator()<KernelID::MatmulInt8Float>(
                        dst, weights()[0]->ptr(), bias, src, M, N, K, p_workspace,
                        p_workspace_size);
                break;
            case DType::Float32:
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        dst, weights()[0]->ptr<float>(), bias, src, M, N, K,
                        p_workspace, p_workspace_size);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
    }
}

size_t MatMulLast::get_workspace_in_byte() {
    uint32_t M = 1;
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32) {
        return kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), M, N, K);
    }
    return 0;
}

/* ------------------------------ Attention 算子 ------------------------------ */

AttentionBase::AttentionBase(
        Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
        uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
        bool fused_weights, bool bias)
        : OpBase(device, name, inputs),
            m_embd(embd),
            m_head(head),
            m_ctx(nr_ctx),
            m_layer_id(layer_id),
            m_fused_weights(fused_weights),
            m_bias(bias) 
{
    add_outputs(std::make_shared<Tensor>(device, name + "_out"));
    std::vector<std::shared_ptr<Tensor>> weights;
    if (m_fused_weights) {
        auto weight_fused = std::make_shared<Tensor>(device, name + ".wqkv.weight");
        weight_fused->set_shape(std::vector<size_t>{m_embd * 3, m_embd});
        weights.push_back(weight_fused);
        if (m_bias) {
            auto weight_bias =
                    std::make_shared<Tensor>(device, name + ".wqkv.bias");
            weight_bias->set_shape(std::vector<size_t>{m_embd * 3});
            weights.push_back(weight_bias);
        }
    } else {
        auto weight_q = std::make_shared<Tensor>(device, name + ".wq.weight");
        weight_q->set_shape(std::vector<size_t>{embd, embd});
        auto weight_k = std::make_shared<Tensor>(device, name + ".wk.weight");
        weight_k->set_shape(std::vector<size_t>{embd, embd});
        auto weight_v = std::make_shared<Tensor>(device, name + ".wv.weight");
        weight_v->set_shape(std::vector<size_t>{embd, embd});
        weights.push_back(weight_q);
        weights.push_back(weight_k);
        weights.push_back(weight_v);
        if (m_bias) {
            auto bias_q = std::make_shared<Tensor>(device, name + ".wq.bias");
            bias_q->set_shape(std::vector<size_t>{embd});
            auto bias_k = std::make_shared<Tensor>(device, name + ".wk.bias");
            bias_k->set_shape(std::vector<size_t>{embd});
            auto bias_v = std::make_shared<Tensor>(device, name + ".wv.bias");
            bias_v->set_shape(std::vector<size_t>{embd});
            weights.push_back(bias_q);
            weights.push_back(bias_k);
            weights.push_back(bias_v);
        }
    }
    set_weights(weights);
}

void AttentionBase::pre_execute(){
    auto token_len = inputs()[0]->shape()[0];
    for (auto weight : weights()) {
        weight->prepare_data();
    }
    auto output = outputs()[0];
    if (output->get_curr_user_count() == 0) {
        output->prepare_data();
        output->resume_user_count();
    }
    m_kstorage->prepare_data_with_length(token_len);
    m_vstorage->prepare_data_with_length(token_len);
}

void AttentionBase::end_execute() {
    for (auto weight : weights()) {
        weight->recall_data();
    }
    for (auto input : inputs()) {
        input->decrease_curr_user_count();
    }
    auto token_len = inputs()[0]->shape()[0];
    m_kstorage->add_id(token_len);
    m_vstorage->add_id(token_len);
    m_kstorage->recall_data();
    m_vstorage->recall_data();
}

bool AttentionBase::need_preprocess_weight(Tensor* weight) {
    auto kernel = get_kernel();
    bool int4 = weight->dtype() == DType::Int4;
    size_t M = weight->shape()[0];
    bool right_weight = false;
    bool optimized = kernel->supported_optimization(KernelOptMethod::MatmulInt4Reorder);
    //! only when the weight is int4
    if (m_fused_weights) {
        right_weight = weight->name() == weights()[0]->name();
    } else {
        right_weight = weight->name() == weights()[0]->name() ||
                        weight->name() == weights()[1]->name() ||
                        weight->name() == weights()[2]->name();
    }
    return optimized && int4 && right_weight && M % PACK_SIZE == 0;
}


size_t AttentionBase::get_workspace_in_byte() {
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto src_dtype = input->dtype();
    auto w_dtype = weights()[0]->dtype();

    uint32_t M = inputs()[0]->shape()[0];
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto kernel = get_kernel();
    if (m_packed_weight) {
        N *= PACK_SIZE;
    }
    uint32_t seqlen = input->shape()[0];

    size_t total = 0;
    if (src_dtype == DType::Float32) {
        //! matmul tmp
        switch (w_dtype) {
            case DType::Int4:
                total += kernel->get_workspace<KernelID::MatmulInt4Float>(
                        kernel->nr_thread(), M, N, K);
                break;
            case DType::Int8:
                total += kernel->get_workspace<KernelID::MatmulInt8Float>(
                        kernel->nr_thread(), M, N, K);
                break;
            case DType::Float32:
                total += kernel->get_workspace<KernelID::MatmulFloatFloat>(
                        kernel->nr_thread(), M, N, K);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
        //! out q
        total += seqlen * m_embd * sizeof(float);
        //! qk out
        total += m_head * seqlen * m_ctx * sizeof(float);
    }
    return total;
}

std::vector<size_t> AttentionBase::preprocess_weight(
        Tensor* tensor, void* src, void* dst) {
    INFER_ASSERT(tensor->dtype() == DType::Int4, "only support optimized int4 kernel");
    auto weight_shape = tensor->shape();
    size_t M = weight_shape[0];
    size_t N = weight_shape[1];
    INFER_ASSERT(N % QK40 == 0, "error of embd size.");
    INFER_ASSERT(M % PACK_SIZE == 0, "the M in matmul is not align to 8.");

    auto kernel = get_kernel();
    kernel->operator()<KernelID::MatmulInt4WeightReorder>(M, N, dst, src, PACK_SIZE);
    size_t block_m = M / PACK_SIZE;

    m_packed_weight = true;
    return {block_m, N * PACK_SIZE};
}

LlamaAttention::LlamaAttention(
        Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
        uint32_t rot, uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
        DType compt_type, bool fused_weights, bool bias,
        RotMode rotary_mode)
    : AttentionBase(device, name, inputs, embd, nr_ctx, head, layer_id, fused_weights,bias) 
{
    m_rot = rot;
    m_rotary_mode = rotary_mode;
    m_kstorage = std::make_unique<KvStorage>(std::vector<size_t>{nr_ctx, embd}, compt_type, device);
    m_vstorage = std::make_unique<KvStorage>(std::vector<size_t>{nr_ctx, embd}, compt_type, device);
}

void LlamaAttention::execute(WorkSpace* workspace, uint32_t nr_past) {
    INFER_ASSERT(
            nr_past == m_kstorage->current_index(),
            "The index in kv storage is not the same as input\n");
    auto w_dtype = weights()[0]->dtype();
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto in_dtype = input->dtype();
    uint32_t seqlen = input->shape()[0];
    uint32_t embd = input->shape()[1];
    uint32_t head = m_head;
    auto kernel = get_kernel();

    void *p_wq = nullptr, *p_wk = nullptr, *p_wv = nullptr;
    float *p_bq = nullptr, *p_bk = nullptr, *p_bv = nullptr;
    if (m_fused_weights) {
        size_t offset =
                embd * embd * dtype_in_byte(w_dtype) / dtype_block_size(w_dtype);
        p_wq = weights()[0]->ptr();
        p_wk = static_cast<int8_t*>(p_wq) + offset;
        p_wv = static_cast<int8_t*>(p_wk) + offset;
        if (m_bias) {
            p_bq = weights()[1]->ptr<float>();
            p_bk = p_bq + embd;
            p_bv = p_bk + embd;
        }
    } else {
        p_wq = weights()[0]->ptr();
        p_wk = weights()[1]->ptr();
        p_wv = weights()[2]->ptr();
        if (m_bias) {
            p_bq = weights()[3]->ptr<float>();
            p_bk = weights()[4]->ptr<float>();
            p_bv = weights()[5]->ptr<float>();
        }
    }

    auto weight_type = weights()[0]->dtype();

    void* p_work = workspace->ptr();
    size_t matmul_size = 0;
    switch (weight_type) {
        case DType::Int4:
            matmul_size = kernel->get_workspace<KernelID::MatmulInt4Float>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        case DType::Int8:
            matmul_size = kernel->get_workspace<KernelID::MatmulInt8Float>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        case DType::Float32:
            matmul_size = kernel->get_workspace<KernelID::MatmulFloatFloat>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        default:
            INFER_ASSERT(0, "not support");
    }

    uint32_t size = workspace->length();

    void* q_out = static_cast<void*>(static_cast<char*>(p_work) + matmul_size);
    void* qk_out = static_cast<void*>(
            static_cast<char*>(q_out) + seqlen * m_embd * sizeof(float));

    if (in_dtype == DType::Float32) {
        //! compute k, q, v
        const float* pdata = input->ptr<float>();
        float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
        float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
        float* p_outq = static_cast<float*>(q_out);
        switch (w_dtype) {
            case DType::Int4:
                if (!m_packed_weight) {
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outk, p_wk, p_bk, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outv, p_wv, p_bv, pdata, seqlen, embd, embd, p_work,
                            size);
                } else {
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outk, p_wk, p_bk, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outv, p_wv, p_bv, pdata, seqlen, embd, embd, p_work,
                            size);
                }
                break;
            case DType::Int8:
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work, size);
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outk, p_wk, p_bk, pdata, seqlen, embd, embd, p_work, size);
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outv, p_wv, p_bv, pdata, seqlen, embd, embd, p_work, size);
                break;
            case DType::Float32:
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outq, (float*)p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                        size);
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outk, (float*)p_wk, p_bk, pdata, seqlen, embd, embd, p_work,
                        size);
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outv, (float*)p_wv, p_bv, pdata, seqlen, embd, embd, p_work,
                        size);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
        //! rope Q

        float* p_totalk = static_cast<float*>(m_kstorage->ptr());
        if (m_rotary_mode == RotMode::ModelRotHalf) {
            kernel->operator()<KernelID::RopeFloat>(
                    p_outq, p_outq, nr_past, m_rot, m_rotary_mode, seqlen, head,
                    embd / head);
            //! rope K
            kernel->operator()<KernelID::RopeFloat>(
                    p_outk, p_outk, nr_past, m_rot, m_rotary_mode, seqlen, head,
                    embd / head);
        } else {
            kernel->operator()<KernelID::RopeFloat>(
                    p_outq, p_outq, nr_past, m_rot, RotMode::Mode0, seqlen, head,
                    embd / head);
            //! rope K
            kernel->operator()<KernelID::RopeFloat>(
                    p_totalk, p_totalk, nr_past, m_rot, RotMode::Mode1,
                    seqlen + nr_past, head, embd / head);
        }
        //! Q*k with transpose
        kernel->operator()<KernelID::MatmulWithHeadStrideFloat>(
                (float*)qk_out, p_totalk, p_outq, seqlen, embd, head, nr_past);
        //! scale and diag
        float scale = 1.0f / sqrt(float(embd) / head);
        kernel->operator()<KernelID::ScaleDiagMaskFloat>(
                (float*)qk_out, (float*)qk_out, scale, nr_past, seqlen, head);
        //! softmax
        kernel->operator()<KernelID::SoftmaxFloat>(
                (float*)qk_out, (float*)qk_out, head * seqlen, nr_past + seqlen);
        //! compute v_out
        float* out = outputs()[0]->ptr<float>();
        float* p_totalv = static_cast<float*>(m_vstorage->ptr());
        kernel->operator()<KernelID::HeadBatchedMatmulFloat>(
                out, p_totalv, (float*)qk_out, seqlen, embd, head, nr_past);
    }
}

GlmAttention::GlmAttention(
        Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
        uint32_t rot, uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
        DType compt_type, bool fused_weights, bool bias,
        RotMode rotary_mode)
    : AttentionBase(device, name, inputs, embd, nr_ctx, head, layer_id, fused_weights, bias) 
{
    m_rotary_mode = rotary_mode;
    m_kstorage = std::make_unique<KvStorage>(std::vector<size_t>{nr_ctx, embd}, compt_type, device);
    m_vstorage = std::make_unique<KvStorage>(std::vector<size_t>{nr_ctx, embd}, compt_type, device);
}


void GlmAttention::execute(WorkSpace* workspace, uint32_t nr_past) {
    INFER_ASSERT(
            nr_past == m_kstorage->current_index(),
            "The index in kv storage is not the same as input\n");
    auto w_dtype = weights()[0]->dtype();
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto in_dtype = input->dtype();
    uint32_t seqlen = input->shape()[0];
    uint32_t embd = input->shape()[1];
    uint32_t head = m_head;
    auto kernel = get_kernel();
    if (nr_past == 0) {
        INFER_ASSERT(
                seqlen > 2, "seqlen is too short, must end with gmask and end token");
        m_gmask_position = seqlen - 2;
    }

    void *p_wq = nullptr, *p_wk = nullptr, *p_wv = nullptr;
    float *p_bq = nullptr, *p_bk = nullptr, *p_bv = nullptr;
    if (m_fused_weights) {
        size_t offset =
                embd * embd * dtype_in_byte(w_dtype) / dtype_block_size(w_dtype);
        p_wq = weights()[0]->ptr();
        p_wk = static_cast<int8_t*>(p_wq) + offset;
        p_wv = static_cast<int8_t*>(p_wk) + offset;
        if (m_bias) {
            p_bq = weights()[1]->ptr<float>();
            p_bk = p_bq + embd;
            p_bv = p_bk + embd;
        }
    } else {
        p_wq = weights()[0]->ptr();
        p_wk = weights()[1]->ptr();
        p_wv = weights()[2]->ptr();
        if (m_bias) {
            p_bq = weights()[3]->ptr<float>();
            p_bk = weights()[4]->ptr<float>();
            p_bv = weights()[5]->ptr<float>();
        }
    }

    auto weight_type = weights()[0]->dtype();

    void* p_work = workspace->ptr();
    size_t matmul_size = 0;
    switch (weight_type) {
        case DType::Int4:
            matmul_size = kernel->get_workspace<KernelID::MatmulInt4Float>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        case DType::Int8:
            matmul_size = kernel->get_workspace<KernelID::MatmulInt8Float>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        case DType::Float32:
            matmul_size = kernel->get_workspace<KernelID::MatmulFloatFloat>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        default:
            INFER_ASSERT(0, "not support");
    }
    uint32_t size = workspace->length();

    void* q_out = static_cast<void*>(static_cast<char*>(p_work) + matmul_size);
    void* qk_out = static_cast<void*>(
            static_cast<char*>(q_out) + seqlen * m_embd * sizeof(float));

    if (in_dtype == DType::Float32) {
        //! compute k, q, v
        const float* pdata = input->ptr<float>();
        float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
        float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
        float* p_outq = static_cast<float*>(q_out);
        switch (w_dtype) {
            case DType::Int4:
                if (!m_packed_weight) {
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outk, p_wk, p_bk, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outv, p_wv, p_bv, pdata, seqlen, embd, embd, p_work,
                            size);
                } else {
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outk, p_wk, p_bk, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outv, p_wv, p_bv, pdata, seqlen, embd, embd, p_work,
                            size);
                }
                break;
            case DType::Int8:
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work, size);
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outk, p_wk, p_bk, pdata, seqlen, embd, embd, p_work, size);
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outv, p_wv, p_bv, pdata, seqlen, embd, embd, p_work, size);
                break;
            case DType::Float32:
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outq, (float*)p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                        size);
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outk, (float*)p_wk, p_bk, pdata, seqlen, embd, embd, p_work,
                        size);
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outv, (float*)p_wv, p_bv, pdata, seqlen, embd, embd, p_work,
                        size);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
        //! rope Q
        kernel->operator()<KernelID::GlmRopeFloat>(
                p_outq, p_outq, nr_past, m_gmask_position, seqlen, head, embd / head);
        //! scale Q
        float scale_q = 1 / ((m_layer_id + 1) * sqrt(embd / head));
        kernel->operator()<KernelID::ElemwiseFloatScale>(
                p_outq, p_outq, seqlen * embd, scale_q);
        //! rope K
        float* p_totalk = static_cast<float*>(m_kstorage->ptr());
        kernel->operator()<KernelID::GlmRopeFloat>(
                p_outk, p_outk, nr_past, m_gmask_position, seqlen, head, embd / head);
        //! Q*k with transpose
        kernel->operator()<KernelID::MatmulWithHeadStrideFloat>(
                (float*)qk_out, p_totalk, p_outq, seqlen, embd, head, nr_past);
        //! qk * (layer + 1)
        kernel->operator()<KernelID::ElemwiseFloatScale>(
                (float*)qk_out, (float*)qk_out, head * seqlen * (nr_past + seqlen),
                (m_layer_id + 1));
        if (seqlen > 1) {
            //! configure the gMask
            kernel->operator()<KernelID::GlmGmask>(
                    (float*)qk_out, nr_past, seqlen, head);
        }
        //! softmax
        kernel->operator()<KernelID::SoftmaxFloat>(
                (float*)qk_out, (float*)qk_out, head * seqlen, nr_past + seqlen);
        //! compute v_out
        float* out = outputs()[0]->ptr<float>();
        float* p_totalv = static_cast<float*>(m_vstorage->ptr());
        kernel->operator()<KernelID::HeadBatchedMatmulFloat>(
                out, p_totalv, (float*)qk_out, seqlen, embd, head, nr_past);
    }
}

Glm2MultiQueryAttention::Glm2MultiQueryAttention(
        Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
        uint32_t query_group_num, uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
        DType compt_type, bool fused_weights, bool bias, RotMode rotary_mode)
        : AttentionBase(device, name, inputs) {
    m_embd = embd;
    m_head = head;
    m_ctx = nr_ctx;
    m_layer_id = layer_id;
    m_fused_weights = fused_weights;
    m_bias = bias;
    m_query_group_num = query_group_num;

    add_outputs(std::make_shared<Tensor>(device, name + "_out"));
    std::vector<std::shared_ptr<Tensor>> weights;
    INFER_ASSERT(
            m_fused_weights,
            "Glm2MultiQueryAttention only support fused weights.\n");
    auto weight_fused = std::make_shared<Tensor>(device, name + ".wqkv.weight");

    uint32_t weight_dim0 = m_embd + query_group_num * 2 * m_embd / head;
    weight_fused->set_shape(std::vector<size_t>{weight_dim0, m_embd});
    weights.push_back(weight_fused);
    if (m_bias) {
        auto weight_bias = std::make_shared<Tensor>(device, name + ".wqkv.bias");
        weight_bias->set_shape(std::vector<size_t>{weight_dim0});
        weights.push_back(weight_bias);
    }
    set_weights(weights);

    uint32_t sub_dim = embd / head;
    m_kstorage = std::make_unique<KvStorage>(
            std::vector<size_t>{nr_ctx, sub_dim * query_group_num}, compt_type,
            device);
    m_vstorage = std::make_unique<KvStorage>(
            std::vector<size_t>{nr_ctx, sub_dim * query_group_num}, compt_type,
            device);
}


void Glm2MultiQueryAttention::execute(WorkSpace* workspace, uint32_t nr_past) {
    INFER_ASSERT(
            nr_past == m_kstorage->current_index(),
            "The index in kv storage is not the same as input\n");
    auto w_dtype = weights()[0]->dtype();
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto in_dtype = input->dtype();
    uint32_t seqlen = input->shape()[0];
    uint32_t embd = input->shape()[1];
    uint32_t head = m_head;
    auto kernel = get_kernel();

    void *p_wq = nullptr, *p_wk = nullptr, *p_wv = nullptr;
    float *p_bq = nullptr, *p_bk = nullptr, *p_bv = nullptr;
    if (m_fused_weights) {
        size_t offset_q =
                embd * embd * dtype_in_byte(w_dtype) / dtype_block_size(w_dtype);
        size_t offset_kv = (embd / head * m_query_group_num) * embd *
                           dtype_in_byte(w_dtype) / dtype_block_size(w_dtype);
        p_wq = weights()[0]->ptr();
        p_wk = static_cast<int8_t*>(p_wq) + offset_q;
        p_wv = static_cast<int8_t*>(p_wk) + offset_kv;
        if (m_bias) {
            p_bq = weights()[1]->ptr<float>();
            p_bk = p_bq + embd;
            p_bv = p_bk + embd / head * m_query_group_num;
        }
    } else {
        INFER_ASSERT(0, "not support");
    }

    auto weight_type = weights()[0]->dtype();

    void* p_work = workspace->ptr();
    size_t matmul_size = 0;
    switch (weight_type) {
        case DType::Int4:
            matmul_size = kernel->get_workspace<KernelID::MatmulInt4Float>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        case DType::Int8:
            matmul_size = kernel->get_workspace<KernelID::MatmulInt8Float>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        case DType::Float32:
            matmul_size = kernel->get_workspace<KernelID::MatmulFloatFloat>(
                    kernel->nr_thread(), seqlen, embd, embd);
            break;
        default:
            INFER_ASSERT(0, "not support");
    }
    uint32_t size = workspace->length();

    void* q_out = static_cast<void*>(static_cast<char*>(p_work) + matmul_size);
    void* qk_out = static_cast<void*>(
            static_cast<char*>(q_out) + seqlen * m_embd * sizeof(float));

    uint32_t head_dim = embd / head;
    uint32_t kv_length = head_dim * m_query_group_num;

    if (in_dtype == DType::Float32) {
        //! compute k, q, v
        const float* pdata = input->ptr<float>();
        float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
        float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
        float* p_outq = static_cast<float*>(q_out);
        switch (w_dtype) {
            case DType::Int4:
                if (!m_packed_weight) {
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outk, p_wk, p_bk, pdata, seqlen, kv_length, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4Float>(
                            p_outv, p_wv, p_bv, pdata, seqlen, kv_length, embd, p_work,
                            size);
                } else {
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outk, p_wk, p_bk, pdata, seqlen, kv_length, embd, p_work,
                            size);
                    kernel->operator()<KernelID::MatmulInt4FloatPacked>(
                            p_outv, p_wv, p_bv, pdata, seqlen, kv_length, embd, p_work,
                            size);
                }
                break;
            case DType::Int8:
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outq, p_wq, p_bq, pdata, seqlen, embd, embd, p_work, size);
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outk, p_wk, p_bk, pdata, seqlen, kv_length, embd, p_work,
                        size);
                kernel->operator()<KernelID::MatmulInt8Float>(
                        p_outv, p_wv, p_bv, pdata, seqlen, kv_length, embd, p_work,
                        size);
                break;
            case DType::Float32:
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outq, (float*)p_wq, p_bq, pdata, seqlen, embd, embd, p_work,
                        size);
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outk, (float*)p_wk, p_bk, pdata, seqlen, kv_length, embd,
                        p_work, size);
                kernel->operator()<KernelID::MatmulFloatFloat>(
                        p_outv, (float*)p_wv, p_bv, pdata, seqlen, kv_length, embd,
                        p_work, size);
                break;
            default:
                INFER_ASSERT(0, "not support");
        }
        //! rope Q
        kernel->operator()<KernelID::RopeFloat>(
                p_outq, p_outq, nr_past, head_dim / 2, RotMode::Mode0, seqlen, head,
                embd / head);
        //! rope K
        float* p_totalk = static_cast<float*>(m_kstorage->ptr());
        kernel->operator()<KernelID::RopeFloat>(
                p_totalk, p_totalk, nr_past, head_dim / 2, RotMode::Mode1,
                seqlen + nr_past, m_query_group_num, embd / head);

        //! Q*k with transpose
        kernel->operator()<KernelID::MatmulWithHeadStrideQBroadCastKFloat>(
                (float*)qk_out, p_totalk, p_outq, seqlen, embd, head, m_query_group_num,
                nr_past);

        //! scale and diag
        float scale = 1.0f / sqrt(float(embd) / head);
        kernel->operator()<KernelID::ScaleDiagMaskFloat>(
                (float*)qk_out, (float*)qk_out, scale, nr_past, seqlen, head);

        //! softmax
        kernel->operator()<KernelID::SoftmaxFloat>(
                (float*)qk_out, (float*)qk_out, head * seqlen, nr_past + seqlen);
        //! compute v_out
        float* out = outputs()[0]->ptr<float>();
        float* p_totalv = static_cast<float*>(m_vstorage->ptr());
        kernel->operator()<KernelID::HeadBatchedMatmulBroadCastVFloat>(
                out, p_totalv, (float*)qk_out, seqlen, embd, head, m_query_group_num,
                nr_past);
    }
}

/* ------------------------------- DiagMask算子 ------------------------------- */

void DiagMask::execute(WorkSpace*, uint32_t n_past) {
    auto output = outputs()[0];
    uint32_t head = output->shape()[0];
    uint32_t N = output->shape()[1];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        const float* in_data = inputs()[0]->ptr<float>();
        float* dst = output->ptr<float>();
        kernel->operator()<KernelID::DiagMaskFloat>(dst, in_data, n_past, N, head);
    } else {
        //! fp16
    }
}


/* --------------------------------- SoftMax -------------------------------- */


void SoftMax::execute(WorkSpace*, uint32_t) {
    auto input = inputs()[0];
    auto output = outputs()[0];
    uint32_t seq_len = input->shape()[0];
    uint32_t embd = input->shape()[1];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        float* src = input->ptr<float>();
        float* dst = output->ptr<float>();
        // 利用模板的偏特化选择不同的算子实现
        kernel->operator()<KernelID::SoftmaxFloat>(src, dst, seq_len, embd);
    } else {
        //! fp16
    }
}
