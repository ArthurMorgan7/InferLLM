#pragma once

#include "kernel/kernel.h"
#include "kvstorage.h"
#include "tensor.h"

namespace inferllm {

using OpIOs = std::vector<std::shared_ptr<Tensor>>;
constexpr static size_t PACK_SIZE = 8;

//! Base class of an Op, the call step is:
//! call deduce_output_shape to get the output tensor shape
//! call init method to get init the op and compute the workspace
//! before execution, should call pre_execute to prepare the resource
//! call execute to get the compute result
//! call end execution to retrieve the resource

/* --------------------------------- 算子的抽象基类 -------------------------------- */
class OpBase {
public:
    OpBase(Device* device, const std::string& name, OpIOs inputs);

    std::string name() { return m_name; }
    Device* device() { return m_device; }
    OpIOs weights() { return m_weights; }
    OpIOs inputs() { return m_inputs; }
    OpIOs outputs() { return m_outputs; }
    Kernel* get_kernel() { return m_device->kernel(); }
     

    void set_name(std::string name) { m_name = name; }
    void set_weights(OpIOs weights);        // 把权重张量绑定到该算子    
    
    void add_outputs(std::shared_ptr<Tensor> output);   // 把输出张量绑定到该算子

    virtual void pre_execute() {
        // 准备权重数据，加载到内存
        for (auto weight : m_weights) {
            weight->prepare_data();
        }

        // 准备输出张量
        for (auto output : m_outputs) {
            if (output->get_curr_user_count() == 0 && !output->shared()) {
                output->resume_user_count();
                output->prepare_data(); // 加载到内存
            }
        }
    };

    virtual void execute(WorkSpace* workspace, uint32_t nr_past) {}

    virtual void end_execute() {
        for (auto input : m_inputs) {
            input->decrease_curr_user_count();
        }
    };

    virtual void deduce_output_shape() {
        m_outputs[0]->set_shape(m_inputs[0]->shape(), m_inputs[0]->dtype());
    };

    virtual size_t get_workspace_in_byte() { return 0; }

    virtual void load_weights(std::ifstream&){};

    virtual uint32_t nr_weights() { return 1; };

    //! init the op, and return the workspace need when execute
    virtual void init(OpIOs, OpIOs, WorkSpace*){};

    //! for better optimized the compute, some op need preprocess the weight, so that
    //! the compute is friendly to the compute kernel
    virtual bool need_preprocess_weight(Tensor*) { return false; }

    virtual std::vector<size_t> preprocess_weight(Tensor* tensor, void* src, void* dst) {return std::vector<size_t>();
}

private:
    Device* m_device;   // 当前算子的运行设备
    OpIOs m_weights;    // 输入张量列表
    OpIOs m_inputs;     // 输出张量列表
    OpIOs m_outputs;    // 权重张量列表
    std::string m_name; // 当前算子的名称
};

/* ------------------------------ Embedding 算子 ------------------------------ */

class Embedding : public OpBase {
public:
    Embedding(OpIOs inputs, uint32_t embd, uint32_t vocab, DType compt_type,Device* device, const std::string& name);

    void deduce_output_shape() override;

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    DType m_comp_type;
    uint32_t m_embd;
    uint32_t m_vocab;
};

/* ------------------------------- LayerNorm算子 ------------------------------ */

class LayerNorm : public OpBase {
public:
    LayerNorm(Device* device, const std::string& name, OpIOs inputs, size_t embd,
        bool mul = true, bool bias = false, bool rms = true, float eps = 1e-5);

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    bool m_mul;
    bool m_bias;
    bool m_rms;
    float m_norm_eps = 1e-5;
};

/* --------------------------------- 矩阵乘法算子 --------------------------------- */

class MatMul : public OpBase {
public:
    MatMul(Device* device, const std::string& name, OpIOs inputs,
           std::vector<size_t> shape, bool bias = false)
            : OpBase(device, name, inputs), m_bias(bias) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
        std::vector<std::shared_ptr<Tensor>> weights;
        auto weight = std::make_shared<Tensor>(device, name + ".weight");
        weight->set_shape(shape);
        weights.push_back(weight);
        if (bias) {
            auto bias = std::make_shared<Tensor>(device, name + ".bias");
            std::vector<size_t> bias_shape;
            bias_shape.push_back(shape[0]);
            bias->set_shape(bias_shape);
            weights.push_back(bias);
        }
        set_weights(weights);
    }

    virtual void deduce_output_shape() override {
        auto weight_shape = weights()[0]->shape();
        auto input_shape = inputs()[0]->shape();
        size_t M = input_shape.size() == 2 ? input_shape[0] : input_shape[1];
        size_t N = weight_shape[0];
        if (m_weight_packed) {
            N = N * PACK_SIZE;
        }
        outputs()[0]->set_shape({M, N}, inputs()[0]->dtype());
    }

    virtual bool need_preprocess_weight(Tensor* weight) override {
        auto kernel = get_kernel();
        //! only when the weight is int4
        if (weight->name() == weights()[0]->name()) {
            size_t M = weight->shape()[0];
            bool optimized =
                    kernel->supported_optimization(KernelOptMethod::MatmulInt4Reorder);
            bool int4 = weight->dtype() == DType::Int4;
            if (optimized && int4 && (M % PACK_SIZE == 0)) {
                return true;
            }
        }
        return false;
    }

    virtual std::vector<size_t> preprocess_weight(
            Tensor* tensor, void* src, void* dst) override;

    virtual void execute(WorkSpace* workspace, uint32_t nr_past) override;

    size_t get_workspace_in_byte() override;

    bool m_bias = false;
    bool m_weight_packed = false;
};



class MatMulLast : public MatMul {
public:
    using MatMul::MatMul;

    void deduce_output_shape() override {
        auto weight_shape = weights()[0]->shape();
        auto input_shape = inputs()[0]->shape();
        //! only compute the last token
        size_t M = 1;
        size_t K = weight_shape[1];
        size_t N = weight_shape[0];
        if (m_weight_packed) {
            N = N * PACK_SIZE;
        }
        outputs()[0]->set_shape({M, N}, inputs()[0]->dtype());
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;
    virtual bool need_preprocess_weight(Tensor*) override { return false; }

    size_t get_workspace_in_byte() override;
};


/* -------------------------------- SoftMax 算子 ------------------------------- */

class SoftMax : public OpBase {
public:
    SoftMax(Device* device, const std::string& name, OpIOs inputs)
            : OpBase(device, name, inputs) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;
};



class Reshape : public OpBase {
public:
    Reshape(Device* device, const std::string& name, OpIOs inputs,
            std::vector<int> shape)
            : OpBase(device, name, inputs), m_target_shape(shape) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }

    void deduce_output_shape() override {
        size_t len = inputs()[0]->length();
        std::vector<size_t> out_shape;
        out_shape.resize(m_target_shape.size());
        int count = 0;
        for (size_t i = 0; i < m_target_shape.size(); i++) {
            if (m_target_shape[i] != -1) {
                out_shape[i] = m_target_shape[i];
                INFER_ASSERT(len % m_target_shape[i] == 0, "Reshape error.\n");
                len = len / m_target_shape[i];
            } else {
                count++;
            }
        }
        INFER_ASSERT(count == 1, "multi -1 in Reshape param.\n");
        for (size_t i = 0; i < m_target_shape.size(); i++) {
            if (m_target_shape[i] == -1) {
                out_shape[i] = len;
            }
        }
        outputs()[0]->set_shape(out_shape, inputs()[0]->dtype());
    }

private:
    std::vector<int> m_target_shape;
};


/* ----------------------------- Elementwise 算子 ----------------------------- */

class Elemwise : public OpBase {
public:
    Elemwise(Device* device, const std::string& name, OpIOs inputs, ElemMode mode, float scale = -INFINITY)
        : OpBase(device, name, inputs)
        , m_mode(mode)
        , m_scale(scale) 
    {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    
    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    float m_scale;
    ElemMode m_mode;
};


class SpliteHalfActiveMul : public OpBase {
public:
    SpliteHalfActiveMul(
            Device* device, const std::string& name, OpIOs inputs, ElemMode mode)
            : OpBase(device, name, inputs), m_mode(mode) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;

    void deduce_output_shape() override {
        auto input_shape = inputs()[0]->shape();
        input_shape[1] = input_shape[1] / 2;
        outputs()[0]->set_shape(input_shape, inputs()[0]->dtype());
    }

private:
    ElemMode m_mode;
};

class DiagMask : public OpBase {
public:
    DiagMask(Device* device, const std::string& name, OpIOs inputs)
            : OpBase(device, name, inputs) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;
};


/* ------------------------------ Attention 算子 ------------------------------ */

//! Attention with cached the KvStorage, and output the kv with cache, and q
//! a*(wq, wk, qv) = q, k, v
//! out = softmax(q*k)*v
class AttentionBase : public OpBase {
public:
    AttentionBase(Device* device, const std::string& name, OpIOs inputs)
        : OpBase(device, name, inputs) {}
    
    AttentionBase(
            Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
            uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
            bool fused_weights = false, bool bias = false);

    void pre_execute() override;

    virtual void execute(WorkSpace* workspace, uint32_t nr_past) override = 0;

    void end_execute() override;

    size_t get_workspace_in_byte() override;

    void reset_ctx() {
        m_kstorage->reset_id();
        m_vstorage->reset_id();
    }

    virtual bool need_preprocess_weight(Tensor* weight) override;

    virtual std::vector<size_t> preprocess_weight(Tensor* tensor, void* src, void* dst) override;

protected:
    uint32_t m_embd;
    uint32_t m_head;
    uint32_t m_ctx;
    uint32_t m_layer_id;
    bool m_fused_weights;
    bool m_bias;
    bool m_packed_weight = false;

    std::unique_ptr<KvStorage> m_kstorage;
    std::unique_ptr<KvStorage> m_vstorage;
};

class LlamaAttention : public AttentionBase {
public:
    LlamaAttention(
            Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
            uint32_t rot, uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
            DType compt_type, bool fused_weights = false, bool bias = false,
            RotMode rotary_mode = RotMode::Mode0);

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    uint32_t m_rot;
    RotMode m_rotary_mode;
};

class GlmAttention : public AttentionBase {
public:
    GlmAttention(
            Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
            uint32_t rot, uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
            DType compt_type, bool fused_weights = false, bool bias = false,
            RotMode rotary_mode = RotMode::Mode0);

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    uint32_t m_gmask_position;
    RotMode m_rotary_mode;
};

class Glm2MultiQueryAttention : public AttentionBase {
public:
    Glm2MultiQueryAttention(
            Device* device, const std::string& name, OpIOs inputs, uint32_t embd,
            uint32_t query_group_num, uint32_t nr_ctx, uint32_t head, uint32_t layer_id,
            DType compt_type, bool fused_weights = false, bool bias = false,
            RotMode rotary_mode = RotMode::Mode0);

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    uint32_t m_query_group_num;
};




}  // namespace inferllm
