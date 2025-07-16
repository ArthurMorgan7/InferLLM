#pragma once

#include <unordered_map>
#include "kvstorage.h"
#include "op.h"
#include "tensor.h"
#include "kernel/kernel_define.h"

namespace inferllm {

class Graph;

struct UserConfig {
    DType compt_type;
};


/* ---------------------------------- 基础模块 ---------------------------------- */

class OprModuleBase {
public:
    OprModuleBase(std::shared_ptr<Tensor> input, Device* device, const std::string& name);
    OprModuleBase(std::vector<std::shared_ptr<Tensor>> inputs, Device* device, const std::string& name);

    // 访问
    std::shared_ptr<Tensor> input(int id = 0) const { return m_inputs[id]; };
    std::shared_ptr<Tensor> output() const { return m_output; };
    std::vector<std::shared_ptr<Tensor>> inputs() const { return m_inputs; };
    std::string name() const { return m_name; }
    Device* device() const { return m_device; }
    std::vector<std::shared_ptr<OpBase>>& oprs() { return m_oprs; }
    std::vector<std::shared_ptr<Tensor>> get_all_weights();

    // 设置
    void set_input(std::shared_ptr<Tensor> input) { m_inputs.push_back(input); };
    void set_output(std::shared_ptr<Tensor> output) { m_output = output; };


    virtual void execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill = false);
    virtual void reset_ctx() {}
    
    void deduce_output_shape();
    size_t get_workspace_in_byte();

    template <typename Op, typename... Args>
    std::vector<std::shared_ptr<Tensor>> add_opr(Args&&... args) {
        auto opr = std::make_shared<Op>(std::forward<Args>(args)...);
        m_oprs.push_back(opr);
        return opr->outputs();
    }


private:
    std::string m_name;     // 模块名称
    Device* m_device;       // 计算设备
    std::vector<std::shared_ptr<Tensor>> m_inputs;  // 输入
    std::shared_ptr<Tensor> m_output;   // 输出
    std::vector<std::shared_ptr<OpBase>> m_oprs;
};



/* ---------------------------------- 通用单算子模块 --------------------------------- */
template <class Op>
class OneOpModule : public OprModuleBase {
public:
    OneOpModule(Graph* graph, const std::vector<std::shared_ptr<Tensor>>& inputs,Device* device, const std::string& name)
        : OprModuleBase(inputs, device, name)
        , m_graph(graph) {}

    template <typename... Args>
    std::shared_ptr<Tensor> add_opr(Args&&... args) {
        auto opr = std::make_shared<Op>(device(), name(), inputs(), std::forward<Args>(args)...);
        oprs().push_back(opr);
        set_output(opr->outputs()[0]);
        return opr->outputs()[0];
    }

private:
    Graph* m_graph;
};



/* -------------------------------- Embedding 模块 ------------------------------- */

class EmbdModule : public OprModuleBase {
public:
    EmbdModule( Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
                UserConfig model_config, Device* device, const std::string& name);

private:
    uint32_t m_embd;
    uint32_t m_vocab;
    Graph* m_graph;
};




/* -------------------------------- Attention 模块------------------------------- */

template <typename Attention>
class AttentionModule : public OprModuleBase {
public:
    AttentionModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t head,
        uint32_t n_rot, uint32_t n_ctx, UserConfig model_config, Device* device,
        const std::string& name, int layer_id, bool fused_weights = false,
        bool bias = false, RotMode rotary_mode = RotMode::Mode0, bool same_bias = true)
        : OprModuleBase(input, device, name)
        , m_embd(embd)
        , m_head(head)
        , m_rot(n_rot)
        , m_graph(graph) 
    {
        INFER_ASSERT(embd % head == 0, "Embedding and head is not match.");

        //! 算子①：kqv-matmul
        m_attention_op = std::make_shared<Attention>(
                device, name, OpIOs{input}, embd, n_rot, n_ctx, head, layer_id,
                model_config.compt_type, fused_weights, bias, rotary_mode);
        oprs().push_back(m_attention_op);   // 加入 m_oprs
        auto v_out = m_attention_op->outputs()[0];

        //! 算子②：matmul proj
        bool proj_bias = same_bias ? bias : !bias;
        auto proj_out = add_opr<MatMul>(device, name + ".wo", OpIOs{v_out}, std::vector<size_t>{embd, embd},proj_bias)[0];
        
        // 设置模块输出
        set_output(proj_out);
    }

    void reset_ctx() override { m_attention_op->reset_ctx(); }

private:
    uint32_t m_embd;
    uint32_t m_head;
    uint32_t m_rot;
    Graph* m_graph;

    std::shared_ptr<Attention> m_attention_op;
};

/* ----------------------------------- FFN 模块 ---------------------------------- */

class LlamaFFNModule : public OprModuleBase {
public:
    LlamaFFNModule(
            Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
            UserConfig model_config, Device* device, const std::string& name);

private:
    uint32_t m_embd;
    Graph* m_graph;
};



class GlmFFNModule : public OprModuleBase {
public:
    GlmFFNModule(
            Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
            UserConfig model_config, Device* device, const std::string& name);

private:
    uint32_t m_embd;
    Graph* m_graph;
};



class Glm2FFNModule : public OprModuleBase {
public:
    Glm2FFNModule(
            Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
            UserConfig model_config, Device* device, const std::string& name);

private:
    uint32_t m_embd;
    Graph* m_graph;
};


/* ---------------------------------- Head 模块 ---------------------------------- */

class HeadModule : public OprModuleBase {
public:
    HeadModule(
            Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
            UserConfig model_config, Device* device, const std::string& name,
            bool bias = false, float eps = 1e-5);

    void execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill = false) override;

private:
    uint32_t m_embd;
    uint32_t m_vocab;
    Graph* m_graph;
};

}