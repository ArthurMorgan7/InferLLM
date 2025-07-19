#include "module.h"

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>

using namespace inferllm;

/* ---------------------------------- 模块基础 ---------------------------------- */

OprModuleBase::OprModuleBase(std::shared_ptr<Tensor> input, Device* device, const std::string& name)
        : m_name(name)
        , m_device(device) 
{
    m_inputs.push_back(input);
}

OprModuleBase::OprModuleBase(std::vector<std::shared_ptr<Tensor>> inputs, Device* device,const std::string& name)
    : m_name(name)
    , m_device(device)
    , m_inputs(inputs) 
{
}



void OprModuleBase::deduce_output_shape() {
    for (auto opr : m_oprs) {
        opr->deduce_output_shape();
    }
}


void OprModuleBase::execute(WorkSpace* workspace, uint32_t nr_past, bool) {
    for (auto opr : m_oprs) {


        opr->pre_execute();

#ifdef INFER_PROFILE
        struct timeval start, end;
        gettimeofday(&start, NULL);
#endif

        // 执行算子
        opr->execute(workspace, nr_past);   // 工作空间指针 + 已处理的序列长度



#ifdef INFER_PROFILE
        gettimeofday(&end, NULL);
        long seconds = end.tv_sec - start.tv_sec;
        float micros = (seconds * 1000) + (float)(end.tv_usec - start.tv_usec) / 1000;
        printf("Op %s spent time %f ms\n", opr->name().c_str(), micros);
#endif


        opr->end_execute();


    }
}

size_t OprModuleBase::get_workspace_in_byte() {
    size_t max_workspace = 0;
    for (auto opr : m_oprs) {
        size_t workspace = opr->get_workspace_in_byte();
        // 取最大值
        max_workspace = max_workspace < workspace ? workspace : max_workspace;
    }
    return max_workspace;
}

std::vector<std::shared_ptr<Tensor>> OprModuleBase::get_all_weights() {
    std::vector<std::shared_ptr<Tensor>> all_weights;
    for (auto opr : m_oprs) {
        auto weights = opr->weights();
        all_weights.insert(all_weights.end(), weights.begin(), weights.end());
    }
    return all_weights;
}


/* ------------------------------- Embedding 模块 ------------------------------ */

EmbdModule::EmbdModule(
    Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab, 
    UserConfig model_config, Device* device, const std::string& name)
    : OprModuleBase(input, device, name)
    , m_embd(embd)
    , m_graph(graph) 
{
    auto embd_out = add_opr<Embedding>(OpIOs{input}, embd, vocab, model_config.compt_type, device,"tok_embeddings")[0];
    set_output(embd_out);
}

/* -------------------------------- Attention 模块 ------------------------------- */

// 模板类的实现在源文件中



/* ---------------------------------- FFN 模块 --------------------------------- */

LlamaFFNModule::LlamaFFNModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    size_t nff = ((2 * (4 * embd) / 3 + mult - 1) / mult) * mult;
    //! matmul0
    auto matmul_out0 = add_opr<MatMul>(device, name + ".ffn.w3", OpIOs{input}, std::vector<size_t>{nff, embd})[0];
    //! matmul1
    auto matmul_out1 = add_opr<MatMul>(device, name + ".ffn.w1", OpIOs{input}, std::vector<size_t>{nff, embd})[0];
    //! silu activation
    auto silu_out = add_opr<Elemwise>(device, name + ".silu", OpIOs{matmul_out1}, ElemMode::Silu)[0];
    //! elemwise mul
    auto mul_out = add_opr<Elemwise>(device, name + ".elemwise", OpIOs{silu_out, matmul_out0}, ElemMode::Mul)[0];
    //! matmul2
    auto matmul_out2 = add_opr<MatMul>(device, name + ".ffn.w2", OpIOs{mul_out},std::vector<size_t>{embd, nff})[0];

    set_output(matmul_out2);
}

GlmFFNModule::GlmFFNModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    //! matmul0
    auto matmul_out1 = add_opr<MatMul>(
            device, name + ".ffn.matmul1", OpIOs{input},
            std::vector<size_t>{mult, embd}, true)[0];
    //! gelu activation
    auto gelu_out = add_opr<Elemwise>(
            device, name + ".gelu", OpIOs{matmul_out1}, ElemMode::Gelu)[0];
    //! matmul2
    auto matmul_out2 = add_opr<MatMul>(
            device, name + ".ffn.matmul2", OpIOs{gelu_out},
            std::vector<size_t>{embd, mult}, true)[0];
    set_output(matmul_out2);
}

Glm2FFNModule::Glm2FFNModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    //! matmul0
    auto matmul_out1 = add_opr<MatMul>(device, name + ".ffn.matmul1", OpIOs{input},std::vector<size_t>{mult * 2, embd}, false)[0];
    //! gelu activation
    auto gelu_out = add_opr<SpliteHalfActiveMul>(device, name + ".silu", OpIOs{matmul_out1}, ElemMode::Silu)[0];
    //! matmul2
    auto matmul_out2 = add_opr<MatMul>(device, name + ".ffn.matmul2", OpIOs{gelu_out},std::vector<size_t>{embd, mult}, false)[0];
    
    set_output(matmul_out2);
}

/* --------------------------------- Head模块 --------------------------------- */

HeadModule::HeadModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
        UserConfig model_config, Device* device, const std::string& name, bool bias,float eps)
        : OprModuleBase(input, device, name)
        , m_embd(embd), m_graph(graph) 
{
    //! LayerNorm
    auto norm_out = add_opr<LayerNorm>(device, name + ".norm", OpIOs{input}, m_embd, true, bias, true, eps)[0];
    //! matmul
    auto matmul_out = add_opr<MatMulLast>(device, name + ".output", OpIOs{norm_out},std::vector<size_t>{vocab, embd})[0];
    set_output(matmul_out);
}

void HeadModule::execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill) {
    //! prefill is no need to execute
    if (!is_prefill) {
        for (auto opr : oprs()) {
            opr->pre_execute();
            opr->execute(workspace, nr_past);
            opr->end_execute();
        }
    }
}

