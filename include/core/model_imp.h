#pragma once

#include "device.h"
#include "graph.h"
#include "kernel_define.h"
#include "model.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace inferllm {

namespace {

DType dtype_from_str(const std::string& str) {
    if (str == "float32" || str == "fp23") {
        return DType::Float32;
    } else if (str == "float16" || str == "fp16") {
        return DType::Float16;
    } else if (str == "int8") {
        return DType::Int8;
    } else if (str == "uint8") {
        return DType::Uint8;
    } else if (str == "int4") {
        return DType::Int4;
    } else if (str == "uint4") {
        return DType::Uint4;
    } else {
        INFER_ASSERT(0, "Unsupported dytpe.");
    }
}
}  // namespace

//! the implement of model
class ModelImp {
public:
    ModelImp(const ModelConfig& config, const std::string& name);
    
    //! load the model from model_path
    void load(const std::string& model_path);

    //! allocate memory for the model or init its param
    void init(uint32_t top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n, int32_t seed, int32_t end_token);

    //! prefill the model with inference with the given promote
    void prefill(const std::string& promote);

    //! decode the user input sentence
    std::string decode(const std::string& user_input, int& token);

    std::string decode_iter(int& token);

    uint32_t get_remain_token() { return m_graph->get_nr_ctx() - m_past; }

    void reset_token() {
        m_past = 0;
        m_graph->reset_ctx();
    }

    int32_t sample_and_update();

    std::string decode_summary() const;

private:
    std::vector<Vocab::Id> tokenize(const std::string& text, bool bos);

    
    uint32_t m_past = 0;    // 已经处理的 token 数

    uint32_t m_top_k;
    float m_top_p;
    float m_temp;
    float m_repeat_penalty;
    uint32_t m_repeat_last_n;
    int32_t m_end_token;

    int32_t m_pre_token;

    std::string m_name;
    LlmParams m_param;
    ModelConfig m_config;
    //! TODO: support run model on multi device later.
    std::unique_ptr<Device> m_device;
    std::shared_ptr<Graph> m_graph;
    std::shared_ptr<Vocab> m_vocab;
    std::list<int32_t> m_last_queue;
    std::vector<float> m_logist;

    std::mt19937 m_rng;
    Timer m_timer;
    double m_time_cost = 0;
};

}  // namespace inferllm
