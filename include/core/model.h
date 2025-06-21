#pragma once

#include <memory>
#include <string>

// 只向外提供最简单的接口，隐藏内部实现

#define API __attribute__((visibility("default")))

namespace inferllm {

struct ModelConfig {
    //! dtype include 'float32','float16','int8','int4'
    std::string compt_type = "float32";
    //! device_type include 'cpu','gpu'
    std::string device_type = "cpu";
    uint32_t nr_thread;
    uint32_t nr_ctx;
    int32_t device_id;
    bool enable_mmap;
};

class ModelImp;

class API Model {
private:
    std::shared_ptr<ModelImp> m_model_imp;

public:
    //! create a model by the model_name, the model_name must be registered
    //! internal before load it
    Model(const ModelConfig& config, const std::string& model_name);

    //! load the model from model_path
    void load(const std::string& model_path);

    //! allocate memory for the model or init its param
    void init(uint32_t top_k, float top_p, float temp, float repeat_penalty,
              int repeat_last_n, int32_t seed, int32_t end_token);

    //! get the remain token number
    uint32_t get_remain_token();

    //! reset the token
    void reset_token();

    //! prefill the model with inference with the given promote
    void prefill(const std::string& promote);

    //! decode the answer one by one
    std::string decode(const std::string& user_input);

    //! decode the answer one by one
    std::string decode_iter(int& token);

    std::string decode_summary() const;


};

}  // namespace inferllm