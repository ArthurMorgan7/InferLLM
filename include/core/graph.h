#pragma once

#include <unordered_map>
#include "kvstorage.h"
#include "op.h"
#include "tensor.h"
#include "kernel/kernel_define.h"
#include "module.h"


namespace inferllm {


//! TODO: redefine this member of the struct
struct LlmParams {
    bool is_multi_query = false;    // 	是否使用 MQA机制，减小显存和计算开销。
    int32_t multi_query_group_num = 1;  // 开启MQA时，将所有 attention head 分组共享 key/value 的组数
    int32_t n_vocab;    // 词汇表大小
    int32_t n_embd;     // 每个 token 的 embedding 向量维度
    int32_t n_mult;     // MLP 层的倍数扩展因子
    int32_t n_head;     // 注意力头数量
    int32_t n_layer;    // Transformer block 的层数
    int32_t n_rot;      // rotary embedding 的维度，用于 RoPE（旋转位置编码）策略
    int32_t ftype;      // 权重使用的数据类型
    int32_t n_ctx;      // 模型支持的最大上下文长度
};


class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph(UserConfig model_config, Device* device, const std::string& name);

    virtual ~Graph();

    static std::shared_ptr<Graph> make_graph(UserConfig model_config, Device* device, const std::string& name);

    Device* device() { return m_device; }
    std::string name() { return m_name; }
    UserConfig model_config() { return m_model_config; }
    uint32_t get_nr_ctx() { return m_param.n_ctx; }
    uint32_t get_nr_vocab() { return m_param.n_vocab; }
    size_t get_workspace_in_byte();

    virtual void load(std::shared_ptr<InputFile> fin, LlmParams& param,std::shared_ptr<Vocab> vocab);
    virtual void load_param(std::shared_ptr<InputFile> fin, LlmParams& param,std::shared_ptr<Vocab> vocab) {}
    std::string get_weight_alias(const std::string& name);
    void collect_weights();
    virtual void construct_llm() = 0;
    virtual void set_weights_alias(){};




    // 构建并加入一个模块
    template <typename OpModule, typename... Args>
    std::shared_ptr<Tensor> add_module(Args&&... args) {
        auto module = std::make_shared<OpModule>(std::forward<Args>(args)...);
        m_modules.push_back(module);
        return module->output();
    }

    // 单算子模块
    template <typename Op>
    std::shared_ptr<OneOpModule<Op>> add_one_opr_module(
            Graph* graph, std::vector<std::shared_ptr<Tensor>> inputs, Device* device,const std::string& name) 
    {
        auto module = std::make_shared<OneOpModule<Op>>(graph, inputs, device, name);
        m_modules.push_back(module);
        return module;
    }



    // 执行推理
    void execute(std::vector<int32_t> in_token, std::vector<float>& logist, uint32_t nr_past,bool prefill = false);


    // Token 预处理
    virtual void post_tokenize(std::vector<Vocab::Id>& input) {}
 
    static DType convert_dtype(int32_t type);

    bool same_input_shape(std::vector<int32_t> in_token);

    void reset_ctx();


public:
    std::shared_ptr<Tensor> m_input;    // 输入张量
    std::shared_ptr<Tensor> m_output;   // 输出张量
    std::unordered_map<std::string, std::shared_ptr<Tensor>> m_weights_map;
    std::unordered_map<std::string, std::string> m_weights_name_aliases;
    std::vector<std::shared_ptr<OprModuleBase>> m_modules;  // 所有的运算模块

    LlmParams m_param;      // 模型核心参数

private:
    std::string m_name;         // 图的名称
    UserConfig m_model_config;  // 模型的用户配置
    Device* m_device = nullptr; // 执行的设备

    std::shared_ptr<Tensor> m_embeddings;   // 嵌入张量
    std::unique_ptr<WorkSpace> m_workspace; // 临时量的缓存
};
}  // namespace inferllm
