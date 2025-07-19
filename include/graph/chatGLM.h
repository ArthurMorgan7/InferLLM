#pragma once

#include <unordered_map>
#include "graph.h"
#include "kvstorage.h"
#include "op.h"
#include "tensor.h"

namespace inferllm {

// 模型文件各个部分的位置和长度
struct Header {
    int param_offset;   // 模型参数的起始地址
    int param_length;
    int vocab_offset;   // 词汇表的起始地址
    int vocab_length;
    int tensor_offset;  // tensor权重的起始地址
};

struct Param {
    int hidden_size;
    int n_heads;
    int n_layers;
    int embd_dim;
    int fc_hidden;
    int vacab_size;
    int multi_query = 0;
    int multi_query_group_num = 1;
};


class ChatGLMGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;
    void post_tokenize(std::vector<Vocab::Id>& input) override;
};


class ChatGLMGraph2 : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

    void post_tokenize(std::vector<Vocab::Id>& input) override;
};

class ChatGLMGraph3 : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

    void post_tokenize(std::vector<Vocab::Id>& input) override;
};
}  // namespace inferllm