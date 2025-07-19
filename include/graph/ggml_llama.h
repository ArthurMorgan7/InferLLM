#pragma once

#include <unordered_map>
#include "graph.h"
#include "kvstorage.h"
#include "op.h"
#include "tensor.h"

namespace inferllm {

enum class LlamaModelType {
    LLAMA_FILE_VERSION_GGML = 0,
    LLAMA_FILE_VERSION_GGMF_V1,  // added version field and scores in vocab
    LLAMA_FILE_VERSION_GGJT_V1
};

class GgmlLlamaGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;

    void load(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

};
}  // namespace inferllm