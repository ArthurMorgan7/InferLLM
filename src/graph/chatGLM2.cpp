#include "chatGLM.h"

using namespace inferllm;



// 加载二进制文件中的模型参数
void ChatGLMGraph2::load_param(std::shared_ptr<InputFile> fin, LlmParams& param,std::shared_ptr<Vocab> vocab) {
    Header header;
    // 加载魔数后的头部
    fin->read_raw((char*)&header.param_offset, sizeof(header.param_offset));
    fin->read_raw((char*)&header.param_length, sizeof(header.param_length));
    fin->read_raw((char*)&header.vocab_offset, sizeof(header.vocab_offset));
    fin->read_raw((char*)&header.vocab_length, sizeof(header.vocab_length));
    fin->read_raw((char*)&header.tensor_offset, sizeof(header.tensor_offset));

    // 把指针移动到参数起始点
    fin->seek(header.param_offset);
    fin->read_raw((char*)&param.n_embd, sizeof(param.n_embd));
    fin->read_raw((char*)&param.n_head, sizeof(param.n_head));
    fin->read_raw((char*)&param.n_layer, sizeof(param.n_layer));
    fin->read_raw((char*)&param.n_mult, sizeof(param.n_mult));
    fin->read_raw((char*)&param.n_vocab, sizeof(param.n_vocab));
    int32_t multi_query;
    fin->read_raw((char*)&multi_query, sizeof(multi_query));
    param.is_multi_query = multi_query > 0;
    fin->read_raw((char*)&param.multi_query_group_num, sizeof(param.multi_query_group_num));
    m_param = param;

    // 把指针移动到词汇表起始点
    fin->seek(header.vocab_offset);
    vocab->load_vocab(fin, param.n_vocab);

    // TODO 预留一段 token ID 范围给特殊用途？
    m_param.n_vocab = 65024;
    param.n_vocab = 65024;

    // 文件内指针移动到权重数据起始处
    fin->seek(header.tensor_offset);
}

// 构建计算图
void ChatGLMGraph2::construct_llm() {
    // 构建输入 Tensor
    m_input = std::make_shared<Tensor>(device(), name() + ":input");
    std::shared_ptr<Tensor> input = m_input;
    
    
    // Embedding 层，add_module相当于调用了EmbdModule的构造函数 
    input = add_module<EmbdModule>(this, input, m_param.n_embd, m_param.n_vocab, model_config(), device(), "");

    // input 是 embedding 后，对Transformer块的输入

    // Transformer 块 x N
    int nr_layer = m_param.n_layer;
    for (int i = 0; i < nr_layer; i++) {
        std::string name = "layers." + std::to_string(i);
        //! layer norm
        std::shared_ptr<Tensor> attention_input = input;
        auto norm_out_attention = add_one_opr_module<LayerNorm>(this, OpIOs{attention_input}, device(), name + ".attention_norm")
                ->add_opr(m_param.n_embd, /*mul*/ true, /*bias*/ false,/*rms*/ true);


        //! attentin
        auto attention_output = add_module<AttentionModule<Glm2MultiQueryAttention>>(
                this, norm_out_attention, m_param.n_embd, m_param.n_head,
                m_param.multi_query_group_num, m_param.n_ctx, model_config(), device(),
                name + ".attention", i, true /*fused_weights*/, true /*bias*/,
                RotMode::Mode0, false /*proj_bias*/);

        //! add  norm_out_attention * scale + attention_output
        auto add_output = add_one_opr_module<Elemwise>(this, OpIOs{attention_input, attention_output},device(), name + ".attention.Elemwise")
                            ->add_opr(ElemMode::Add);

        std::shared_ptr<Tensor> feed_forward_input = add_output;
        
        //! layer normal
        auto ffn_norm_out = add_one_opr_module<LayerNorm>(this, OpIOs{feed_forward_input}, device(), name + ".ffn_norm")
                ->add_opr(m_param.n_embd, /*mul*/ true, /*bias*/ false, /*rms*/ true);
        
        //! feed forward
        auto ffn_output = add_module<Glm2FFNModule>(this, ffn_norm_out, m_param.n_embd, m_param.n_mult, model_config(),device(), name);
        
        //! add ffn_norm_out * scale + ffn_output
        input = add_one_opr_module<Elemwise>(this, OpIOs{feed_forward_input, ffn_output}, device(), name + ".ffn.Elemwise")
                    ->add_opr(ElemMode::Add);
    }

    // 输出层
    m_output = add_module<HeadModule>(this, input, m_param.n_embd, m_param.n_vocab, model_config(), device(),"head");
}


void ChatGLMGraph2::set_weights_alias() {
    m_weights_name_aliases.clear();
    // clang-format off
    m_weights_name_aliases = {
            {"transformer.embedding.word_embeddings.weight", "tok_embeddings.weight"},
            {"transformer.encoder.layers.x.input_layernorm.weight", "layers.x.attention_norm.weight"},
            {"transformer.encoder.layers.x.self_attention.query_key_value.weight", "layers.x.attention.wqkv.weight"},
            {"transformer.encoder.layers.x.self_attention.query_key_value.bias", "layers.x.attention.wqkv.bias"},
            {"transformer.encoder.layers.x.self_attention.dense.weight", "layers.x.attention.wo.weight"},
            {"transformer.encoder.layers.x.post_attention_layernorm.weight", "layers.x.ffn_norm.weight"},
            {"transformer.encoder.layers.x.mlp.dense_h_to_4h.weight", "layers.x.ffn.matmul1.weight"},
            {"transformer.encoder.layers.x.mlp.dense_4h_to_h.weight", "layers.x.ffn.matmul2.weight"},
            {"transformer.encoder.final_layernorm.weight", "head.norm.weight"},
            {"transformer.output_layer.weight", "head.output.weight"},
    };
    // clang-format on
}

void ChatGLMGraph2::post_tokenize(std::vector<Vocab::Id>& input) {
    std::vector<Vocab::Id> res;
    res.push_back(64790);   // <sop>（start of prompt
    res.push_back(64792);   // <gmask> 或 <mask>，用于表示生成任务的起点
    // add a space in the head
    // 将 res 中的两个 token 插入到 input 的开头，形成新的输入序列
    input.insert(input.begin(), res.begin(), res.end());
}
