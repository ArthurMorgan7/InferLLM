#include "chatGLM.h"

using namespace inferllm;

void ChatGLMGraph::set_weights_alias() {
    m_weights_name_aliases.clear();
    // clang-format off
    m_weights_name_aliases = {
            {"transformer.word_embeddings.weight", "tok_embeddings.weight"},
            {"transformer.layers.x.input_layernorm.weight", "layers.x.attention_norm.weight"},
            {"transformer.layers.x.input_layernorm.bias", "layers.x.attention_norm.bias"},
            {"transformer.layers.x.attention.rotary_emb.inv_freq", "layers.x.attention.rotary.inv_freq"},
            {"transformer.layers.x.attention.query_key_value.weight", "layers.x.attention.wqkv.weight"},
            {"transformer.layers.x.attention.query_key_value.bias", "layers.x.attention.wqkv.bias"},
            {"transformer.layers.x.attention.dense.weight", "layers.x.attention.wo.weight"},
            {"transformer.layers.x.attention.dense.bias", "layers.x.attention.wo.bias"},
            {"transformer.layers.x.post_attention_layernorm.weight", "layers.x.ffn_norm.weight"},
            {"transformer.layers.x.post_attention_layernorm.bias", "layers.x.ffn_norm.bias"},
            {"transformer.layers.x.mlp.dense_h_to_4h.weight", "layers.x.ffn.matmul1.weight"},
            {"transformer.layers.x.mlp.dense_h_to_4h.bias", "layers.x.ffn.matmul1.bias"},
            {"transformer.layers.x.mlp.dense_4h_to_h.weight", "layers.x.ffn.matmul2.weight"},
            {"transformer.layers.x.mlp.dense_4h_to_h.bias", "layers.x.ffn.matmul2.bias"},
            {"transformer.final_layernorm.weight", "head.norm.weight"},
            {"transformer.final_layernorm.bias", "head.norm.bias"},
            {"lm_head.weight", "head.output.weight"},
            {"lm_head.bias", "head.output.bias"},
    };
    // clang-format on
}

//! LlamaGraph
void ChatGLMGraph::load_param(
        std::shared_ptr<InputFile> fin, LlmParams& param,
        std::shared_ptr<Vocab> vocab) {

    Header header;
    // load model header
    fin->read_raw((char*)&header.param_offset, sizeof(header.param_offset));
    fin->read_raw((char*)&header.param_length, sizeof(header.param_length));
    fin->read_raw((char*)&header.vocab_offset, sizeof(header.vocab_offset));
    fin->read_raw((char*)&header.vocab_length, sizeof(header.vocab_length));
    fin->read_raw((char*)&header.tensor_offset, sizeof(header.tensor_offset));

    fin->seek(header.param_offset);
    // load param
    fin->read_raw((char*)&param.n_embd, sizeof(param.n_embd));
    fin->read_raw((char*)&param.n_head, sizeof(param.n_head));
    fin->read_raw((char*)&param.n_layer, sizeof(param.n_layer));
    fin->read_raw((char*)&param.n_mult, sizeof(param.n_mult));
    fin->read_raw((char*)&param.n_vocab, sizeof(param.n_vocab));
    m_param = param;

    // load vocab
    fin->seek(header.vocab_offset);
    vocab->load_vocab(fin, param.n_vocab);

    // create the graph
    m_param.n_vocab = 130528;
    param.n_vocab = 130528;
    fin->seek(header.tensor_offset);
}

void ChatGLMGraph::construct_llm() {
    m_input = std::make_shared<Tensor>(device(), name() + ":input");
    std::shared_ptr<Tensor> input = m_input;
    
    //! embd
    input = add_module<EmbdModule>(
            this, input, m_param.n_embd, m_param.n_vocab, model_config(), device(), "");

    int nr_layer = m_param.n_layer;
    float scale = sqrt(2 * nr_layer);
    for (int i = 0; i < nr_layer; i++) {
        std::string name = "layers." + std::to_string(i);
        //! layer norm
        std::shared_ptr<Tensor> attention_input = input;
        auto norm_out_attention =
                add_one_opr_module<LayerNorm>(
                        this, OpIOs{attention_input}, device(),
                        name + ".attention_norm")
                        ->add_opr(
                                m_param.n_embd, /*mul*/ true, /*bias*/ true,
                                /*rms*/ false);
        //! attentin
        auto attention_output = add_module<AttentionModule<GlmAttention>>(
                this, norm_out_attention, m_param.n_embd, m_param.n_head, m_param.n_rot,
                m_param.n_ctx, model_config(), device(), name + ".attention", i,
                true /*fused_weights*/, true /*bias*/);
        //! add  norm_out_attention * scale + attention_output
        auto add_output = add_one_opr_module<Elemwise>(
                                  this, OpIOs{norm_out_attention, attention_output},
                                  device(), name + ".attention.Elemwise")
                                  ->add_opr(ElemMode::Add, scale);

        std::shared_ptr<Tensor> feed_forward_input = add_output;
        //! layer normal
        auto ffn_norm_out =
                add_one_opr_module<LayerNorm>(
                        this, OpIOs{feed_forward_input}, device(), name + ".ffn_norm")
                        ->add_opr(
                                m_param.n_embd, /*mul*/ true, /*bias*/ true,
                                /*rms*/ false);
        //! feed forward
        auto ffn_output = add_module<GlmFFNModule>(
                this, ffn_norm_out, m_param.n_embd, m_param.n_mult, model_config(),
                device(), name);
        //! add ffn_norm_out * scale + ffn_output
        input = add_one_opr_module<Elemwise>(
                        this, OpIOs{ffn_norm_out, ffn_output}, device(),
                        name + ".ffn.Elemwise")
                        ->add_opr(ElemMode::Add, scale);
    }
    //! the last layer
    m_output = add_module<HeadModule>(
            this, input, m_param.n_embd, m_param.n_vocab, model_config(), device(),
            "head", true);
}

void ChatGLMGraph::post_tokenize(std::vector<Vocab::Id>& input) {
    //! the begin token
    input.insert(input.begin(), 5);
    //! the end token
    input.push_back(130001);
    input.push_back(130004);
}
