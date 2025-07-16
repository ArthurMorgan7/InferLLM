#include "file.h"
#include "graph.h"
#include "model_imp.h"
#include "utils.h"

#include <algorithm>
#include <fstream>
#include <vector>


using namespace inferllm;

/* -------------------------------------------- 初始化 ----------------------------------------- */

ModelImp::ModelImp(const ModelConfig& config, const std::string& name)
    : m_name(name)
    , m_config(config) 
{
    // 读取配置参数
    uint32_t nr_thread = config.nr_thread;
    std::string device_type = config.device_type;


    // 创建算子调用接口类
    if (device_type == "CPU" || device_type == "cpu") {
        m_device = make_unique<CPUDevice>(KernelType::X86, nr_thread);
    } 
    else if (
            device_type == "GPU" || device_type == "CUDA" || device_type == "gpu") {
        // if compile with GPU, use GPU, else use CPUDevice
#if ENABLE_GPU
        m_device = make_unique<GPUDevice>(0);
#else
        INFER_ASSERT(0, "GPU is disabled when build, please build with GPU.");
#endif
    }

    // 创建计算图
    UserConfig user_config;
    user_config.compt_type = dtype_from_str(config.compt_type);
    m_graph = Graph::make_graph(user_config, m_device.get(), name);


    m_past = 0; 

    // 创建词表
    m_vocab = std::make_shared<Vocab>();
}

/* ---------------------------------- 模型加载及初始化 ---------------------------------- */

// 加载模型
void ModelImp::load(const std::string& model_path) {
    
    // 用类管理模型文件
    std::shared_ptr<InputFile> fin = std::make_shared<InputFile>(model_path, m_config.enable_mmap);

    m_param.n_ctx = m_config.nr_ctx;

    // 使用计算图加载参数
    m_graph->load(fin, m_param, m_vocab);

    // 给存储每个词出现概率的数组分配大小，等于词表的大小，因为每个词都要有概率
    m_logist.resize(m_param.n_vocab);  
}


// 进一步初始化（参数设置）
void ModelImp::init(uint32_t top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n, int32_t seed, int32_t end_token) {
    m_top_k = top_k;
    m_top_p = top_p;
    m_temp = temp;
    m_repeat_penalty = repeat_penalty;
    m_repeat_last_n = repeat_last_n;    // 滑动窗口的大小
    m_end_token = end_token;    // 终止的 token
    for (uint32_t i = 0; i < m_repeat_last_n; i++) {
        m_last_queue.push_back(0);
    }
    m_rng = std::mt19937(seed);
}


/* ------------------------------------------ decode ----------------------------------------- */

//! decode the user input sentence
std::string ModelImp::decode(const std::string& user_input, int& token) {
    // 字符编码为一个个 token
    auto tokens = tokenize(user_input, false);
    
    // 添加个引导头 64790, 64792
    m_graph->post_tokenize(tokens);


    for (auto token : tokens) {
        m_last_queue.push_back(token);
        m_last_queue.pop_front();
    }
    
    //auto start = m_timer.get_time();
    m_graph->execute(tokens, m_logist, m_past, false);
    //auto end = m_timer.get_time();
    //m_time_cost += end - start;
    
    sample_and_update();    // 找到概率最大的toekn加入m_last_queue，并作为 m_pre_token
    m_past += tokens.size();    // 在每次对话的第一次生成阶段，一次处理输入的所有 token
    
    // 找到生成的 token 对应的字符作为输出
    token = m_pre_token;
    return m_vocab->id_to_token[m_pre_token].tok;   
}



// 把字符串翻译成 最佳的 token Id 序列
#define MAX_TOKEN_LEN 18    // toekn 最长为18个字符
std::vector<Vocab::Id> ModelImp::tokenize(const std::string& text, bool bos) {
    std::vector<int> score;
    std::vector<Vocab::Id> prev;
    int len = text.length();

    score.resize(len + 1);  // 记录以第 i 个字符为结尾的最佳得分
    prev.resize(len + 1);   // 记录从哪个 token（其 ID）转移到第 i 位。

    // Forward pass
    for (int i = 0; i < len; i++) {
        // 从 i 开始，尝试所以长度的子串是否在词汇表中
        for (int sub_len = 1; sub_len <= len - i; sub_len++) {
            std::string sub = text.substr(i, sub_len);
            auto token = m_vocab->token_to_id.find(sub);

            // 在词表里有这个字符串
            if (token != m_vocab->token_to_id.end()) {
                int token_score = sub.length() * sub.length();  //  token 的长度平方作为 token 匹配得分
                int local_score = score[i] + token_score;   // 起始得分 + 子串得分 = 终点得分
                int next = i + sub_len;     // [i, i+sub_len-1] 的后一个位置
                if (local_score > score[next]) {
                    score[next] = local_score;
                    prev[next] = (*token).second;   // next 位置的前一个 token 的 Id
                }
            }
        }
    }

    // 从后往前回溯 token 序列，迭代取以当前字符为结尾的得分最高的token，最后翻转一下
    std::vector<Vocab::Id> res;     // Id == uint32_t
    int i = len;
    while (i > 0) {
        Vocab::Id token_id = prev[i];   // 上一跳的token id
        if (token_id == 0) {
            // TODO: Return error or something more meaningful
            printf("failed to tokenize string!\n");
            break;
        }
        res.push_back(token_id);
        auto token = m_vocab->id_to_token[token_id].tok;    // 以 i 为结尾的得分最高的 token
        i -= token.length();
    }

    // 添加起始符号
    if (bos) {
        res.push_back(1);  // TODO: replace with vocab.bos
    }

    // Pieces are in reverse order so correct that
    std::reverse(res.begin(), res.end());

    // res 是最合适的词元分割方式
    return res;
}



/* ------------------------------------------ decode_iter ----------------------------------------- */

//! decode the user input sentence
std::string ModelImp::decode_iter(int& token) {
    auto start = m_timer.get_time();

    m_graph->execute({m_pre_token}, m_logist, m_past);

    auto end = m_timer.get_time();
    m_time_cost += end - start;


    sample_and_update();

    m_past++;   // 在补全阶段，一次处理一个 token
    token = m_pre_token;

    return m_vocab->id_to_token[m_pre_token].tok;
}


// 取出概率最大的那个token
int32_t ModelImp::sample_and_update() {
    // sample the next token
   auto token = llama_sample_top_p_top_k(*m_vocab, m_logist.data(), m_last_queue, m_repeat_penalty, m_top_k, m_top_p, m_temp, m_rng);
    // update the last queue
    m_last_queue.push_back(token);
    m_last_queue.pop_front();
    m_pre_token = token;
    if (token == m_end_token) {
        m_device->deactive();
    }
    return token;
}









std::string ModelImp::decode_summary() const {
    std::string ret = "Run Model Summary:\n";
    ret += "Total Model Compute Time:   " + std::to_string(m_time_cost) + "s\n";
    ret += "Total Model Compute Token:  " + std::to_string(m_past) + "\n";
    ret += "Average Token Compute Time: " + std::to_string(m_time_cost * 1000 / m_past) + "ms\n";
    ret += "Average Token Generation Speed: " + std::to_string(m_past / m_time_cost) + "token/s\n";
    return ret;
}



// 暂时没看到有用
void ModelImp::prefill(const std::string& promote) {
    auto tokens = tokenize(promote, true);
    m_graph->post_tokenize(tokens);
    for (auto token : tokens) {
        m_last_queue.push_back(token);
        m_last_queue.pop_front();
    }
    //auto start = m_timer.get_time();
    m_graph->execute(tokens, m_logist, m_past, true);
    //auto end = m_timer.get_time();
    //m_time_cost += end - start;
    m_past = tokens.size();
}