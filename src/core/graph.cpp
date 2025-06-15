#include "graph.h"

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>


using namespace inferllm;


Graph::Graph(UserConfig model_config, Device* device, const std::string& name)
    : m_name(name)
    , m_model_config(model_config)
    , m_device(device) 
{
    m_workspace = make_unique<WorkSpace>();
}

Graph::~Graph() {
    if (m_workspace->ptr()) {
        m_device->free_device(m_workspace->ptr());
    }
}

/* ---------------------------------- 加载计算图 ---------------------------------- */

void Graph::load(std::shared_ptr<InputFile> fin, LlmParams& param,std::shared_ptr<Vocab> vocab) {
    // 验证模型头部的特殊字符 0x123456
    uint32_t magic;
    fin->read_raw((char*)&magic, sizeof(magic));
    INFER_ASSERT(magic == 0x123456, "model magic is not create!!!!");
    
    // 加载参数和词汇表(为model_imp对象的成员变量服务)
    load_param(fin, param, vocab);  // 虚函数

    // 构建模型
    construct_llm();                // 虚函数

    // 收集各模块权重
    collect_weights();

    // 设置映射表
    set_weights_alias();            // 虚函数


    size_t weight_length = 0;
    while (true) {
        if (fin->eof()) break;
   
        // 读取一个 tensor 的元数据
        int32_t n_dims;     // 维度
        int32_t length;     // ???
        int32_t ftype;      // 数据类型
        fin->read_raw(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        fin->read_raw(reinterpret_cast<char*>(&length), sizeof(length));
        fin->read_raw(reinterpret_cast<char*>(&ftype), sizeof(ftype));

        if (fin->eof())  break;

        // 读取每个维度的大小 
        size_t nr_number = 1;       // 一个Tensor的元素总量
        int32_t shape[2] = {1, 1};  // 一个Tensor的维度
        for (int i = 0; i < n_dims; ++i) {
            fin->read_raw(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
            nr_number *= shape[i];
        }

        // 读取权重名称
        std::string name(length, 0);
        fin->read_raw(&name[0], length);
        auto alias_name = get_weight_alias(name);   // 映射
        
        // 如果该模型中没有该权重，则跳过
        if (m_weights_map.count(alias_name) == 0) {
            INFER_LOG("skip weight %s\n", alias_name.c_str());
            auto dtype = convert_dtype(ftype);
            size_t length = nr_number * dtype_in_byte(dtype) / dtype_block_size(dtype);
            fin->skip(length);
            continue;
        }

        INFER_ASSERT(m_weights_map.count(alias_name) == 1,"Error weight is not found when loading.");
        auto weight = m_weights_map[alias_name];
        if (weight->length() != nr_number) {
            INFER_LOG("weight %s %zu is not match.\n", alias_name.c_str(), weight->length());
        }

        // 注册时的权重长度与文件中声明的不一致，报错
        INFER_ASSERT(weight->length() == nr_number, "Error length of weight is mismatch.");
        
        // 记录文件位置，设置数据类型，跳过实际数据
        // 没有马上加载数据，而是记录偏移以便后续读取
        weight->set_file(fin, fin->tell());     // 这一步就是载入权重
        weight->set_dtype(convert_dtype(ftype));
        fin->skip(weight->length_in_byte());
        weight_length += weight->length_in_byte();
    }
    INFER_LOG("total weight length = %lu\n", weight_length);
}

void Graph::collect_weights() {
    //! collect all the weights
    for (auto module : m_modules) {
        auto all_weights = module->get_all_weights();
        for (auto weight : all_weights) {
            std::string name = weight->name();
            INFER_ASSERT(m_weights_map.count(name) == 0, "dumplicated weight.");
            m_weights_map[name] = weight;
        }
    }
}

std::string Graph::get_weight_alias(const std::string& name) {
    std::regex reg_get("\\.(\\d+)\\.");
    std::smatch match;
    //! if find in map directly
    if (m_weights_name_aliases.find(name) != m_weights_name_aliases.end()) {
        return m_weights_name_aliases[name];
        //! if matmul "xxx.[layer_num].xxx"
    } else if (std::regex_search(name, match, reg_get)) {
        auto layer_num = match[1].str();
        std::regex reg_replace("\\.\\d+\\.");
        std::string reg_name = regex_replace(name, reg_replace, ".x.");
        //! if "aaa.x.bbbb" is found
        if (m_weights_name_aliases.find(reg_name) != m_weights_name_aliases.end()) {
            auto tmp_alias = m_weights_name_aliases[reg_name];
            //! replace "cccc.x.dddd" to "cccc.[layer_num].dddd"
            std::regex regx("\\.x\\.");
            return regex_replace(tmp_alias, regx, "." + layer_num + ".");
        } else {
            return name;
        }
        //! return origin
    } else {
        return name;
    }
}

DType Graph::convert_dtype(int32_t type) {
    switch (type) {
        case 0:
            return DType::Float32;
        case 1:
            return DType::Float16;
        case 2:
            return DType::Int4;
        case 3:
            return DType::Uint4;
        case 4:
            return DType::Int8;
        default:
            INFER_ASSERT(0, "unsupported weight type");
    }
};
/* ----------------------------------------------------------------------------------- */





/* ---------------------------------- 执行计算图 ---------------------------------- */
void Graph::execute(std::vector<int32_t> in_token, std::vector<float>& logist, uint32_t nr_past, bool prefill) {
    
    // 检查输入token序列的长度
    if (m_input->dims() == 0 || !same_input_shape(in_token)) {
        m_input->set_shape({in_token.size()}, DType::Int32);
        size_t len = get_workspace_in_byte();
        
        // 没有空间，直接申请
        if(m_workspace->ptr() == nullptr) {
            auto data = m_device->allocate(len);
            m_workspace->set_memory(data, len);
        }
        // 有空间但不够大，释放后申请
        else if (m_workspace->ptr() && len > m_workspace->length()) {
            m_device->free_device(m_workspace->ptr());
            auto data = m_device->allocate(len);
            m_workspace->set_memory(data, len);
        }
    }

    // 然后将输入 token 数据从主机（CPU）复制到设备（GPU/其他）
    m_input->resume_user_count();
    m_input->prepare_data();
    m_device->host2device_copy(m_input->ptr(), in_token.data(), in_token.size() * sizeof(int32_t), true);
    
    INFER_ASSERT(m_output->length() == logist.size(), "output length is not match with logist size");
    
    // 执行每个计算模块
    for (size_t i = 0; i < m_modules.size(); i++) {
        m_modules[i]->execute(m_workspace.get(), nr_past, prefill);
    }

    if (!prefill) {
        m_device->device2host_copy(logist.data(), m_output->ptr(), logist.size() * sizeof(float), true);
    }

    // 同步设备，回收数据
    m_device->sync();
    m_output->recall_data();
}



// 重置计算图中的上下文状态
void Graph::reset_ctx() {
    for (size_t i = 0; i < m_modules.size(); i++) {
        m_modules[i]->reset_ctx();
    }
}



bool Graph::same_input_shape(std::vector<int32_t> in_token) {
    INFER_ASSERT(m_input->dims() == 1, "input tensor should be one dim.");
    return m_input->shape()[0] == in_token.size();
}


size_t Graph::get_workspace_in_byte() {
    size_t max_workspace = 0;
    for (size_t i = 0; i < m_modules.size(); i++) {
        m_modules[i]->deduce_output_shape();
        size_t workspace = m_modules[i]->get_workspace_in_byte();
        max_workspace = workspace > max_workspace ? workspace : max_workspace;
    }
    return max_workspace;
}
/* ------------------------------------ - ----------------------------------- */

