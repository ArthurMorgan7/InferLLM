#include "tensor.h"
#include "kernel_define.h"
#include "memory.h"
#include "utils.h"
#include "op.h"

using namespace inferllm;

float inferllm::dtype_in_byte(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
            return 4;
        case DType::Float16:
        case DType::Int16:
            return 2;
        case DType::Float8:
        case DType::Uint8:
            return 1;
        case DType::Int8:
            return sizeof(BlockQ80);
        case DType::Int4:
            //! QK number int4 as a block, and share a float scale
            return sizeof(BlockQ40);
        default:
            INFER_ASSERT(0, "No support data type.");
    }
}

uint32_t inferllm::dtype_block_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
        case DType::Float16:
        case DType::Int16:
        case DType::Float8:
        case DType::Uint8:
            return 1;
        case DType::Int8:
            return QK80;
        case DType::Int4:
            return QK40;
        default:
            INFER_ASSERT(0, "No support data type.");
    }
}

TensorState Tensor::prepare_data() {
    // 计算张量所占的内存大小
    size_t length = length_in_byte();

    // 数据未加载
    if (!m_data && m_state == TensorState::OutSide) {
        // 数据映射自文件，通常是权重张量
        if (m_file) {
            read_data_from_file();
        } 
        // 需要动态分配，通常是输入输出张量
        else {
            m_data = m_device->allocate(length);
        }
    }

    // 数据已加载，并返回状态
    m_state = TensorState::Own;
    return m_state;
}

TensorState Tensor::recall_data() {
    if (m_shared) {
        return m_state;
    }
    //! if the tensor data is from allocate by itself, we need free the memory
    if (!m_file && m_data != nullptr && m_state == TensorState::Own) {
        m_device->free_device(m_data);
        m_data = nullptr;
    }
    m_state = TensorState::OutSide;
    return m_state;
}

// 从文件加载数据到张量
size_t Tensor::read_data_from_file() {
    size_t length = length_in_byte();
    // 使用 mmap
    if (m_file->enable_mmap()) {
        // 无统一内存时，需将数据从映射内存复制到设备内存
        if (!m_device->unified_memory()) {
            auto temp_ptr = m_file->get_mmap_data(length, m_file_offset);
            m_data = m_device->allocate(length);
            m_device->host2device_copy(m_data, temp_ptr, length);
        } 
        else {
            // 统一内存可直接使用映射地址
            m_data = m_file->get_mmap_data(length, m_file_offset);
        }
    } 
    // 使用IO
    else if (m_data == nullptr) {
        // 无统一内存时的处理
        if (!m_device->unified_memory()) {
            m_data = m_device->allocate(length);
            auto host_ptr = m_device->allocate_host(length);
            auto opr = this->owner_op();
            
            // 检查是否需要预处理权重
            if (opr->need_preprocess_weight(this)) {
                auto host_ptr2 = m_device->allocate_host(length);
                m_file->read_data(host_ptr2, length, m_file_offset);
                auto shape = opr->preprocess_weight(this, host_ptr2, host_ptr);
                set_shape(shape);
                m_device->free_host(host_ptr2);
            } 
            
            else {
                m_file->read_data(host_ptr, length, m_file_offset);
            }
            m_device->host2device_copy(m_data, host_ptr, length);
            m_device->free_host(host_ptr);
        } 
        // 统一内存时的处理
        else {
            m_data = m_device->allocate(length);
            auto opr = this->owner_op();
            if (opr->need_preprocess_weight(this)) {
                auto host_data = m_device->allocate_host(length);
                m_file->read_data(host_data, length, m_file_offset);
                auto shape = opr->preprocess_weight(this, host_data, m_data);
                set_shape(shape);
                m_device->free_host(host_data);
            } else {
                m_file->read_data(m_data, length, m_file_offset);
            }
        }
    }
    return length;
}


void Tensor::set_shared_memory(void* data, size_t size) {
    INFER_ASSERT(
            data == nullptr || size >= length_in_byte(),
            "the memory set to tensor is not enough");
    m_data = data;
    m_state = TensorState::Own;
    m_shared = true;
}

void Tensor::set_shape(std::vector<size_t> shape) {
    m_dims = shape.size();
    m_shape = shape;
    //! init the tensor as continue tensor
    m_stride.resize(m_dims);
    m_stride[m_dims - 1] = 1;
    for (uint32_t i = 1; i < m_dims; i++) {
        m_stride[m_dims - 1 - i] = m_stride[m_dims - i] * m_shape[m_dims - i];
    }
    m_length = m_shape[0] * m_stride[0];
}

void Tensor::set_shape(std::vector<size_t> shape, DType dtype) {
    set_shape(shape);
    set_dtype(dtype);
}

void Tensor::set_file(std::shared_ptr<InputFile> file, size_t offset) {
    m_state = TensorState::OutSide;
    m_file = file;
    m_file_offset = offset;
}


Tensor::~Tensor() {
    if (m_state == TensorState::Own) {
        recall_data();
    }
    //! the data read from file by m_file->read_data
    if (m_file && !m_file->enable_mmap() && m_data) {
        m_device->free_device(m_data);
    }
}


    // 增加引用计数
int32_t Tensor::add_user() {
    m_usr_count++;
    return m_usr_count;
}
// 恢复使用者计数到初始值
int32_t Tensor::resume_user_count() {
    m_cur_count = m_usr_count;
    return m_cur_count;
}


// 减少引用计数
int32_t Tensor::decrease_curr_user_count() {
    if (!m_shared) {
        INFER_ASSERT(m_cur_count > 0, "The user count is less than 0.");
        m_cur_count--;
        if (m_cur_count == 0) {
            recall_data();  // 释放数据
        }
    }
    return m_cur_count;
};