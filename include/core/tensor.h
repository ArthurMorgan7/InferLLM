#pragma once

#include "device.h"
#include "file.h"
#include "utils.h"

namespace inferllm {

enum class DType {
    Float32 = 0,
    Float16 = 1,
    Float8 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    Uint8 = 6,
    Int4 = 7,
    Uint4 = 8,
    Int2 = 9,
};

float dtype_in_byte(DType dtype);

//! the data arrangement
uint32_t dtype_block_size(DType dtype);

class OpBase;


/* -------------------------------------------------------------------------- */
/*                                   Tensor                                   */
/* -------------------------------------------------------------------------- */
enum class TensorState {
    Own = 0,        // 数据以加载
    OutSide = 1,    // 数据未加载
};
//! the tensor memory is from three ways:
//! 1. the tensor is own the memory, allocate by itself
//! 2. the tensor memory is shared from outside, such as the input tensor,
//! output tensor
//! 3. the tensor memory is map from file, such as the weight tensor
class Tensor {
/* ---------------------------------- 成员函数 ---------------------------------- */
public:
    Tensor(Device* device, std::string name) : m_device(device), m_name(name) {
        m_state = TensorState::OutSide;
    }

    Tensor(std::vector<size_t> shape, DType dtype, Device* device) {
        m_device = device;
        set_shape(shape);
        set_dtype(dtype);
        m_state = TensorState::OutSide;
    }

    ~Tensor();

    /* ------------------------------- 获取成员变量的接口函数 ------------------------------ */
    std::vector<size_t> shape() const { return m_shape; }
    DType dtype() const { return m_dtype; }
    OpBase* owner_op() { return m_owner_op; }
    std::string name() { return m_name; }
    std::vector<size_t> stride() const { return m_stride; }
    bool is_own() const { return m_state == TensorState::Own; }
    uint32_t dims() { return m_dims; }
    size_t length() { return m_length; }
    Device* device() { return m_device; }
    bool shared() const { return m_shared; }
    int32_t get_curr_user_count() { return m_cur_count; };
    size_t length_in_byte() {
        //! TODO: assert the length is int
        //! uint4 and int4 data arrangement: 32 data as a blcok and share the
        //! same scale and zero
        return m_length * dtype_in_byte(m_dtype) / dtype_block_size(m_dtype);
    }
    void* ptr() {
        INFER_ASSERT(is_own(), "Tensor is OutSide the device, can't get the memory.");
        return m_data;
    }

    const void* ptr() const {
        INFER_ASSERT(is_own(), "Tensor is OutSide the device, can't get the memory.");
        return m_data;
    }
    template <typename T>
    T* ptr() {
        INFER_ASSERT(is_own(), "Tensor is OutSide the device, can't get the memory.");
        return static_cast<T*>(m_data);
    }
    

    /* -------------------------------- 设置成员变量的接口 ------------------------------- */
    void set_owner_op(OpBase* owner_op) { m_owner_op = owner_op; }
    void set_name(const std::string& name) { m_name = name; }
    void set_dtype(DType dtype) { m_dtype = dtype; }
    void set_shape(std::vector<size_t> shape);
    void set_shape(std::vector<size_t> shape, DType dtype);
    void set_file(std::shared_ptr<InputFile> file, size_t offset);
    virtual void set_shared_memory(void* data, size_t length = 0);

    

    /* ----------------------------------- 其他 ----------------------------------- */

    int32_t add_user(); // 增加引用计数
    int32_t resume_user_count();    // 恢复使用者计数到初始值
    int32_t decrease_curr_user_count(); // 减少引用计数
    TensorState recall_data();
    virtual TensorState prepare_data(); // 确保张量数据准备好，可供使用
    size_t read_data_from_file();   // 从文件读取张量数据
    void preprocess_data();     // 数据预处理


/* ---------------------------------- 成员变量 ---------------------------------- */
private:
    bool m_shared = false;      // 是否使用共享内存
    int32_t m_usr_count = 0;    // 总使用者计数
    int32_t m_cur_count = 0;    // 当前剩余活跃着使用者数量

    Device* m_device;           // 所属设备
    OpBase* m_owner_op;         // 张量的操作

    TensorState m_state;        // 当前状态
    std::shared_ptr<InputFile> m_file;  // 对应文件的数据 mmap
    size_t m_file_offset = 0;           // 在文件中的起始地址

    uint32_t m_dims = 0;        // 张量维度
    size_t m_length = 0;        // 元素总个数
    DType m_dtype;              // 数据类型
    std::vector<size_t> m_shape;// 张量形状
    std::vector<size_t> m_stride;// 张量步长
    void* m_data = nullptr;     // 数据地址
    std::string m_name;         // 张量名称
};


/* -------------------------------------------------------------------------- */
/*                                  WorkSpace                                 */
/* -------------------------------------------------------------------------- */


class WorkSpace {
public:
    void* ptr() { return m_data; };

    template <typename T>
    T* ptr() {
        return static_cast<T*>(m_data);
    }

    size_t length() { return m_length; }

    void set_memory(void* data, size_t length) {
        m_data = data;
        m_length = length;
    }

private:
    void* m_data = nullptr;
    size_t m_length = 0;
};

}  // namespace inferllm
