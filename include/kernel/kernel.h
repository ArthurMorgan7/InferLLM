#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "thread_pool.h"
#include "kernel_define.h"
#include "utils.h"

#if INFER_X86
#include "kernel/optimized/x86/kernel.h"
#elif INFER_ARM
#include "kernel/optimized/arm/kernel.h"
#elif INFER_RVV
#include "kernel/optimized/rvv/kernel.h"
#else
#include "kernel/cpu/kernel.h"
#endif

#ifdef ENABLE_GPU
#include "kernel/gpu/kernel_gpu.h"
#endif

namespace inferllm {

class Kernel {
public:
    Kernel(KernelType kernel_type) 
        : m_kernel_type(kernel_type) 
    {}

    
    Kernel(KernelType kernel_type, ThreadPool* thread_pool)
        : m_kernel_type(kernel_type)
        , m_thread_pool(thread_pool) 
    {}

    uint32_t nr_thread() const {
        if (m_thread_pool == nullptr)
            return 1;
        return m_thread_pool->nr_threads();
    }

    bool supported_optimization(KernelOptMethod method) {
        if (m_kernel_type == KernelType::Arm || m_kernel_type == KernelType::Naive) {
            if (method == KernelOptMethod::MatmulInt4Reorder) {
#if defined(__ARM_FEATURE_DOTPROD)
                return true;
#else
                return false;
#endif
            }
            return false;
        }
        return false;
    }

    //! compute
    template <KernelID Id, typename... Args>
    void operator()(Args... args) {
        // GPU计算
        if (m_kernel_type == KernelType::GPU) {
#if ENABLE_GPU
            gpu::Comp<Id, Args...>::exec(std::forward<Args>(args)..., m_handle);
#endif
        } 

        // CPU 计算
        else {
            TaskSet task_set = opt::Comp<Id, Args...>::get_all_task(std::forward<Args>(args)...);
            for (auto& task : task_set) {
                // 交给线程池去计算
                m_thread_pool->add_task(task.first, task.second);
            }
        }
    }


    template <KernelID Id, typename... Args>
    size_t get_workspace(Args... args) {
        return opt::Space<Id, Args...>::get(std::forward<Args>(args)...);
    }


public:
    ThreadPool* m_thread_pool = nullptr;
    KernelType m_kernel_type;
#if ENABLE_GPU 
    void set_handle(cudaHandle* handle) { m_handle = handle; }
    cudaHandle* m_handle;
#endif
    };

}  // namespace inferllm
