#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "kern/kernel_define.h"
#include "utils.h"

// clang-format off
#ifndef INFER_PAUSE
# if defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#   if !defined(__SSE2__)
      static inline void non_sse_mm_pause() { __asm__ __volatile__ ("rep; nop"); }
#     define _mm_pause non_sse_mm_pause
#   else
#       include <immintrin.h>
#   endif
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { _mm_pause(); } } while (0)
# elif defined __GNUC__ && defined __aarch64__
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("yield" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __arm__
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __riscv
// PAUSE HINT is not part of RISC-V ISA yet, but is under discussion now. For details see:
// https://github.com/riscv/riscv-isa-manual/pull/398
// https://github.com/riscv/riscv-isa-manual/issues/43
// #   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("pause"); } } while (0)
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("nop"); } } while (0)
# else
#   warning "Can't detect 'pause' (CPU-yield) instruction on the target platform. Specify INFER_PAUSE() definition via compiler flags."
#   define INFER_PAUSE(...) do { /* no-op: works, but not effective */ } while (0)
# endif
#endif // MTDA_PAUSE
// clang-format on

namespace inferllm {


/* -------------------------------------------------------------------------- */
/*                  对线程的封装，给线程附加更多的信息                            */
/* -------------------------------------------------------------------------- */
struct Worker {
public:
    Worker(std::function<void()>&& run) 
        : thread{run} {}

    ~Worker() { thread.join(); }

public:
    std::thread thread; // 线程本身
    std::atomic<bool> work_flag{false}; // 控制线程是否工作
};



/* -------------------------------------------------------------------------- */
/*                                    线程池类                                    */
/* -------------------------------------------------------------------------- */
class ThreadPool {
private:
    uint32_t m_nr_threads = 1;          // 线程数量
    uint32_t m_nr_task = 0;             // 任务数量
    uint32_t m_task_per_thread = 0;     // 每个线程的任务数量
    std::atomic_bool m_stop{false};     // 停止状态标志
    std::atomic_bool m_active{false};   // 运行状态标志
    std::vector<Worker*> m_workers;     // 池中的线程对象的指针的集合

    MultiThreadingTask m_task;          // 任务函数


    // 工具
    std::condition_variable m_cv;       // 条件变量
    std::mutex m_mutex;                 // 互斥量

    //! The number of iterations < main thread yeild resource>
    static constexpr int MAIN_THREAD_ACTIVE_WAIT = 10000;
    //! The number of iterations < worker thread yeild resource>
    static constexpr int WORKER_ACTIVE_WAIT = 2000;
    //! The number of iterations <pause>
    static constexpr int ACTIVE_WAIT_PAUSE_LIMIT = 16;


public:
    // 在构造函数中创建线程，来启动线程池
    ThreadPool(uint32_t nr_threads);
    
    // 回收线程，来销毁线程池
    ~ThreadPool();
    
    // 添加任务
    void add_task(const MultiThreadingTask& task, uint32_t nr_task);
    
    // 将线程池设为休眠状态
    void deactive();

    // 获取线程池的线程数量
    uint32_t nr_threads() const { return m_nr_threads; }

private:
    inline void sync();
    //! wake up all the threads from cv.wait(), when the thread pool is not
    //! active, all the threads will go to sleep.
    inline void active();
    
};

}  // namespace inferllm