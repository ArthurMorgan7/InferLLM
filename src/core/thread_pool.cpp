#include "thread_pool.h"

using namespace inferllm;

ThreadPool::ThreadPool(uint32_t threads_num)
    : m_nr_threads(threads_num) // 线程池的线程数
    , m_stop{false}
    , m_active{false} 
{
    if (threads_num < 1) {
        m_nr_threads = 1;
    }
    if (m_nr_threads > 1) {
        auto system_cpu_count = std::thread::hardware_concurrency();
        if (m_nr_threads > system_cpu_count) {
            INFER_LOG(
                    "The number of threads is bigger than number of "
                    "physical cpu cores, got: %d core_number: %d",
                    system_cpu_count, nr_threads());
        }
        // 为每一个线程安排任务
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers.push_back( new Worker([this, i]() {
                // 不设置stop，每个线程就一直循环
                while (!m_stop) {
                    // 线程处于活动状态
                    while (m_active) {
                        //! 任务需要处理
                        if (m_workers[i]->work_flag.load(std::memory_order_acquire)) {
                            m_task(TaskId{
                                    i * m_task_per_thread,  // 开始
                                    std::min((i + 1) * m_task_per_thread, m_nr_task),   // 结束
                                    i}  // 线程ID
                                );    
                            //! Flag worker is finished
                            m_workers[i]->work_flag.store(false, std::memory_order_release);
                        }

                        //! 等待任务进入
                        for (int it = 0; it < WORKER_ACTIVE_WAIT; it++) {
                            if (m_workers[i]->work_flag.load( std::memory_order_acquire)) {
                                break;
                            }
                            if (it < ACTIVE_WAIT_PAUSE_LIMIT || (it & 1)) {
                                INFER_PAUSE(16);  // Spin lock's CPU-level yield
                            } 
                            else {
                                // Spin lock's OS-level yield
                                std::this_thread::yield();
                            }
                        }
                    }

                    // 刚开始，每个线程都跳过while循环，进入休眠状态
                    {   // 把锁限定在局部作用域
                        std::unique_lock<std::mutex> lock(m_mutex);
                        // 线程池没有停止，但线程没有工作
                        if (!m_stop && !m_active) {
                            // 线程进入休眠，直到线程池停止或者线程激活
                            m_cv.wait(lock, [this] { return m_stop || m_active; });
                        }
                    }
                }
            }));
        }
    }
}

ThreadPool::~ThreadPool() {
    
    // 为了智能锁声明的局部作用域
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop = true;
        m_active = false;
        m_cv.notify_all();
    }

    // 销毁每一个线程对象，
    for (auto& worker : m_workers) {
        delete worker;
    }
}


void ThreadPool::add_task(const MultiThreadingTask& task, uint32_t nr_task) {
    
    // 如果只有一个主线程，直接让主线程处理任务
    if (m_nr_threads == 1 || nr_task == 1) {

        task({0, nr_task, m_nr_threads - 1});
        return;
    } 
    else {
        // 唤醒线程池
        active();
        INFER_ASSERT(m_active, "thread pool is not actived.");

        //! Set the task number, task iter and task
        m_nr_task = nr_task;
        m_task_per_thread = (nr_task + m_nr_threads - 1) / m_nr_threads;

        m_task = std::move(task);


        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            // work_flag 为 true，表明任务需要被处理
            m_workers[i]->work_flag.store(true, std::memory_order_release);
        }

        // 主线程也需要工作，这是计算主线程的 start
        uint32_t start = (m_nr_threads - 1) * m_task_per_thread;
        m_task({start, nr_task, m_nr_threads - 1});


        // 同步，等待所有线程处理完
        sync();
    }
}

inline void ThreadPool::sync() {
    bool no_finished = false;
    uint32_t no_finished_id = 0;
    do {
        no_finished = false;
        for (uint32_t i = no_finished_id; i < m_nr_threads - 1; ++i) {
            if (m_workers[i]->work_flag.load(std::memory_order_acquire)) {
                no_finished = true;
                no_finished_id = i;
                break;
            }
        }
        if (no_finished) {
            for (int it = 0; it < MAIN_THREAD_ACTIVE_WAIT; it++) {
                if (!m_workers[no_finished_id]->work_flag.load(
                            std::memory_order_acquire)) {
                    break;
                }
                if ((it < ACTIVE_WAIT_PAUSE_LIMIT || (it & 1))) {
                    INFER_PAUSE(16);
                } else {
                    std::this_thread::yield();
                }
            }
        }
    } while (no_finished);
}

inline void ThreadPool::active() {
    // 如果线程处于休眠状态，才进行唤醒
    // TODO 对于已经唤醒的线程， 再 notify_all() 会发生什么？
    if (!m_active) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_active = true;
        m_cv.notify_all();
    }
}

// 
void ThreadPool::deactive() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_active = false;
}
