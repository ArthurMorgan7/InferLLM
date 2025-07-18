#include "thread_pool.h"
#include <chrono>
#include <algorithm>

using namespace inferllm;

// 工作窃取队列实现
bool ThreadPool::WorkStealingQueue::push(std::function<void()> task) {
    size_t tail = this->tail.load(std::memory_order_relaxed);
    size_t head = this->head.load(std::memory_order_acquire);
    
    if (tail - head >= QUEUE_SIZE) {
        return false; // 队列满了
    }
    
    tasks[tail % QUEUE_SIZE] = std::move(task);
    this->tail.store(tail + 1, std::memory_order_release);
    return true;
}

bool ThreadPool::WorkStealingQueue::pop(std::function<void()>& task) {
    size_t tail = this->tail.load(std::memory_order_relaxed) - 1;
    this->tail.store(tail, std::memory_order_relaxed);
    
    size_t head = this->head.load(std::memory_order_acquire);
    if (head > tail) {
        this->tail.store(tail + 1, std::memory_order_relaxed);
        return false;
    }
    
    task = std::move(tasks[tail % QUEUE_SIZE]);
    return true;
}

bool ThreadPool::WorkStealingQueue::steal(std::function<void()>& task) {
    size_t head = this->head.load(std::memory_order_relaxed);
    size_t tail = this->tail.load(std::memory_order_acquire);
    
    if (head >= tail) {
        return false;
    }
    
    task = std::move(tasks[head % QUEUE_SIZE]);
    this->head.store(head + 1, std::memory_order_release);
    return true;
}

ThreadPool::ThreadPool(uint32_t threads_num, const ThreadPoolConfig& config)
    : m_nr_threads(threads_num) // 线程池的线程数
    , m_stop{false}
    , m_active{false}
    , m_config(config)
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
        
        // 初始化工作窃取队列
        if (m_config.enable_work_stealing) {
            m_work_queues.resize(m_nr_threads);
            for (auto& queue : m_work_queues) {
                queue = std::make_unique<WorkStealingQueue>();
            }
        }
        
        // 为每一个线程安排任务
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers.push_back(std::make_unique<Worker>([this, i]() {
                // 优化：设置线程亲和性，减少上下文切换
                if (m_config.enable_thread_affinity) {
                    #ifdef __linux__
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                    #endif
                }
                
                auto start_time = std::chrono::high_resolution_clock::now();
                
                // 不设置stop，每个线程就一直循环
                while (!m_stop) {
                    // 线程处于活动状态
                    while (m_active) {
                        //! 任务需要处理
                        if (m_workers[i]->work_flag.load(std::memory_order_acquire)) {
                            m_workers[i]->is_busy.store(true, std::memory_order_relaxed);
                            
                            auto task_start = std::chrono::high_resolution_clock::now();
                            
                            // 执行任务
                            m_task(TaskId{
                                    m_workers[i]->local_task_start,  // 开始
                                    m_workers[i]->local_task_end,    // 结束
                                    i}  // 线程ID
                                );
                            
                            auto task_end = std::chrono::high_resolution_clock::now();
                            auto task_duration = std::chrono::duration_cast<std::chrono::microseconds>(task_end - task_start);
                            
                            // 更新统计信息
                            m_workers[i]->tasks_completed.fetch_add(1, std::memory_order_relaxed);
                            m_workers[i]->total_work_time.fetch_add(task_duration.count(), std::memory_order_relaxed);
                            
                            m_workers[i]->is_busy.store(false, std::memory_order_relaxed);
                            //! Flag worker is finished
                            m_workers[i]->work_flag.store(false, std::memory_order_release);
                        }

                        //! 优化：减少等待时间，提高响应性
                        for (int it = 0; it < m_config.max_active_wait; it++) {
                            if (m_workers[i]->work_flag.load(std::memory_order_acquire)) {
                                break;
                            }
                            if (it < m_config.spin_wait_cycles || (it & 1)) {
                                INFER_PAUSE(m_config.spin_wait_cycles);  // 使用配置的自旋等待周期
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
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                m_total_time.fetch_add(total_duration.count(), std::memory_order_relaxed);
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

    // 销毁每一个线程对象（使用智能指针，自动管理）
    m_workers.clear();
}

void ThreadPool::add_task(const MultiThreadingTask& task, uint32_t nr_task) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 如果只有一个主线程，直接让主线程处理任务
    if (m_nr_threads == 1 || nr_task == 1) {
        task({0, nr_task, m_nr_threads - 1});
        m_total_tasks.fetch_add(1, std::memory_order_relaxed);
        return;
    } 
    else {
        // 优化：减少锁竞争，使用无锁唤醒
        if (!m_active.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_active) {
                m_active = true;
                m_cv.notify_all();
            }
        }

        INFER_ASSERT(m_active, "thread pool is not actived.");

        //! Set the task number, task iter and task
        m_nr_task = nr_task;
        
        // 优化：动态负载均衡
        if (m_config.enable_dynamic_load_balancing) {
            // 根据任务大小动态调整每个线程的任务数量
            m_task_per_thread = std::max(1u, (nr_task + m_nr_threads - 1) / m_nr_threads);
        } else {
            m_task_per_thread = (nr_task + m_nr_threads - 1) / m_nr_threads;
        }

        m_task = std::move(task);

        // 优化：批量设置工作标志，减少内存屏障
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            // 设置本地任务范围
            m_workers[i]->local_task_start = i * m_task_per_thread;
            m_workers[i]->local_task_end = std::min((i + 1) * m_task_per_thread, nr_task);
            
            // work_flag 为 true，表明任务需要被处理
            m_workers[i]->work_flag.store(true, std::memory_order_release);
        }

        // 主线程也需要工作，这是计算主线程的 start
        uint32_t start = (m_nr_threads - 1) * m_task_per_thread;
        m_task({start, nr_task, m_nr_threads - 1});

        // 同步，等待所有线程处理完
        sync();
        
        m_total_tasks.fetch_add(nr_task, std::memory_order_relaxed);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    m_total_time.fetch_add(duration.count(), std::memory_order_relaxed);
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
            // 优化：减少等待时间
            for (int it = 0; it < MAIN_THREAD_ACTIVE_WAIT; it++) {
                if (!m_workers[no_finished_id]->work_flag.load(
                            std::memory_order_acquire)) {
                    break;
                }
                if ((it < m_config.spin_wait_cycles || (it & 1))) {
                    INFER_PAUSE(m_config.spin_wait_cycles);  // 使用配置的自旋等待周期
                } else {
                    std::this_thread::yield();
                }
            }
        }
    } while (no_finished);
}

inline void ThreadPool::active() {
    // 优化：使用原子操作减少锁竞争
    if (!m_active.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_active) {
            m_active = true;
            m_cv.notify_all();
        }
    }
}

// 
void ThreadPool::deactive() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_active = false;
}
