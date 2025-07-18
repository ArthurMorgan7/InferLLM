// Various helper functions and utilities

#pragma once

#include "file.h"
#include "vocab.h"

#include <chrono>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>


namespace inferllm {
class Timer {
    using Time = std::chrono::high_resolution_clock;
    using ms = std::chrono::milliseconds;
    using fsec = std::chrono::duration<float>;

public:
    Timer() { start = Time::now(); }
    double get_time() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<fsec>(end - start);
        return duration.count();
    };

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//
Vocab::Id llama_sample_top_p_top_k(
        const Vocab& vocab, const float* logits, std::list<Vocab::Id>& last_n_tokens,
        double repeat_penalty, int top_k, double top_p, double temp, std::mt19937& rng);

// filer to top K tokens from list of logits
void sample_top_k(std::vector<std::pair<double, Vocab::Id>>& logits_id, int top_k);

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

};  // namespace inferllm

//! branch prediction hint: likely to take
#define infer_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define infer_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

#define INFER_LOG(format, ...)   fprintf(stderr, format, ##__VA_ARGS__)
#define INFER_ERROR(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define INFER_ASSERT(expr, message)                                           \
    do {                                                                      \
        if (infer_unlikely(!(expr))) {                                        \
            INFER_ERROR(                                                      \
                    "Assert \' %s \' failed at file : %s \n"                  \
                    "line %d : %s,\nextra "                                   \
                    "message: %s",                                            \
                    #expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); \
            abort();                                                          \
        }                                                                     \
    } while (0)

#if ENABLE_GPU
#define CUDA_CHECK(expr)                                                  \
    do {                                                                  \
        cudaError_t err = (expr);                                         \
        if (err != cudaSuccess) {                                         \
            auto error = cudaGetErrorString(err);                         \
            INFER_ERROR(                                                  \
                    "CUDA error %d: %s, at file : %s \n"                  \
                    "line %d : %s ",                                      \
                    err, error, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
            abort();                                                      \
        }                                                                 \
    } while (0)
#if CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err)                                                       \
    do {                                                                        \
        cublasStatus_t err_ = (err);                                            \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n", err_, __FILE__, \
                    __LINE__, cublasGetStatusString(err_));                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                              \
    do {                                                                               \
        cublasStatus_t err_ = (err);                                                   \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)
#endif  // CUDART_VERSION >= 11
#else
#define CUDA_CHECK(expr)
#define CUBLAS_CHECK(err)
#endif
