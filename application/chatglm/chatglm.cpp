#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <signal.h>
#include <unistd.h>


#include "model.h"

struct app_params {
    // 基础配置参数
    int32_t seed = -1;  // TODO 具体怎么使用？ 随机数种子，确保每次生成不同的输出 
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());  // 推理时最多使用的线程数
    int32_t n_predict = 128;     // 要生成的 token 数量
    int32_t repeat_last_n = 64;  // 重复惩罚机制中考虑的“最近 N 个 token”。用于控制模型避免重复。
    int32_t n_ctx = 2048;        // 上下文窗口大小

    // 采样相关参数
    int32_t top_k = 40;             // Top-K 采样：只从概率前 K 个候选中采样，K 越小越保守
    float top_p = 0.95f;            // Top-P（nucleus）采样：从概率总和累积到 p（如 0.95）的一组 token 中采样。更灵活。
    float temp = 0.10f;             // 控制采样分布的随机性。越小越确定（0 接近贪婪），越大越多样化。
    float repeat_penalty = 1.30f;   // 重复惩罚系数：用于降低模型输出重复词语的概率。值大于 1 时惩罚更强。

    // 模型加载参数
    std::string model;              // 模型路径
    bool use_color = true;          // 终端输出是否带颜色
    bool use_mmap = false;          // 是否使用 mmap 映射模型文件到内存
    std::string dtype = "float32";  // 指定推理时使用的数据类型
    std::string device = "cpu";     // 指定运行设备
    int32_t version = 1;            // 模型版本号
};

// 打印二进制文件的用户手册
void app_print_usage(int argc, char** argv, const app_params& params) {
    // clang-format off
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", params.repeat_penalty);
    fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  --mmap                enable mmap when read weights, default = false\n");
    fprintf(stderr, "  -d type               configure the compute type, default float32, can be float32 and flot16 now.\n");
    fprintf(stderr, "  -g type               configure the compute device type, default CPU, can be CPU and GPU now.\n");
    fprintf(stderr, "\n");
    // clang-format on
}

// 解析命令行参数
bool app_params_parse(int argc, char** argv, app_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];  // 隐式调用了类型转换构造函数
        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);     // 隐式调用了类型转换构造函数
        } 
        else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } 
        else if (arg == "--top_k") {
            params.top_k = std::stoi(argv[++i]);
        } 
        else if (arg == "-c" || arg == "--ctx_size") {
            params.n_ctx = std::stoi(argv[++i]);
        } 
        else if (arg == "-d" || arg == "--dtype") {
            params.dtype = argv[++i];
        } 
        else if (arg == "-g") {
            params.device = argv[++i];
        } 
        else if (arg == "--top_p") {
            params.top_p = std::stof(argv[++i]);
        } 
        else if (arg == "--temp") {
            params.temp = std::stof(argv[++i]);
        } 
        else if (arg == "--repeat_last_n") {
            params.repeat_last_n = std::stoi(argv[++i]);
        } 
        else if (arg == "--repeat_penalty") {
            params.repeat_penalty = std::stof(argv[++i]);
        } 
        else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } 
        else if (arg == "--color") {
            params.use_color = true;
        } 
        else if (arg == "--mmap") {
            params.use_mmap = true;
        } 
        else if (arg == "-v" || arg == "--version") {
            params.version = std::stoi(argv[++i]);
        } 
        else if (arg == "-h" || arg == "--help") {
            app_print_usage(argc, argv, params);
            exit(0);
        } else {
            exit(0);
        }
    }

    return true;
}

std::string running_summary;
void sigint_handler(int signo) {
    // 
    if (signo == SIGINT) {
        printf("\n");
        printf("%s", running_summary.c_str());
        exit(130);
    }
};


// Llambda表达式 —— 将模型输出的一些特殊编码还原成原始字符
void fix_word(std::string& word) {
    auto ret = word;
    if (word == "<n>" || word == "<n><n>")
        word = "\n";
    if (word == "<|tab|>")
        word = "\t";
    int pos = word.find("<|blank_");
    if (pos != -1) {
        int space_num = atoi(word.substr(8, word.size() - 10).c_str());
        word = std::string(space_num, ' ');
    }
    pos = word.find("▁");
    if (pos != -1) {
        word.replace(pos, pos + 3, " ");
    }
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' &&
        word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word = static_cast<char>(num);
    }
}


int main(int argc, char** argv) {
    // 定义一个结构体用于存储命令行参数
    app_params params;

    // 解析命令行参数
    if (app_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    // 命令行参数 → 模型参数
    inferllm::ModelConfig config;
    config.compt_type = params.dtype;       // 数据类型
    config.device_type = params.device;     // 运行设备
    config.nr_thread = params.n_threads;    // 线程束
    config.enable_mmap = params.use_mmap;   // 是否使用 mmap
    config.nr_ctx = params.n_ctx;           // 上下文窗口大小

    std::string model_name;    // 模型名称
    uint32_t etoken;           // 结尾token的id
    if(params.version == 1){
        model_name = "chatglm";
        etoken = 130005;
    }
    else if (params.version == 2) {
        model_name = "chatglm2";
        etoken = 2;
    }else if(params.version == 3){
        model_name = "chatglm3";
        etoken = 2;
    }

    // 创建模型实例 model ※ 
    std::shared_ptr<inferllm::Model> model = std::make_shared<inferllm::Model>(config, model_name);
    model->load(params.model);
    model->init(params.top_k, params.top_p, params.temp, params.repeat_penalty, params.repeat_last_n, params.seed, etoken);

    // 自定义 Ctrl C 信号的处理函数
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;  // 设置信号处理函数
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    
    /* ---------------------------------- 可视化输出（不重要） --------------------------------- */
    // 打印基本的参数
    fprintf(stderr, "%s: interactive mode on.\n", __func__);
    fprintf(stderr,
            "sampling parameters: temp = %f, top_k = %d, top_p = %f, "
            "repeat_last_n = %i, repeat_penalty = %f\n",
            params.temp, params.top_k, params.top_p, params.repeat_last_n,
            params.repeat_penalty);
    fprintf(stderr, "\n\n");

    std::vector<char> embd;

    // 文本提示
    fprintf(stderr,
            "== 运行模型中. ==\n"
            " - 输入 Ctrl+C 将在退出程序.\n"
            " - 如果你想换行，请在行末输入'\\'符号.\n");

    

    /* -------------------------------------------------------------------------- */
    /*                                  main loop                                 */
    /* -------------------------------------------------------------------------- */
    bool is_interacting = true;     // True: 输入用户问题   False：输出模型响应
    std::string user_input, output; // 输入输出的字符串
    int iter = 0;       // 轮次编号
    int token_id = 0;   // 当前轮输出的 token 计数

    // 直到模型剩余的 token 用完才结束
    while (model->get_remain_token() > 0) {
        printf("\n> ");
        bool another_line = true;

        // 如果是版本 2，则在输入前添加对话格式前缀
        if (params.version == 2) {
            user_input = "[Round " + std::to_string(iter) + "]\n\n问：";
        }


        /* ---------------------------------- 处理输入 ---------------------------------- */
        while (another_line) {
            fflush(stdout);
            std::string input;  // input 暂存用户输入
            input.resize(256);
            char* buf = const_cast<char*>(input.data());
            int n_read;
            int res = scanf("%255[^\n]%n%*c", buf, &n_read);    // 一次读取一行
            

            // 读取错误的处理
            if (res == EOF) return 0;   
            else if (res == 0) {
                if (scanf("%*c") <= 0) {} // 读取一个字符，但不赋值给任何变量
                n_read = 0;
            }


            // 读取成功且输入末尾是 '\'
            if (n_read > 0 && buf[n_read - 1] == '\\') {
                buf[n_read - 1] = '\n'; // 把 \\ 替换成 \n
                // another_line = true;
                input.resize(n_read + 1);
            } 
            else {
                another_line = false;
                input.resize(n_read);
            }
            user_input += input;    // 把当前行的输入拼接到整体的输入后面
        }

        // 如果是版本 2，则在输入后添加对话格式后缀，引导模型回答
        if (params.version == 2) {
            user_input += "\n\n答：";
        }

        if (user_input.empty()) continue;

        /* ---------------------------------- 模型推理 ---------------------------------- */
        
        // 初次生成
        int token;
        output = model->decode(user_input, token);  // decode 模型的接口函数，理解输入，生成输出 output，
        fix_word(output); 
        printf("%s", output.c_str());
        fflush(stdout);  
        user_input.clear();
        iter++;

        // 迭代生成
        is_interacting = false;
        while (!is_interacting) {
            std::string o = model->decode_iter(token);  // 以 input 生成的第一个 token 为基础继续补全
            fix_word(o);
            printf("%s", o.c_str());
            fflush(stdout);
            token_id++;
            iter++;           // after answering the question, get the user input again

            // TODO 实时更新摘要？
            if (token_id % 10 == 0) {
                running_summary = model->decode_summary();
            }

            // 输出完毕
            if (token == etoken) {
                printf("\n");
                running_summary = model->decode_summary();
                is_interacting = true;
            }
        } 
    }
    return 0;
}
