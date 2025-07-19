#pragma once

#include "file.h"

#include <map>
#include <memory>
#include <string>
#include <vector>


namespace inferllm{
/* -------------------------------------------------------------------------- */
/*                       管理词和编号之间的映射关系                                */
/* -------------------------------------------------------------------------- */
class Vocab {
public:
    // 内部类型
    using Id = int32_t;         // 编号
    using Token = std::string;  // 词
    struct TokenScore {
        Token tok;      // string
        float score;    // 得分
    };                          // 词分数

    void load_vocab(std::shared_ptr<InputFile> fs, size_t size) {
        id_to_token.resize(size);
        std::string word;
        for (size_t i = 0; i < size; i++) {
            float score = 0;
            uint32_t len;
            fs->read_raw((char*)&len, sizeof(len));
            word.resize(len);
            fs->read_raw((char*)word.data(), len);

            token_to_id[word] = i;
            id_to_token[i].tok = word;
            id_to_token[i].score = score;
        }
    }

    void load_vocab_with_score(std::shared_ptr<InputFile> fs, size_t size) {
        id_to_token.resize(size);
        std::string word;
        for (size_t i = 0; i < size; i++) {
            float score = 0;
            uint32_t len;
            fs->read_raw((char*)&len, sizeof(len));
            word.resize(len);
            fs->read_raw((char*)word.data(), len);
            fs->read_raw((char*)&score, sizeof(score));

            token_to_id[word] = i;
            id_to_token[i].tok = word;
            id_to_token[i].score = score;
        }
    }

    Id map_to_id(const Token& str) { return token_to_id[str]; }

    Token unmap_to_token(Id id) { return id_to_token[id].tok; }

    std::map<Token, Id> token_to_id;        // 词 → 编号
    std::vector<TokenScore> id_to_token;    // 编号 → 得分
};
};