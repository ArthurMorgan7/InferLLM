# ---------------------------------------------------------------------------- #
#                              电脑内存太小了，加载不了                            #
# ---------------------------------------------------------------------------- #

# Convert a chatglm model checkpoint to a InferLLM compatible file
#
# Load the model using Torch
# Iterate over all variables and write them to a binary file.
#
# Model Structure Header:
#  - Magic number (int)
#  - Param Offset (int)
#  - Param Length (int)
#  - Vocabulary Offset (int)
#  - Vocabulary Length (int)
#  - Tensor offset (int)
#
# Param :
#  - Hidden Size (int)
#  - Number of heads (int)
#  - Number of layers (int)
#  - Embedding Size (int)
#  - FC hidden size (int)
#  - Vocabulary Size (int)
#  - Weight Data Type (int) (0 = float32, 1 = float16, 2 = int8, 3 = uint8)
#
# For each tensor, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (int8_t[len])
#
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.

import sys
import json
import struct
from enum import Enum
import numpy as np
import torch
import argparse
import tempfile 
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentencepiece import SentencePieceProcessor 

# 命令行参数解析
parser = argparse.ArgumentParser(description="Convert a ChatGLM model to a InferLLM compatible fp16 data type file")
parser.add_argument("-o", "--outfile", type=str, help="the output file")
parser.add_argument("-v", "--version", type=int, default=1, help="the chatglm mode version")
parser.add_argument("-q", "--quantization", type=int, default=32, help="quantization bits")
args = parser.parse_args()

# output in the same directory as the model
model_out_path = args.outfile

class GGMLType(Enum):
    # src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L32
    F32 = 0
    F16 = 1
    QInt4 = 2
    # QUInt4 = 3
    QInt8 = 4

alignment_size = 32
bits = args.quantization
if bits == 32:
    dtype = GGMLType.F32
elif bits == 16:
    dtype = GGMLType.F16
    raise NotImplementedError(f"kernel not suport bits: {bits}")
elif bits == 8:
    dtype = GGMLType.QInt8
elif bits == 4:
    dtype = GGMLType.QInt4
else:
    raise NotImplementedError(f"Unknown quantization bits: {bits}")


# 模型加载和参数提取
version = args.version
if version == 1:
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float().state_dict()
    auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
elif version == 2:
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).float().state_dict()
    auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
# elif version == 3:
#     # 从 HuggingFace 库加载模型 THUDM/chatglm3-6b；以float32加载权重；返回模型的权重参数字典。
#     model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float().state_dict()
#     # 从 HuggingFace 库加载模型对应的分词器 
#     auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
#     # 从 HuggingFace 库加载模型对应的参数
#     config = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
elif version == 3:
    # 从 HuggingFace 库加载模型 THUDM/chatglm3-6b；以float32加载权重；返回模型的权重参数字典。
    model = AutoModel.from_pretrained("../../chatglm3-6b", trust_remote_code=True).float().state_dict()
    # 从 HuggingFace 库加载模型对应的分词器 
    auto_tokenizer = AutoTokenizer.from_pretrained("../../chatglm3-6b", trust_remote_code=True)
    # 从 HuggingFace 库加载模型对应的参数
    config = AutoConfig.from_pretrained("../../chatglm3-6b", trust_remote_code=True)



_, vocab_file = tempfile.mkstemp()
auto_tokenizer.save_vocabulary(vocab_file)
tokenizer = SentencePieceProcessor(vocab_file)

hparams = {
        "embd_size": config.hidden_size,
        "n_heads": config.num_attention_heads,
        "n_layers": config.num_layers,
}
hparams.update({"vocab_size": tokenizer.vocab_size()})

if version > 1:
    hparams.update({"multi_qeury": 1 if config.multi_query_attention else 0})
    hparams.update({"attention_patition": config.multi_query_group_num})
    hparams.update({"fc_hidden": config.ffn_hidden_size})


print(hparams)


##########################################  打开文件，开始写入 ############################################
# 打开一个二进制文件
fout = open(model_out_path, "wb")

fout.write(struct.pack("i", 0x0123456))     # 写入魔数。 i 表示 4 字节带符号整数； 被打包的数据



# 构造二进制的 param 数据
param_byte = struct.pack("i", hparams["embd_size"])
param_byte +=struct.pack("i", hparams["n_heads"])
param_byte +=struct.pack("i", hparams["n_layers"])
param_byte +=struct.pack("i", hparams["fc_hidden"])
param_byte +=struct.pack("i", hparams["vocab_size"])
if version > 1:
    param_byte +=struct.pack("i", hparams["multi_qeury"])
    param_byte +=struct.pack("i", hparams["attention_patition"])



# 构造二进制 vocab 数据
vocab_byte = bytearray()
for i in range(tokenizer.vocab_size()):
    if tokenizer.is_unknown(i):
        # "<unk>" token (translated as ??)
        text = " \u2047 ".encode("utf-8")
        vocab_byte += struct.pack("i", len(text))
        vocab_byte += text
    elif tokenizer.is_control(i):
        # "<s>"/"</s>" tokens
        vocab_byte += struct.pack("i", 0)
    elif tokenizer.is_byte(i):
        # "<U+XX>" tokens (which may be invalid UTF-8)
        piece = tokenizer.id_to_piece(i)
        if len(piece) != 6:
            print("Invalid token: " + piece)
            sys.exit(1)
        byte_value = int(piece[3:-1], 16)
        vocab_byte += struct.pack("i", 1)
        vocab_byte += struct.pack("B", byte_value)
    else:
        # normal token. Uses U+2581 (LOWER ONE EIGHTH BLOCK) to represent spaces.
        text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
        vocab_byte += struct.pack("i", len(text))
        vocab_byte += text



# 写入模型头信息
param_offset_addr = fout.tell() # 记录 param 头信息的地址
fout.seek(4, 1) # 数据的地址
fout.write(struct.pack("i", len(param_byte)))  # +4后，写入 param 数据的长度

vocab_offset_addr = fout.tell() # 记录 vocab 头信息的地址
fout.seek(4, 1) # 数据的地址
fout.write(struct.pack("i", len(vocab_byte))) # +4后，写入 vocab 数据的长度

tensor_offset_addr = fout.tell() # 记录 tensor 头信息的地址
fout.seek(4, 1) # 数据的地址



# 写入真正的数据
param_offset = fout.tell()  # 记录 param 数据的地址
fout.write(param_byte)      # 写入 param 数据


vocal_offset = fout.tell()  # 记录 vocab 数据的地址
fout.write(vocab_byte)      # 写入 vocab 数据


tensor_offset = fout.tell() # 记录 tensor 数据的地址


# 写入数据的地址
fout.seek(param_offset_addr, 0)
fout.write(struct.pack("i", param_offset))  # 写入 param 数据的地址

fout.seek(vocab_offset_addr, 0)
fout.write(struct.pack("i", vocal_offset))  # 写入 vocab 数据的地址

fout.seek(tensor_offset_addr, 0)
fout.write(struct.pack("i", tensor_offset)) # 写入 tensor 数据的地址





GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32


GGML_MEM_ALIGN = 16

def float32Toint8(tensor):
    oriShape = tensor.shape
    newLastElement = oriShape[-1] * 4
    newShape = oriShape[:-1] + (newLastElement,)
    tensor_bytes = tensor.numpy().tobytes()
    return torch.tensor(np.frombuffer(tensor_bytes, dtype=np.int8)).view(newShape)

def offset(tensor, alignment):
    # 计算tensor所占用的字节数
    num_bytes = tensor.element_size() * tensor.nelement()
    # 计算需要填充的字节数
    padding = (alignment - (num_bytes % alignment)) % alignment
    return num_bytes+padding, padding

def quantize_q8_0(tensor: torch.Tensor) -> torch.Tensor:
    """
    src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L51
    """
    # equivalent to ggml_quantize_q8_0 in ggml.c

    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).type(torch.int8)
    # add scale into each block
    tensor = torch.cat((float32Toint8(scale.float()), tensor), dim=-1)
    return tensor

def quantize_quint4(tensor: torch.Tensor) -> torch.Tensor:
    """
    src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L62
    """
    # equivalent to ggml_quantize_q4_0 in ggml.c
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4).type(torch.int8)
    # add scale into each block
    tensor = torch.cat((float32Toint8(scale.float()), tensor), dim=-1)
    return tensor

def quantize_qint4(tensor: torch.Tensor) -> torch.Tensor:
    """
    src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L62
    """
    # equivalent to ggml_quantize_q4_0 in ggml.c
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale).round().clamp(min=-8, max=7).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4).type(torch.int8)
    # add scale into each block
    tensor = torch.cat((float32Toint8(scale.float()), tensor), dim=-1)
    return tensor

def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32
    shape = tensor.shape

    # skip layers.X.attention.inner_attention.rope.freqs
    # 推理时不需要频率张量
    if name[-5:] == "freqs" or name[-4:]=="freq":
        return


    if name.endswith("query_key_value.weight") or name.endswith("attention.query_key_value.bias"):
        if version == 1:
            tensor = tensor.reshape(32, 3, -1).transpose(0, 1).reshape(-1, 4096)
    dshape = tensor.shape
    sname = name.encode('utf-8')


    # 量化
    if "layernorm" not in name:
        # tensor data
        if ggml_type == GGMLType.F32:       
            tensor = tensor.float()         # FP32
        elif ggml_type == GGMLType.F16:     
            tensor = tensor.half()          # FP16
        elif ggml_type == GGMLType.QInt8:   
            tensor = quantize_q8_0(tensor)  # INT8
        elif ggml_type == GGMLType.QInt4:   
            tensor = quantize_qint4(tensor) # INT4
        else:
            raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")
    else:
        tensor = tensor.float()
        ggml_type = GGMLType.F32

    n_dims = len(shape)
    print("Processing variable: " + name + " with shape: ", shape, " and type: ", ggml_type.value)

    f.write(struct.pack("iii", n_dims, len(sname), ggml_type.value))

    for i in range(n_dims):
        f.write(struct.pack("i", dshape[i]))
    
    f.write(sname)
    print("write tensor: ", name, " to file :", f.tell())

    tensor.numpy().tofile(f)
    # align address
    if ggml_type == GGMLType.QInt8 or ggml_type == GGMLType.QInt4:
        length, paddingSize =offset(tensor, alignment_size)
        if paddingSize>0:
            paddingTensor = torch.zeros(paddingSize)
            paddingTensor.numpy().tofile(f)
            print("write paddingTensor: ", name, "paddingSize:", paddingSize," to file :", f.tell())

# 移动到文件末尾
fout.seek(0, 2)
for k, v in model.items():
    dump_tensor(fout, k, v, dtype)


# I hope this deallocates the memory ..
model = None

# 关闭文件，收尾工作
fout.close()

print("Done. Output file: " + model_out_path)
print("")
