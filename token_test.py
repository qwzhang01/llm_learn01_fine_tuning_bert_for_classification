"""
字符数据处理方式
"""
# 导入 transformers 的 BertTokenizer类
# BertTokenizer 是bert的分词器，负责将文本切割为Token
from transformers import BertTokenizer

# 加载分词器，使用模型的绝对路径
token = BertTokenizer.from_pretrained(
    r"/Users/avinzhang/git/avin-kit/llm_test/test02/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

# 定义需要处理的中文文本列表
sentences = ["我们都有一个家，", "沉睡百年，国人渐已醒，"]

# 对多个句子进行批量编码
out = token.batch_encode_plus(
    batch_text_or_text_pairs=sentences,
    # 添加 BERT 所需的特殊 token：[CLS]（序列开头）和 [SEP]（序列结尾）。
    add_special_tokens=True,
    # 当句子长度大于max_length（上限是model_max_length）时,截断
    truncation=True,
    # 设置最大序列长度为 15（包括特殊 token）。超过 15 的部分会被截断。
    max_length=15,
    # 将所有序列填充到 max_length（15），短于 15 的用 [PAD]（ID 为 0）补齐。
    padding="max_length",
    # 可取值为 tf, pt, np, 默认为list
    # 返回结果为 Python 列表，而不是张量（如 PyTorch 的 pt 或 TensorFlow 的 tf）。
    return_tensors=None,
    # 返回 attention_mask，标记有效 token（1）和填充 token（0）。
    return_attention_mask=True,
    # 返回 token_type_ids，区分不同句子段（通常用于句子对任务）。
    return_token_type_ids=True,
    # 返回 special_tokens_mask，标记特殊 token 的位置（1 表示特殊 token，0 表示普通 token）。
    return_special_tokens_mask=True,
    # 返回序列长度
    return_length=True
)

print(out)
# input_ids 就是编码后的词
# token_type_ids 第一个句子和特殊符号的位置，第二个句子的位置1（）只针对于上下文编码
# special_tokens_mask 特殊符号位置是1，其他位置是0
# length 编码之后的序列长度

# 打印Token
for k, v in out.items():
    print(k, ":", v)

# 将Token还原为自然语言
print(token.decode(out["input_ids"][0]), token.decode(out["input_ids"][1]))