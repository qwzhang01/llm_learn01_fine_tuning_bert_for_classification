# 从 transformers 库中导入两个类
# AutoTokenizer：用于加载与模型匹配的分词器，负责将文本转换为模型可处理的数字输入。
# AutoModelForMaskedLM：用于加载模型
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 定义要加载的模型名称
model_name = "bert-base-chinese"
# 指定模型和分词器文件的本地缓存目录。
cache_dir = "model/bert-base-chinese"

# 加载与模型匹配的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# 从 Hugging Face 模型库加载预训练模型
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
print(f"模型分词器下载到：{cache_dir}")
