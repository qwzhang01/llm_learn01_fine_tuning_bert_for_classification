from datasets import load_dataset

# 加载在线数据
dataset = load_dataset(
    path="lansinuote/ChnSentiCorp",  # HuggingFace数据集名称
    cache_dir="data/"  # 数据集下载/缓存位置
)
print(dataset)

# 打印数据集
train_data = dataset['train']
for data in train_data:
    print(data)
