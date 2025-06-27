

# TODO:<description>:创建的 Hugging Face 训练项目的结构，但是它没有健虚拟环境，指令为
# python3 create_hf_project.py

import os
import json
from pathlib import Path


def create_project(project_name="huggingface-project"):
    # 创建基础目录
    base_path = Path(project_name)
    base_path.mkdir(exist_ok=True)

    # 创建子目录
    (base_path / "data").mkdir(exist_ok=True)
    (base_path / "scripts").mkdir(exist_ok=True)
    (base_path / "config").mkdir(exist_ok=True)
    (base_path / "models").mkdir(exist_ok=True)
    (base_path / "logs").mkdir(exist_ok=True)

    # 创建数据文件
    (base_path / "data" / "train.csv").write_text("""text,label
"这部电影太精彩了，强烈推荐！",1
"画面粗糙，剧情无聊，浪费时间",0
"演员表演出色，但剧情有硬伤",0
"年度最佳影片，不容错过",1
""")

    (base_path / "data" / "test.csv").write_text("""text,label
"这款产品性价比很高，值得购买",1
"服务态度差，再也不会光顾",0
"功能强大但操作复杂",0
"超出预期的好产品",1
""")

    # 创建预处理脚本
    (base_path / "scripts" / "preprocess.py").write_text("""\"\"\"数据预处理脚本\"\"\"
import pandas as pd
import os
from pathlib import Path

def preprocess_data():
    print("="*50)
    print("开始数据预处理")
    print("="*50)

    # 定义路径
    data_dir = Path("data")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_processed_path = data_dir / "train_processed.csv"
    test_processed_path = data_dir / "test_processed.csv"

    # 读取数据
    print(f"读取训练数据: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"读取测试数据: {test_path}")
    test_df = pd.read_csv(test_path)

    print(f"原始训练集大小: {len(train_df)}")
    print(f"原始测试集大小: {len(test_df)}")

    # 这里可以添加数据清洗和预处理步骤
    # 示例：删除空值
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    print(f"清洗后训练集大小: {len(train_df)}")
    print(f"清洗后测试集大小: {len(test_df)}")

    # 保存处理后的数据
    train_df.to_csv(train_processed_path, index=False)
    test_df.to_csv(test_processed_path, index=False)

    print("预处理完成!")
    print(f"处理后的训练集保存到: {train_processed_path}")
    print(f"处理后的测试集保存到: {test_processed_path}")
    print("="*50)

if __name__ == "__main__":
    preprocess_data()
""")

    # 创建推理脚本
    (base_path / "scripts" / "inference.py").write_text("""\"\"\"模型推理脚本\"\"\"
from transformers import pipeline
import pandas as pd
import os
from pathlib import Path

def run_inference(model_path, data_path=None, output_path=None):
    print("="*50)
    print("开始模型推理")
    print("="*50)

    # 设置默认路径
    if data_path is None:
        data_path = Path("data") / "test.csv"
    if output_path is None:
        output_path = Path("data") / "predictions.csv"

    print(f"加载模型: {model_path}")
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path
    )

    # 读取数据
    print(f"读取数据: {data_path}")
    df = pd.read_csv(data_path)
    texts = df["text"].tolist()

    print(f"对 {len(texts)} 条文本进行推理...")
    results = classifier(texts)

    # 处理结果
    df["predicted_label"] = [result['label'] for result in results]
    df["confidence"] = [result['score'] for result in results]

    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"推理结果保存到: {output_path}")

    # 打印部分结果
    print("示例预测结果:")
    for i, row in df.head(3).iterrows():
        print(f"文本: '{row['text']}'")
        print(f"预测标签: {row['predicted_label']} (置信度: {row['confidence']:.4f})")
        print("-" * 50)

    print("推理完成!")
    print("="*50)

if __name__ == "__main__":
    model_path = "./models/final_model"
    print(f"使用默认模型路径: {model_path}")
    run_inference(model_path)
""")

    # 创建配置文件
    config = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "save_total_limit": 2
    }
    (base_path / "config" / "training_args.json").write_text(json.dumps(config, indent=2))

    # 创建主训练文件
    (base_path / "train.py").write_text("""\"\"\"主训练脚本\"\"\"
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from pathlib import Path

print("="*50)
print("Hugging Face 文本分类训练")
print("="*50)

# 加载配置
print("加载训练配置...")
with open("config/training_args.json") as f:
    config = json.load(f)

# 1. 加载数据集
print("加载数据集...")
# 检查是否有预处理后的数据
data_dir = Path("data")
train_path = data_dir / "train_processed.csv" if os.path.exists(data_dir / "train_processed.csv") else data_dir / "train.csv"
test_path = data_dir / "test_processed.csv" if os.path.exists(data_dir / "test_processed.csv") else data_dir / "test.csv"

print(f"使用训练数据: {train_path}")
print(f"使用测试数据: {test_path}")

dataset = load_dataset('csv', data_files={
    'train': str(train_path),
    'test': str(test_path)
})

print(f"训练集大小: {len(dataset['train'])}")
print(f"测试集大小: {len(dataset['test'])}")

# 2. 加载分词器
model_name = "bert-base-chinese"
print(f"加载分词器: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 数据预处理
print("数据预处理中...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("数据预处理完成!")

# 4. 创建模型
print(f"创建模型: {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

# 5. 评估指标函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

# 6. 训练参数配置
print("配置训练参数...")
training_args = TrainingArguments(
    output_dir="./models",
    logging_dir="./logs",
    **config
)

# 7. 创建Trainer
print("创建Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 8. 开始训练
print("开始训练...")
print(f"将训练 {training_args.num_train_epochs} 个epochs")
train_result = trainer.train()

# 9. 保存最终模型
print("训练完成，保存模型...")
model_path = "./models/final_model"
trainer.save_model(model_path)
print(f"模型已保存到 {model_path}")

# 保存训练指标
metrics = train_result.metrics
metrics_file = os.path.join(training_args.output_dir, "train_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"训练指标已保存到 {metrics_file}")

print("" + "="*50)
print("训练完成!")
print("="*50)
""")

    # 创建依赖文件
    (base_path / "requirements.txt").write_text("""transformers>=4.30.0
datasets
torch
tensorboard
scikit-learn
pandas
numpy
""")

    # 创建README文件
    (base_path / "README.md").write_text(f"""# {project_name}""")


if __name__ == "__main__":
    project_name = input("请输入项目名称 (默认为 'huggingface-project'): ") or "huggingface-project"
    create_project(project_name)