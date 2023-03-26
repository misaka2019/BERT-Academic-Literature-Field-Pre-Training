# BERT 预训练模型

这是一个使用 BERT 进行学术文献领域预训练任务的示例代码库。在本示例中，我们使用了一种基于实体增强的 MLM（Masked Language Modeling）方法，并使用了 chinese-roberta-wwm-ext 模型进行参数初始化。

## 数据预处理

我们使用了包含 653 万条学术文本的数据集来进行预训练。数据集来源于[CSL](https://github.com/ydli-ai/CSL)。CSL-大规模中文科学文献数据集，包含约40万篇中文论文元数据（标题、摘要、关键词以及学科领域标签）。
**数据预处理流程如下：**

1. 读取数据中的标题和摘要。
2. 由于摘要文本较长，对每个摘要进行精细切分句子，得到若干个句子。
3. 对分好的句子进行清洗，如删除不存在中文的样本、删除空格和存在的连续符号。
4. 对句子进行筛选，丢弃句子长度大于128或者小于10的句子。
5. 将处理后的文本数据按行保存到文件中，作为模型的输入数据。

## 模型

我们使用了 chinese-roberta-wwm-ext 模型进行参数初始化，该模型在中文语言处理任务中表现良好。csl数据集中包含关键词数据，于是我们将关键词数据加入jieba词库中。
然后通过wwm机制每轮动态对实体数据进行替换、删除、mask等操作以实现实体增强。

## 训练过程

我们使用了一张 2080ti 显卡，在 17 个小时内训练了 188500 步。在训练过程中，我们的模型从初始的 loss 值 2.4 逐渐降低到了 0.9左右。
![](p![](./png/loss.png)
## 使用示例
### 环境配置
````
pip install torch
pip install transformers
pip install jieba
pip install pandas
````


### 预训练
首先下载csl数据集到data文件下，然后运行data文件下的preprocess.py，再运行main.py
````
###
!python data/preprocess.py
!python main.py 
````

### 微调
在训练完成后，你可以使用我们提供的模型来进行下游任务。你可以在 model 目录下找到我们训练好的模型文件，并将其用于你自己的任务中。

````
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('model/yourmodelpath')
model = BertForMaskedLM.from_pretrained('model/yourmodelpath')

input_text = "在学术界，BERT 是一种非常流行的预训练模型，其表现在多项任务上都超过了人类水平。"
masked_index = 7
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_ids[masked_index] = tokenizer.mask_token_id

with torch.no_grad():
    outputs = model(torch.tensor([input_ids]))
    prediction_scores = outputs[0][0, masked_index]

predicted_index = torch.argmax(prediction_scores).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(predicted_token)
`````
在这个示例中，我们使用了 transformers 库来加载我们训练好的模型，并使用它来预测给定句子中被遮蔽的词语。你可以根据你自己的任务来修改代码，并使用我们提供的模型进行下游任务的训练。

## 后续优化方向
- 训练更多的轮数
- 使用LAMB优化器加速收敛
- 继续扩充学术领域数据集
- 结合具体任务进行微调

## 参考
- https://zhuanlan.zhihu.com/p/465462642
- https://github.com/BSlience/search-engine-zerotohero/tree/main/public/bert_wwm_pretrain
- https://github.com/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb