from torch.utils.data import Dataset
import torch
import random
import jieba
import numpy as np
from tqdm import tqdm
import pandas as pd
import jieba
import os
import torch
from transformers import (BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, DataCollatorForWholeWordMask,
                          BertTokenizer, LineByLineTextDataset, TrainingArguments, Trainer)


class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = open(data_dir, 'r').read().split('\n')
        self.data = [i.strip(' ') for i in self.data if len(i.strip(' ')) > 10]
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.encode_sent(self.data[idx], self.tokenizer)
        data = (
            data_dict['input_ids'],
            data_dict['token_type_ids'],
            data_dict['attention_mask'],
        )
        # return {"input_ids": torch.tensor(data, dtype=torch.long)}
        return data

    def encode_sent(self, sent, tokenizer):
        inputs_dict = tokenizer(sent, add_special_tokens=True, truncation=True, max_length=max_seq_len)

        return inputs_dict



class DataCollator:
    def __init__(self, max_seq_len, tokenizer, mlm_probability=0.10):
        # max_seq_len 用于截断的最大长度
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability  # 遮词概率

    def get_wwd(self, ids):
        text = self.tokenizer.decode(ids[1:-1])
        text = ''.join(text.split(' '))
        # 根据输入ID获取
        token_jieba = jieba.lcut(text)
        token_bert = self.tokenizer.tokenize(text)
        wwd_id = []
        for word in token_jieba:
            for i, char in enumerate(token_bert):
                if char == word[0]:
                    continue
                if char in word:
                    wwd_id.append(self.tokenizer.encode(text)[i + 1])

        return wwd_id

    # 截断和填充
    def truncate_and_pad(self, input_ids_list, token_type_ids_list, attention_mask_list, max_seq_len):
        # 初始化一个样本数量 * max_seq_len 的二维tensor
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])  # 当前句子的长度
            # 如果长度小于最大长度
            if seq_len <= max_seq_len:
                # 把input_ids_list中的值赋值给input_ids
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:  # self.tokenizer.sep_token_id = 102
                # 度超过最大长度的句子，input_ids最后一个值设置为102即分割词
                # input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
                #                                       [self.tokenizer.sep_token_id], dtype=torch.long)
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:max_seq_len - 1] +
                                                      [self.tokenizer.sep_token_id], dtype=torch.long)

            seq_len = min(len(input_ids_list[i]), max_seq_len)
            token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
        # print('截断和填充之前' + '*' * 30)
        # print(input_ids_list)  # 每个句子向量长度不一
        # print('截断和填充之后' + '*' * 30)
        # print(input_ids)    # 每个句子向量长度统一
        return input_ids, token_type_ids, attention_mask

    def _whole_word_mask(self, input_ids_list, max_seq_len, max_predictions=512):
        cand_indexes = []
        for (i, token) in enumerate(input_ids_list):
            # 跳过开头与结尾
            if (token == str(self.tokenizer.cls_token_id)  # 101
                    or token == str(self.tokenizer.sep_token_id)):  # 102
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)  # 打乱
        # 根据句子长度*遮词概率算出要预测的个数，最大预测不超过512，不足1的按1
        # round()四舍五入，但是偶数.5会舍去，不过这是细节问题，影响不是很大
        num_to_predict = min(max_predictions, max(1, int(round(len(input_ids_list) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        # mask 掉的 token 使用 1 来进行标记，否则使用 0 来标记
        mask_labels = [1 if i in covered_indexes else 0 for i in range(min(len(input_ids_list), max_seq_len))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels)

    def whole_word_mask(self, input_ids_list, max_seq_len) -> torch.Tensor:
        mask_labels = []
        for input_ids in input_ids_list:
            # 随机获取遮罩词
            wwm_id = self.get_wwd(input_ids)
            if len(wwm_id) > len(input_ids) * 0.2:
                wwm_id = np.random.choice(wwm_id, int(len(input_ids) * 0.2))

            # 给挑选出来的位置添加 "##"标记
            input_id_str = [f'##{id_}' if i in wwm_id else str(id_) for i, id_ in enumerate(input_ids)]
            mask_label = self._whole_word_mask(input_id_str, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs, mask_labels):
        # 这个函数用于屏蔽输入序列中的一些标记，将它们替换为掩蔽标记或随机标记。
        labels = inputs.clone()
        probability_matrix = mask_labels

        # 创建特殊标记掩蔽，标记标签张量中的特殊标记。
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 将特殊标记的屏蔽概率设置为0。
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # 如果有一个填充标记，则创建填充掩蔽，并将屏蔽填充标记的概率设置为0。
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # 创建掩蔽索引，其中包含哪些标记需要被屏蔽的布尔掩蔽。
        masked_indices = probability_matrix.bool()
        # 将未屏蔽的标记的标签设置为-100。
        labels[~masked_indices] = -100
        # 创建一个应替换为掩蔽标记的索引张量。
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # 将所选索引替换为掩蔽标记。
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 创建应替换为随机标记的索引张量。
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # 生成要替换所选索引的随机标记。
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    # 重写魔术方法，可以把类的对象当做函数去调用
    def __call__(self, examples):
        # pad前的（句子不一样长，需要填充）
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        # 动态识别batch中最大长度，用于padding操作
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        # 如果这一批中，所有句子都比设定的最大长度还小，那直接使用该批次的最大长度，
        # 可以减少运算数据量，加快速度
        # 如果这一批中有句子比设定的最大长度还长，后续就会被截断
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        # pad后的
        input_ids, token_type_ids, attention_mask = self.truncate_and_pad(
            input_ids_list, token_type_ids_list, attention_mask_list, max_seq_len
        )

        # 遮蔽单词,whole word mask策略
        batch_mask = self.whole_word_mask(input_ids_list, max_seq_len)
        # 针对得到的需要mask的词，进行实际mask操作（80%进行mask掉，10%进行随机替换，10%选择保持不变）
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }
        return data_dict


def main():
    tokenizer = BertTokenizer.from_pretrained(bert_file)
    model = BertForMaskedLM.from_pretrained(bert_file)
    print('model of parameters: ', model.num_parameters())

    # dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='../plm/text.txt', block_size=128)
    dataset = TextDataset(data_dir, tokenizer)
    # data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    data_collator = DataCollator(max_seq_len=max_seq_len, tokenizer=tokenizer, mlm_probability=0.15)
    print('data of lines: ', len(dataset))

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_total_limit=10,
        num_train_epochs=10,
        fp16_backend='auto',
        per_device_train_batch_size=64,
        prediction_loss_only=True,
        weight_decay=0.1,
        dataloader_num_workers=12,
        logging_dir='../tf_logs',
        logging_first_step=True,
        #  deepspeed='plm/deepseed.json',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        # callbacks=[es],
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    bert_file = "../plm/chinese-roberta-wwm-ext"
    data_dir = './data/clean_text_list.txt'
    output_dir = "../hy-tmp"
    max_seq_len = 128

    get_keyword()
    main()
