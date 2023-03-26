from tqdm import tqdm
import pandas as pd
from data.utils import *
import jieba
import os


def get_sent_list(data_dir, val=True):
    text_list = []
    clean_text_list = []
    if os.path.exists(data_dir):
        # 保存文本
        clean_text_list = open(data_dir, 'r').read().strip('\n')
        train_num = len(clean_text_list) * 0.9
        if val:
            train_text = clean_text_list[:train_num]
            val_text = clean_text_list[train_num:]
            return train_text, val_text
        else:
            return clean_text_list

    # 读取TSV文件并将其转换为DataFrame对象
    df = pd.read_csv('./data/csl_camera_readly.tsv', sep='\t', error_bad_lines=False, header=None)
    # 读取csv数据
    with open('./data/train_unsupervised.csv') as f:
        for line in tqdm(f.readlines()):
            sent = line.split(',')[0].strip('\n')
            text_list.append(sent)

    text_list += list(df[0])

    # filtered_df[1]为摘要数据，要进行句子切分
    for i in tqdm(df[1]):
        text_list += cut_sent(i)

    # 文本清洗
    for text in tqdm(text_list):
        if clean_text(text):
            text = re.sub(pattern3, '', text).strip(' ')
            clean_text_list.append(text)

    clean_text_list = list(set(clean_text_list))
    # 保存文本
    with open(data_dir, 'w') as f:
        for item in clean_text_list:
            f.write("%s\n" % item)
    train_num = len(clean_text_list) * 0.9
    if val:
        train_text = clean_text_list[:train_num]
        val_text = clean_text_list[train_num:]
        return train_text, val_text
    else:
        return clean_text_list


def get_keyword(new_words_file):
    df = pd.read_csv('./data/csl_camera_readly.tsv', sep='\t', error_bad_lines=False, header=None)
    add_word = []
    for words in df[2]:
        add_word += words.split('_')
    add_word = list(set(add_word))
    # 读取新增词语的文本文件
    with open(new_words_file, 'w') as f:
        for item in add_word:
            f.write("%s " % item)
    with open(new_words_file, 'r', encoding='utf-8') as f:
        new_words = f.read()

    # 将新增词语加入词典中
    for word in new_words.split():
        jieba.add_word(word)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='preprocess')

    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store data')
    parser.add_argument('--new_words_file', type=str, default='./data', help='Directory to store keywords')

    args = parser.parse_args()
    get_sent_list(args.data_dir)
    get_keyword(args.new_words_file)
