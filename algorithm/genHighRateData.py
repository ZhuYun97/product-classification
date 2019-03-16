import pandas as pd
import jieba
import math
import codecs
import random
import re
from collections import Counter
train_csv = pd.read_csv("./train/train.tsv", sep="\t", encoding = 'gb18030')

train_csv = train_csv.dropna()

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    stopwords.append("★")
    stopwords.append("[")
    stopwords.append("]")
    stopwords.append("【")
    stopwords.append("】")
    stopwords.append("(")
    stopwords.append(")")
    stopwords.append("/")
    stopwords.append("-")
    stopwords.append("（")
    stopwords.append("）")
    stopwords.append("／")
    stopwords.append("-")
    stopwords.append("+")
    stopwords.append("*")
    stopwords.append("|")
    stopwords.append("·")
    stopwords.append(":")
    stopwords.append("…")
    stopwords.append("’")
    stopwords.append("'")
    stopwords.append('"')
    stopwords.append('◆')
     #  空格不能去，否则准确率极低
    return stopwords


stopwords = stopwordslist("./stopwords/chinese_stops.txt")

freq = Counter()
for index, row in train_csv.iterrows():
    # jieba分词
    seg_text = jieba.cut(row["ITEM_NAME"].replace("\t", " ").replace("\n", " "), cut_all=False)

    # 去停用词
    seg_text_filter = [word for word in seg_text if word not in stopwords]
    for word in seg_text_filter:
        freq[word] += 1

# 把常用词保存下来
most_freq_words = freq.most_common(70000)
most_freq_words = [word[0] for word in most_freq_words if word[1] >= 5]
train_file = codecs.open("./highrate.train", 'w', 'utf-8')
# val_file = codecs.open("./goods.data.seg.val", 'w', 'utf-8')
test_file = codecs.open("./highrate.test", 'w', 'utf-8')

jieba.suggest_freq('q币', True)
jieba.suggest_freq('Q币', True)
jieba.suggest_freq('qb', True)
jieba.suggest_freq('红钻', True)
jieba.suggest_freq('绿钻', True)
jieba.suggest_freq('黄钻', True)
jieba.suggest_freq('蓝钻', True)
jieba.suggest_freq('紫钻', True)
jieba.suggest_freq('黑钻', True)


for index, row in train_csv.iterrows():

    seg_text = jieba.cut(row["ITEM_NAME"].replace("\t", " ").replace("\n", " "), cut_all=False)

    seg_text_filter = [word for word in seg_text if word not in stopwords and word in most_freq_words and not re.match("^\d*元$|^\d*$", word)]
    outline = " ".join(seg_text_filter)

    rand_num = random.random()

    if rand_num < 0.9:
        if len(row['ITEM_NAME']) and row['TYPE']: # 有大概100多条数据格式是不规范的，不带标签的。如"视听：幻觉的构建：开创电影声音…|4168523	图书杂志--艺术--影视"
            outline = outline + "\t__label__" + row["TYPE"] + "\n"
            train_file.write(outline)
            train_file.flush()
    else:
        if len(row['ITEM_NAME']) and row['TYPE']:
            outline = outline + "\t__label__" + row["TYPE"] + "\n"
            test_file.write(outline)
            test_file.flush()

train_file.close()
test_file.close()
