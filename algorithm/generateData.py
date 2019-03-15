import pandas as pd
import jieba
import math
import codecs
import random

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
    return stopwords


train_file = codecs.open("./goods.data.seg.train", 'w', 'utf-8')
# val_file = codecs.open("./goods.data.seg.val", 'w', 'utf-8')
test_file = codecs.open("./goods.data.seg.test", 'w', 'utf-8')

jieba.suggest_freq('q币', True)
jieba.suggest_freq('Q币', True)
jieba.suggest_freq('qb', True)
jieba.suggest_freq('红钻', True)
jieba.suggest_freq('绿钻', True)
jieba.suggest_freq('黄钻', True)
jieba.suggest_freq('蓝钻', True)
jieba.suggest_freq('紫钻', True)
jieba.suggest_freq('黑钻', True)

stopwords = stopwordslist("./stopwords/chinese_stops.txt")
for index, row in train_csv.iterrows():

    seg_text = jieba.cut(row["ITEM_NAME"].replace("\t", " ").replace("\n", " "), cut_all=False)

    seg_text_filter = [word for word in seg_text if word not in stopwords]
    outline = "".join(seg_text_filter)

    rand_num = random.random()

    if rand_num < 0.9:
        if len(row['ITEM_NAME']):
            outline = outline + "\t__label__" + row["TYPE"] + "\n"
            train_file.write(outline)
            train_file.flush()
    else:
        if len(row['ITEM_NAME']):
            outline = outline + "\t__label__" + row["TYPE"] + "\n"
            test_file.write(outline)
            test_file.flush()

train_file.close()
test_file.close()
