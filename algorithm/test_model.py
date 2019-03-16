import fasttext
import jieba
import re
import codecs
classifier1 = fasttext.load_model("./a.bin", label_prefix='__label__')
#classifier2 = fasttext.load_model("./b.bin", label_prefix='__label__')

partial_common_words = [" ", '\r']
with open('./partial_common_words.txt', 'r', encoding = "ISO-8859-1") as file:
    lines = file.readlines()
    for line in lines:
        # ["a b c d"] 所有数据得是一行的
        partial_common_words += line.split()

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

# result = classifier1.test("./highrate.test")
# print("Precision on testing data:", result.precision)
# result = classifier1.test("./highrate.train")
# print("Precision on training data:", result.precision)


def deal(raw_text):
    seg_text = jieba.cut(raw_text.replace("\t", " ").replace("\n", " "), cut_all=False)

    seg_text_filter = [word for word in seg_text if
                       word not in stopwords and word in partial_common_words and not re.match("^\d*元$|^\d*$", word)]
    text = "".join(seg_text_filter)
    return text


raw_text1 = "影视编导专业考前辅导教程 赵西盈 中国传媒大学出版社 适用于广播电视编导导演、戏剧影视文"
text1 = deal(raw_text1)

result = classifier1.predict("QB充值")
print("----------QB充值-----------")
print(result)
print(result[0][0])
print("----------兰亭集序-----------")
result = classifier1.predict("兰亭集序")
print(result)
print(result[0][0])
print("---------------------")
result = classifier1.predict("")
print(result)
print(result[0][0])
print("----------影视-----------")
result = classifier1.predict(raw_text1)
print(result)
print(result[0][0])
result = classifier1.predict(text1)
print(result)
print(result[0][0])


