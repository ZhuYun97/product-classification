import fasttext
import jieba
import re
import codecs
# classifier1 = fasttext.load_model("./c.bin", label_prefix='__label__')
import time
since = time.time()
classifier1 = fasttext.load_model("./b.bin", label_prefix='__label__')
time_elapsed = time.time() - since
print('The fasttext model is loaded using {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

since = time.time()
result = classifier1.test("./highrate.test")
time_elapsed = time.time() - since
print('The prediction is finished using {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print("Precision on testing data:", result.precision)
result = classifier1.test("./highrate.train")
print("Precision on training data:", result.precision)


# partial_common_words = [" ", '\r']
# with open('./partial_common_words.txt', 'r', encoding = "utf-8") as file:
#     lines = file.readlines()
#     for line in lines:
#         # ["a b c d"] 所有数据得是一行的
#         partial_common_words += line.split()
#
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     stopwords.append("★")
#     stopwords.append("[")
#     stopwords.append("]")
#     stopwords.append("【")
#     stopwords.append("】")
#     stopwords.append("(")
#     stopwords.append(")")
#     stopwords.append("/")
#     stopwords.append("-")
#     stopwords.append("（")
#     stopwords.append("）")
#     stopwords.append("／")
#     stopwords.append("-")
#     stopwords.append("+")
#     stopwords.append("*")
#     stopwords.append("|")
#     stopwords.append("·")
#     stopwords.append(":")
#     stopwords.append("…")
#     stopwords.append("’")
#     stopwords.append("'")
#     stopwords.append('"')
#     stopwords.append('◆')
#      #  空格不能去，否则准确率极低
#     return stopwords
#
# stopwords = stopwordslist("./stopwords/chinese_stops.txt")
#
# # result = classifier1.test("./highrate.test")
# # print("Precision on testing data:", result.precision)
# # result = classifier1.test("./highrate.train")
# # print("Precision on training data:", result.precision)
#
#
# def deal(raw_text):
#     seg_text = jieba.cut(raw_text.replace("\t", " ").replace("\n", " "), cut_all=False)
#
#     seg_text_filter = [word for word in seg_text if
#                        word not in stopwords and word in partial_common_words and not re.match("^\d*元$|^\d*$", word)]
#     text = "".join(seg_text_filter)
#     return text
#
#
# raw_text1 = "影视编导专业考前辅导教程 赵西盈 中国传媒大学出版社 适用于广播电视编导导演、戏剧影视文"
# text1 = deal(raw_text1)
# raw_text2 = "2016春季新款舒适透气韩版气垫内增高女鞋圆头套脚平底鞋运动鞋休闲单鞋子 枪色 35"
# text2 = deal(raw_text2)
# raw_text3 = "apphome 手机镜头 8倍变焦摄像头套装 广角微距鱼眼自拍镜头 出游必备拍照神器 iphone6/6s"
# text3 = deal(raw_text3)
# raw_text4 = "【618】父亲节礼物送朋友实用咖啡杯套装马克杯子创意成套杯子陶瓷叠叠杯带铁架无盖杯早餐 粉嘟嘟"
# text4 = deal(raw_text4)
#
#
# print("----------鞋子-----------")
# result = classifier1.predict(raw_text2)
# print(result[0][0])
# result = classifier1.predict(text2)
# print(result[0][0])
# print("----------书-----------")
# result = classifier1.predict(raw_text1)
# print(result[0][0])
# result = classifier1.predict(text1)
# print(result[0][0])
# print("--------- ------------")
# result = classifier1.predict(" ")
# print(result[0][0])
# print("----------手机-----------")
# result = classifier1.predict(raw_text3)
# print(result[0][0])
# result = classifier1.predict(text3)
# print(result[0][0])
# print("----------礼品-----------")
# result = classifier1.predict(raw_text4)
# print(result[0][0])
# result = classifier1.predict(text4)
# print(result[0][0])



