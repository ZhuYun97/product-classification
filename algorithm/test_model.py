import fasttext
classifier1 = fasttext.load_model("./a.bin", label_prefix='__label__')
#classifier2 = fasttext.load_model("./b.bin", label_prefix='__label__')

# result = classifier1.test("./highrate.test")
# print("Precision on testing data:", result.precision)
# result = classifier1.test("./highrate.train")
# print("Precision on training data:", result.precision)

print(classifier1.predict("影视编导专业考前辅导教程 赵西盈 中国传媒大学出版社 适用于广播电视编导导演、戏剧影视文")[0][0])
