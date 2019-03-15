import fasttext
classifier1 = fasttext.load_model("./a.bin", label_prefix='__label__')
#classifier2 = fasttext.load_model("./b.bin", label_prefix='__label__')

result = classifier1.test("./highrate.test")
print("Precision on testing data:", result.precision)
result = classifier1.test("./highrate.train")
print("Precision on training data:", result.precision)
