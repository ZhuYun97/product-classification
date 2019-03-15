import logging
import fasttext
import pandas as pd
import codecs
import time

basedir = './'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = fasttext.supervised(basedir + "highrate.train", basedir + "b", label_prefix="__label__", dim=180, ws=5, neg=5, epoch=20, min_count=2, lr=0.1, lr_update_rate=1000, bucket=200000)
result = classifier.test("./highrate.test")
print("Precision on testing data:", result.precision)

classifier = fasttext.supervised(basedir + "highrate.train", basedir + "b", label_prefix="__label__", dim=180, ws=5, neg=5, epoch=20, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
result = classifier.test("./highrate.test")
print("Precision on testing data:", result.precision)

classifier = fasttext.supervised(basedir + "highrate.train", basedir + "b", label_prefix="__label__", dim=170, ws=5, neg=5, epoch=20, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
result = classifier.test("./highrate.test")
print("Precision on testing data:", result.precision)
"""
result2 = classifier1.test("./highrate.train")
print("Precision on training data:", result2.precision)
"""