import logging
import fasttext
import pandas as pd
import codecs
import time

basedir = './'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = fasttext.supervised(basedir + "highrate.train", basedir + "b", label_prefix="__label__", dim=200, ws=5, neg=5, epoch=100, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
classifier = fasttext.supervised(basedir + "highrate.train", basedir + "c", label_prefix="__label__", dim=150, ws=5, neg=5, epoch=80, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
classifier = fasttext.supervised(basedir + "highrate.train", basedir + "d", label_prefix="__label__", dim=220, ws=5, neg=5, epoch=50, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
classifier = fasttext.supervised(basedir + "highrate.train", basedir + "e", label_prefix="__label__", dim=200, ws=5, neg=5, epoch=20, min_count=2, lr=0.1, lr_update_rate=1000, bucket=200000)
classifier = fasttext.supervised(basedir + "highrate.train", basedir + "f", label_prefix="__label__", dim=100, ws=5, neg=5, epoch=20, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
classifier = fasttext.supervised(basedir + "highrate.train", basedir + "b", label_prefix="__label__", dim=50, ws=5, neg=5, epoch=50, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)