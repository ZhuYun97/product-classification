import logging
import fasttext
import pandas as pd
import codecs
import time

basedir = './'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = fasttext.supervised(basedir + "highrate.train", basedir + "b", label_prefix="__label__", dim=200, ws=5, neg=5, epoch=200, min_count=2, lr=0.1, lr_update_rate=1000, bucket=200000)