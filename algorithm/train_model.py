import logging
import fasttext
import pandas as pd
import codecs
import time

basedir = './'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = fasttext.supervised(basedir + "highrate.train", basedir + "a", label_prefix="__label__", dim=180, ws=5, neg=5, epoch=150, min_count=2, lr=0.1, lr_update_rate=1000, bucket=100000)
