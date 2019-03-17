import jieba


def add_corpus():
    jieba.suggest_freq('q币', True)
    jieba.suggest_freq('Q币', True)
    jieba.suggest_freq('QQ', True)
    jieba.suggest_freq('qb', True)
    jieba.suggest_freq('红钻', True)
    jieba.suggest_freq('绿钻', True)
    jieba.suggest_freq('黄钻', True)
    jieba.suggest_freq('蓝钻', True)
    jieba.suggest_freq('紫钻', True)
    jieba.suggest_freq('黑钻', True)