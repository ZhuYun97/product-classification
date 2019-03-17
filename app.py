from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, jsonify
import fasttext
import traceback
import os
# from werkzeug.utils import secure_filename
import jieba
import re
# import csv



def create_app():
    app = Flask(__name__)
    # 返回的json可以显示中文
    app.config['JSON_AS_ASCII'] = False
    fasttext_model = None
    fasttext_model2 = None
    fasttext_model3 = None
    fasttext_model4 = None
    partial_common_words = None
    all_common_words = None
    try:
        fasttext_model = fasttext.load_model("./algorithm/e.bin", label_prefix='__label__')
        partial_common_words = [" ", '\r']
        with open('./algorithm/partial_common_words.txt', 'r', encoding = "utf-8") as file:
            lines = file.readlines()
            for line in lines:
                # ["a b c d"] 所有数据得是一行的
                partial_common_words += line.split()
        # 懒加载
        all_common_words = [" ", '\r']
    except Exception:
        return jsonify({
            "code": 3,
            "message": "加载数据出错"
        })
    file_name = ""

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

    stopwords = stopwordslist("./algorithm/stopwords/chinese_stops.txt")

    def deal(raw_text):
        seg_text = jieba.cut(raw_text.replace("\t", " ").replace("\n", " "), cut_all=False)

        seg_text_filter = [word for word in seg_text if
                           word not in stopwords and word in partial_common_words and not re.match("^\d*元$|^\d*$",
                                                                                                   word)]
        text = "".join(seg_text_filter)
        return text

    @app.route('/')
    def hello_world():
        return '欢迎来到猪事顺心小组的作品。\n该项目正在开发中，敬请期待！'

    @app.route("/upload", methods=["GET"])
    def upload_one(fasttext_model = fasttext_model):
        try:
            # 获取get的参数
            pn = request.args.get("productname")
            if not fasttext_model:
                fasttext_model = fasttext.load_model("./algorithm/e.bin", label_prefix='__label__')
            pn = deal(pn)
            # 输入文本过短

            resultlist = fasttext_model.predict([pn], 5)
            prolist = fasttext_model.predict_proba([pn], 5)
            return jsonify({
                "result": resultlist,
                "problist": prolist
            })
            # if resultlist:
            #     result = resultlist[0]
            #     return jsonify({
            #         # "words": partial_common_words,
            #         "pn": pn,
            #         "result_a": result,
            #         "code": 0
            #     })
            # else:
            #     return jsonify({
            #         # "words": partial_common_words,
            #         "pn": pn,
            #         "result": "无法预测",
            #         "code": 1
            #     })
        except Exception as e:
            traceback.print_exc()
            # 或者得到堆栈字符串信息
            info = traceback.format_exc()
            print(info, str(e))
            return jsonify({
                "code": 2,
                "message": "错误发生在预测时"
            })

    # @app.route("/upload", methods=["POST"])
    # def upload_csv(fasttext_model = fasttext_model):
    #     try:
    #         f = request.files['file_test']
    #         basepath = os.path.dirname(__file__)
    #         upload_path = os.path.join(basepath, 'static/uploads', f.filename)
    #         f.save(upload_path)
    #         test(f.filename)  # 调用测试函数
    #         return send_from_directory('static/uploads', 'result.csv', as_attachment=True)
    # except Exception:
    #         pass

    return app


