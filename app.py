from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, jsonify
import fasttext
import traceback
import os
# from werkzeug.utils import secure_filename
import jieba
# import csv



def create_app():
    app = Flask(__name__)
    fasttext_model = fasttext.load_model("./algorithm/a.bin", label_prefix='__label__')
    file_name = ""

    @app.route('/')
    def hello_world():
        return '欢迎来到猪事顺心小组的作品。\n该项目正在开发中，敬请期待！'

    @app.route("/upload")
    def upload_one(fasttext_model = fasttext_model):
        try:
            # 获取get的参数
            pn = request.args.get("productname")
            if not fasttext_model:
                fasttext_model = fasttext.load_model("./algorithm/a.bin", label_prefix='__label__')
            resultlist = fasttext_model.predict(pn)[0]  # 是否要返回前三个？
            if resultlist:
                result = resultlist[0]
                return jsonify({
                    "result": result,
                    "code": 0
                })
            else:
                return jsonify({
                    "result": "无法预测",
                    "code": 1
                })
        except Exception as e:
            traceback.print_exc()
            # 或者得到堆栈字符串信息
            info = traceback.format_exc()
            print(info, str(e))
            return jsonify({
                "code": 2,
                "message": "错误发生在预测时"
            })

    return app


