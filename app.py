from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort
import fasttext
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

    @app.route("/upload", methods=["GET"])
    def upload_one():
        # 获取get的参数
        pn = request.args.get("productname")
        if not fasttext_model:
            fasttext_model = fasttext.load_model("./algorithm/a.bin", label_prefix='__label__')
        try:
            result = fasttext_model.predict(pn)[0][0]  # 是否要返回前三个？
            return {
                "result": result,
                "code": 200
            }
        except err:
            return {
                "code": 500,
                "message": "错误发生在预测时"
            }
    return app


