from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort
import os
# from werkzeug.utils import secure_filename
import jieba
# import csv

app = Flask(__name__)
fasttext_model = None
file_name=""


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/upload", methods="GET")
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
    except Exception, err:
        return {
            "code": 500,
            "message": "错误发生在预测时"
        }

if __name__ == '__main__':
    fasttext_model = fasttext.load_model("./algorithm/a.bin", label_prefix='__label__')
    app.run(host='0.0.0.0',port=3000)

