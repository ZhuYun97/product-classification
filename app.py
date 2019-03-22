from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, jsonify, make_response
import fasttext
import traceback
import os
from werkzeug.utils import secure_filename
import jieba
import re
from utils import stopwordslist
import pandas as pd


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


def create_app():
    app = Flask(__name__)
    # 返回的json可以显示中文
    app.config['JSON_AS_ASCII'] = False
    fasttext_model = None
    # 选择模型
    fm_name = "b.bin"
    partial_common_words = None
    all_common_words = None

    add_corpus()
    try:
        fasttext_model = fasttext.load_model("./algorithm/save/" + fm_name, label_prefix='__label__')
        # g.fasttext_model = fasttext_model  这样会太费内存把
        partial_common_words = [" ", '\r']
        with open('./algorithm/processed_data/partial_common_words.txt', 'r', encoding="utf-8") as file:
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

    stopwords = stopwordslist("./algorithm/stopwords/chinese_stops.txt")

    def deal(raw_text):
        # 添加语料

        seg_text = jieba.cut(raw_text.replace("\t", " ").replace("\n", " "), cut_all=False)
        seg_text_filter = [word for word in seg_text if
                           word not in stopwords and word in partial_common_words and not re.match("^\d*元$|^\d*$",
                                                                                                   word)]
        text = " ".join(seg_text_filter)
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
                fasttext_model = fasttext.load_model("./algorithm/save/" + fm_name, label_prefix='__label__')
            pn = deal(pn)
            # 输入文本过短

            # resultlist = fasttext_model.predict([pn], 5)
            try:
                item = fasttext_model.predict_proba([pn], 1)[0][0]
            except Exception as e:
                json_data = jsonify({
                    "pn": pn,
                    "code": 1,
                    "message": "输入不规范，请重新输入"
                })
                response = make_response(json_data)
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
                response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
                return response
            print("item:", item)
            if len(item) != 2:
                raise Exception("未知错误")
            json_data = jsonify({
                "pn": pn,
                "code": 0,
                "type": item[0],
                "prob": item[1]
            })
            response = make_response(json_data)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
            response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
            return response
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
                "message": str(e)
            })


    def deal_result(results):
        dealed = []

        for key, value in results.items():
            words = key.split("--")
            sign = False  # 判断一级商品名存不存在
            for item in dealed:
                if words[0] == item["name"]:
                    sign = True
                    sign2 = False
                    for child in item["children"]:
                        if words[1] == child["name"]:
                            sign2 = True
                            child["children"].append({"name": words[2], "value": value})
                        else:
                            continue
                    if not sign2:
                        tmp = {"name": words[1], "children": [{"name": words[2], "value": value}]}
                        item["children"].append(tmp)
                else:
                    continue
            if not sign:
                dealed.append({"name": words[0], "children": [{"name": words[1], "children": [{"name": words[2],
                                                                                               "value": value}]}]})
        return {"name": "flare", "children": dealed}


    def predict(testdata, encoding, path, fasttext_model = fasttext_model):
        try:
            results = dict()
            perfect_results = {
                "name": "flare",
                "children": []
            }
            for index, row in testdata.iterrows():
                raw_text = row["ITEM_NAME"]
                text = deal(raw_text)
                label = "Unknown--Unknown--Unknown"
                try:
                    if fasttext_model:
                        label = fasttext_model.predict([text], 1)[0][0]
                    else:
                        fasttext_model = fasttext.load_model("./algorithm/save/" + fm_name, label_prefix='__label__')
                        label = fasttext_model.predict([text], 1)[0][0]
                    # row["TYPE"] = label
                except Exception as e:
                    print(1, str(e))
                finally:
                    row["TYPE"] = label
                    if label in results.keys():
                        results[label] += 1
                    else:
                        results[label] = 1
            print("before saving")
            if encoding == "utf-8":
                testdata.to_csv(path, encoding=encoding, sep=",", index=False)
            else:
                testdata.to_csv(path, encoding=encoding, sep="\t", index=False)
            print("after saving")

            return deal_result(results)
        except Exception as e:
            print(2, str(e))


    @app.route("/uploadfile", methods=["POST"])
    def upload_csv():
        import time
        unique = str(int(time.time()))
        filename = unique + ".csv"
        testdata = None
        try:
            encoding = request.form.get("encoding")
            f = request.files['file_test']
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'static/uploads', f.filename)
            resultname = os.path.join(basepath, 'static/uploads', filename)
            f.save(upload_path)
            # 根据后缀名来决定读取方式
            if(f.filename.find(".") == -1):
                raise Exception("文件类型错误，请选择xls，csv，tsv类型的文件")
            suffix = f.filename.split(".")[-1]
            if suffix == 'xls' or suffix == "xlsx":
                try:
                    if encoding == "utf-8":
                        testdata = pd.read_excel(upload_path, sep=",", encoding = encoding)
                    else:
                        testdata = pd.read_excel(upload_path, sep="\t", encoding=encoding)
                    results = predict(testdata, encoding, resultname)
                # return send_from_directory('static/uploads/', unique + ".csv", as_attachment=True)
                    json_data = jsonify({
                        "code": 0,
                        "filename": resultname,
                        "results": results
                    })
                    response = make_response(json_data)
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
                    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
                    return response
                except Exception as e:
                    return jsonify({
                        "code": 2,
                        "message": "编码不正确，请选择其他的编码方式",
                        "error": str(e)
                    })
            elif suffix == 'csv' or suffix == 'tsv':
                try:
                    if encoding == "utf-8":
                        testdata = pd.read_csv(upload_path, sep=",", encoding=encoding)
                    else:
                        testdata = pd.read_csv(upload_path, sep="\t", encoding=encoding)
                    results = predict(testdata, encoding, resultname)
                    # return send_from_directory('static/uploads/', unique + ".csv", as_attachment=True)
                    json_data = jsonify({
                        "code": 0,
                        "filename": resultname,
                        "results": results
                    })
                    response = make_response(json_data)
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
                    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
                    return response
                except Exception as e:
                    return jsonify({
                        "code": 2,
                        "message": "编码不正确，请选择其他的编码方式",
                        "error": str(e)
                    })
            else:
                raise Exception("文件类型错误，请选择xls，csv，tsv类型的文件")

        except Exception as e:
            return jsonify({
                "code": 1,
                "message": str(e)
            })

    @app.route("/download/<path:filename>")
    def downloader(filename):
        return send_from_directory('static/uploads/', filename, as_attachment=True)

    return app


