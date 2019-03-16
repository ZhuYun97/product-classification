# wsgi.py
from app import create_app

from flask import Flask

application = create_app()

if __name__ == '__main__':
    # 返回的json可以显示中文
    application.config['JSON_AS_ASCII'] = False
    application.run()
