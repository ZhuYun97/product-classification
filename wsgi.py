# wsgi.py
from app import create_app
from flask_cors import CORS


application = create_app()

if __name__ == '__main__':
    CORS(application, supports_credentials=True)
    CORS(application, resources=r'/*')
    application.run()
