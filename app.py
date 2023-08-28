from flask import Flask
from flask_cors import CORS
from crawler import crawler
from rq import Connection, Worker
import redis
import os
from dotenv import load_dotenv
app = Flask(__name__)
CORS(app)
REDIS_URL = os.environ.get('REDIS_URL')
app.config['WTF_CSRF_ENABLED'] = True
app.config['REDIS_URL'] = REDIS_URL
app.config['QUEUES'] = ["default"]
app.register_blueprint(crawler)
if __name__ == '__main__':    
	app.run(debug=True)