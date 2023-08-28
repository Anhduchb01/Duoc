from rq import Connection, Worker
import redis
import os
REDIS_URL = os.environ.get('REDIS_URL')
redis_connection = redis.from_url(REDIS_URL)
with Connection(redis_connection):
    worker = Worker(["default"])
    worker.work()