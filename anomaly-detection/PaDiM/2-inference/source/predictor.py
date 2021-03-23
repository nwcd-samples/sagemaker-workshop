import os
import sys
import boto3
import flask
import json
import shutil
import time,datetime
import random
from inference import DetectionSystem
import _thread

DEBUG = False

# The flask app for serving predictions
app = flask.Flask(__name__)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    #health = boto3.client('s3') is not None  # You can insert a health check here

    #status = 200 if health else 404
    status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/')
def hello_world():
    return 'PaDiM endpoint'


@app.route('/invocations', methods=['POST'])
def invocations():
    content_type = flask.request.content_type
    if content_type != 'application/json' :
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')
    
    tt = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for i in range(0,5):
        randomstr = str(random.randint(1000,9999))
        current_data_dir = os.path.join(init_data_dir,tt+randomstr)
        if not os.path.exists(current_data_dir):
            try:
                os.mkdir(current_data_dir)
                break
            except FileExistsError:
                logger.info("Dir Exist."+current_data_dir)
    else:
        return flask.Response(response='Make dir error', status=500, mimetype='text/plain')
    
    data = flask.request.data.decode('utf-8')
    logger.info("invocations params [{}]".format(data))
    try:
        data = json.loads(data)
    except:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    bucket = data['bucket']
    for image_uri in data['image_uri']:
        download_file_name = image_uri.split('/')[-1]
        download_file_name = os.path.join(current_data_dir, download_file_name)
        s3_client.download_file(bucket, image_uri, download_file_name)
    upload_bucket = data['upload_bucket']
    upload_path = data['upload_path']

    #inference_result = detection.predict(current_data_dir)
    #shutil.rmtree(current_data_dir)
    _thread.start_new_thread( asyncPredict, (current_data_dir,upload_bucket,upload_path) )
    
    _payload = json.dumps({'code': 1, 'msg': 'async Predict'})
    return flask.Response(response=_payload, status=200, mimetype='application/json')

def asyncPredict(current_data_dir,bucket,path):
    inference_result = detection.predict(current_data_dir,bucket,path)

#---------------------------------------
init_data_dir = '/opt/ml/data_dir'

if not os.path.exists(init_data_dir):
    try:
        os.mkdir(init_data_dir)
    except FileExistsError:
        logger.info("Dir Exist.")

s3_client = boto3.client("s3")
detection = DetectionSystem()
#---------------------------------------


if __name__ == '__main__':
    app.run()