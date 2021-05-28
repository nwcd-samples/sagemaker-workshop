import os
import sys
import boto3
import flask
import json
import shutil
import time
import random
from detect import detect

DEBUG = False

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
    return 'YOLOv5 endpoint'


@app.route('/invocations', methods=['POST'])
def invocations():
    data = None
    #解析json，
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        logger.info("invocations params [{}]".format(data))
        bucket = data['bucket']
        image_uri = data['image_uri']
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')    
    
    tt = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for i in range(0,5):
        current_output_dir = os.path.join(init_output_dir,tt+str(random.randint(1000,9999)))
        if not os.path.exists(current_output_dir):
            try:
                os.mkdir(current_output_dir)
                break
            except FileExistsError:
                logger.info("Dir Exist."+current_output_dir)
    else:
        return flask.Response(response='Make dir error', status=500, mimetype='text/plain')

    download_file_name = image_uri.split('/')[-1]
    download_file_name = os.path.join(current_output_dir, download_file_name)
    s3_client.download_file(bucket, image_uri, download_file_name)
    
    img_size = 640
    if "img_size" in data:
        img_size = data["img_size"]
    inference_result = detect(download_file_name, img_size)

    
    _payload = json.dumps({'status': 500, 'message': 'YOLOv5 failed!'})
    if inference_result:
         _payload = json.dumps(inference_result)
    
    
    shutil.rmtree(current_output_dir)
    
    return flask.Response(response=_payload, status=200, mimetype='application/json')


#---------------------------------------
init_output_dir = '/opt/ml/output_dir'
if not os.path.exists(init_output_dir):
    try:
        os.mkdir(init_output_dir)
    except FileExistsError:
        logger.info("Dir Exist.")

#load model
source_file = '/opt/ml/model/runs/train/exp/weights/best.pt'
destination_file = "yolov5s.pt"
if os.path.isfile(source_file) and not os.path.isfile(destination_file):
    shutil.copy(source_file,destination_file)
    logger.info("Model file copied.")
else:
    logger.info("Model file not copy.")

s3_client = boto3.client('s3')

if __name__ == '__main__':
    app.run()