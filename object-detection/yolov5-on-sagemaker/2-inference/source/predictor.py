# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import os
import sys
import boto3
import flask
import json
import shutil
import time,datetime
from detect import detect

DEBUG = False

# The flask app for serving predictions
app = flask.Flask(__name__)

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
        print("  invocations params [{}]".format(data))
        bucket = data['bucket']
        image_uri = data['image_uri']
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')    
    
    download_file_name = image_uri.split('/')[-1]
    #s3_client.download_file(bucket, image_uri, download_file_name)

    tt = time.mktime(datetime.datetime.now().timetuple())
    args_output_dir = os.path.join(init_output_dir,  str(int(tt)))
    if not os.path.exists(args_output_dir):
        os.mkdir(args_output_dir)

    download_file_name = os.path.join(args_output_dir, download_file_name)
    s3_client.download_file(bucket, image_uri, download_file_name)
    
    #print("download_file_name : {} ".format(download_file_name))
    img_size = 640
    if "img_size" in data:
        img_size = data["img_size"]
    inference_result = detect(download_file_name, img_size)

    
    _payload = json.dumps({'status': 500, 'message': 'YOLOv5 failed!'})
    if inference_result:
         _payload = json.dumps(inference_result)
    
    
    shutil.rmtree(args_output_dir)
    
    return flask.Response(response=_payload, status=200, mimetype='application/json')




#---------------------------------------
init_output_dir = '/opt/ml/output_dir'

if not os.path.exists(init_output_dir):
    os.mkdir(init_output_dir)
else:
    print("-------------init_output_dir ", init_output_dir)

s3_client = boto3.client('s3')
#---------------------------------------


if __name__ == '__main__':
    app.run()
    """
    print("server ------run")
    
    output_dir = os.path.join('./', 'temp')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    ocr_main('../../../sample_data/images/test.jpg', output_dir)
    """ 