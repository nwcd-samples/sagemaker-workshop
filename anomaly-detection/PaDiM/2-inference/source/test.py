import _thread
from inference import DetectionSystem

detection = DetectionSystem()
def asyncPredict(current_data_dir,bucket,path):
    inference_result = detection.predict(current_data_dir,bucket,path)

current_data_dir="/opt/ml/data_dir/202103180920572600"
_thread.start_new_thread( asyncPredict, (current_data_dir,"junzhong","result/ad") )
while 1:
   pass