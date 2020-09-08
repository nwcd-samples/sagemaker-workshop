import os
import shutil
import numpy as np
import glob
import random
    
    
def processing_data(training_dir, validation_dir, testing_dir, validation_rate=0.1,  testing_rate=0.1):
    class_equal,class_list = check_class(training_dir, validation_dir)
    if not class_equal:
        files = os.listdir(training_dir)
        for file in files :
            dir_path = os.path.join(os.path.join(training_dir, file))
            if os.path.isdir(dir_path):
                print('处理 ：', dir_path)
                get_filelist(dir_path, file, validation_dir, validation_rate, testing_dir,testing_rate)
                
            
    training_count = get_all_count(training_dir)
    validation_count = get_all_count(validation_dir)
    testing_count = get_all_count(testing_dir)
    
    print("==================================================")
    print('training count        : ', training_count)
    print('validation count      : ', validation_count)
    print('testing count         : ', testing_count)
    print("class: ", class_list)
    print("==================================================")
    return class_list,training_count
    

def check_class(training_dir, validation_dir):
    class_list = []
    for file in os.listdir(training_dir):
        if os.path.isdir(os.path.join(training_dir , file)):
            class_list.append(file)
    class_list.sort()
    
    if not os.path.exists(validation_dir):
        return False,class_list
        
    class_list_validation = []
    for file in os.listdir(validation_dir):
        if os.path.isdir(os.path.join(validation_dir , file)):
            class_list_validation.append(file)
    class_list_validation.sort()
    
    class_equal = (class_list == class_list_validation)
    return class_equal,class_list
    
    
def get_all_count(dir_path):
    files = os.listdir(dir_path)
    count = 0 
    for file in files :
        if os.path.isdir(dir_path):
            label_dir = os.path.join(dir_path, file)
            images = os.listdir(label_dir)
            tmp_count = len(images)
            #print('{}   {}'.format(label_dir, tmp_count))
            count += tmp_count
    return count


    
def move_file(file_path, target_path, class_name, item ):
    target_dir = os.path.join(target_path, class_name)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    shutil.move(file_path, os.path.join(target_dir, item))

def get_filelist(dir_path, class_name, validation_dir, validation_rate, testing_dir,testing_rate):
    files = os.listdir(dir_path)
    random.shuffle(files)
    count = len(files)
    
    validation_count = int(count * validation_rate)
    testing_count = int(count * testing_rate)
    
    validation_list = files[0:validation_count]
    testing_list = files[validation_count: validation_count + testing_count]
    training_list = files[validation_count + testing_count: ]
    
    for item in validation_list:
        move_file(os.path.join(os.path.join(dir_path, item)) , validation_dir, class_name, item ) 
    for item in testing_list:
        move_file(os.path.join(os.path.join(dir_path, item)) , testing_dir,class_name, item)
    

    
    
if __name__ == '__main__':
    processing_data("/opt/ml/input/data/training","/opt/ml/input/data/validation","/opt/ml/input/data/testing")