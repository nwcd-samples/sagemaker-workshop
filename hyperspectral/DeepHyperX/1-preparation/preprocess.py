import numpy as np
import os
import scipy.io
import spectral.io.envi as envi


DATASET_NAME = 'leaf'
NEW_DATA_PATH = os.path.join(os.getcwd(), "Datasets/"+DATASET_NAME)  # 存放数据路径 patch是文件夹名称


"""
temp_split: 对数据进行拆分
"""
def loadData(flieName, dataIndex, temp_split=4):
    
    print("------------  loadData  ", dataIndex)
    # 原始数据路径
    DATA_PATH = os.path.join(os.getcwd(), flieName)

    index = str(dataIndex)
    data = envi.open( os.path.join(DATA_PATH, "{}.hdr".format(index)) ,os.path.join(DATA_PATH, "{}.dat".format(index)))
    mask_data = envi.open( os.path.join(DATA_PATH, "mask_{}.hdr".format(index)) ,os.path.join(DATA_PATH, "mask_{}.tiff".format(index)))

    HEIGHT = data.shape[0] //temp_split
    WIDTH = data.shape[1] //temp_split
    BAND = data.shape[2]
#     BAND = BAND_SIZE
    new_shape=(BAND,HEIGHT,WIDTH)
    new_data = np.zeros(new_shape, dtype = float)
    label = np.zeros((HEIGHT, WIDTH), dtype = int)
    

    sample_count = 0
    for h in range(HEIGHT): 
        for w in range(WIDTH):
            x = h*temp_split
            y = w*temp_split
            for b in range(BAND):
                new_data[b][h][w] = data[x,y][b]

            if(sum(mask_data[x, y])  > 0.01 ):
                label[h][w] = dataIndex 
                sample_count += 1
            else:
                label[h][w] = 0
    
    
    new_data = np.transpose(new_data, (1, 2, 0))  # 将通道数提前，便于数组处理操作
    print("sample_count = {} ".format(sample_count))
    print("data shape : ", new_data.shape)
    print("label shape : ", label.shape)
    return new_data, label

if not os.path.exists(NEW_DATA_PATH):
    print("  ", NEW_DATA_PATH)
    os.makedirs(NEW_DATA_PATH)
    print("create dataset dir success.")

data1, label1 = loadData("dataset", 1)
data2, label2 = loadData("dataset", 2)
data3, label3 = loadData("dataset", 3)
data4, label4 = loadData("dataset", 4)



X1 = np.hstack((data1, data2))
X2 = np.hstack((data3, data4))

gt1 = np.hstack((label1, label2))
gt2 = np.hstack((label3, label4))

X = np.vstack((X1, X2))

gt = np.vstack((gt1, gt2))




    
    
train_dict, test_dict = {}, {}
train_dict[DATASET_NAME] = X
file_name = "{}.mat".format(DATASET_NAME) 
scipy.io.savemat(os.path.join(NEW_DATA_PATH, file_name), train_dict)
test_dict["{}_gt".format(DATASET_NAME)] = gt
file_name = "{}_gt.mat".format(DATASET_NAME)
scipy.io.savemat(os.path.join(NEW_DATA_PATH, file_name), test_dict)
print("Save target data success ---------------------------------\n")