# 说明

演示使用Sagemaker 封装图片分类算法, 使用TensorFlow-server 部署模型，并在客户端进行调用。 

# 使用步骤

##  本地测试
```
if [ ! -d "output" ];then
mkdir output
fi

python source/train.py --epoch_count=20 --batch_size=32 --input_dir='./cat-vs-dog' --output_dir='./output'

docker build -t sagemaker-demo .

docker run -p 8501:8501 -d sagemaker-demo

```


## 使用sagemaker 训练

打开 [train.ipynb](train.ipynb) 进行训练，然后打开[inference-custom-image.ipynb](inference-custom-image.ipynb)或[inference-default-image.ipynb](inference-default-image.ipynb)进行部署和使用

## 超级参数优化
打开 [hyperparameter-tuning.ipynb](hyperparameter-tuning.ipynb) 进行参数优化