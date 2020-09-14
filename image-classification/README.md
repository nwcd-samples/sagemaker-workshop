# 利用Amazon SageMaker内置算法进行图片分类
Image-classification-lst-format.ipynb演示了利用Amazon SageMaker内置算法进行图片分类模型的训练和部署。

## 启动Amazon SageMaker笔记本实例
通过以下步骤启动Amazon SageMaker的笔记本实例
* 访问SageMaker主页，点击左边栏目笔记本实例链接
* 创建笔记本实例
* 当笔记本实例处于InService状态时，可以通过点击JupyterLab链接进入到实例中

## 上传源文件到笔记本实例
点击左上角上传按钮，将Image-classification-lst-format.ipynb文件上传到笔记本实例中。

## 升级相应Kernel中sagemaker版本
* conda env list
* source  activate mxnet_p36 
* pip install sagemaker --upgrade <br>
执行完以上命令重启kenrnel


## 运行笔记本实例中的每个Cell
阅读每个Cell运行相关程序进行模型训练和推理