# Amazon SageMaker Workshop
利用Amazon SageMaker进行机器学习和深度学习开发。
## 免责声明
建议测试过程中使用此方案，生产环境使用请自行考虑评估。

当您对方案需要进一步的沟通和反馈后，可以联系 nwcd_labs@nwcdcloud.cn 获得更进一步的支持。

欢迎联系参与方案共建和提交方案需求, 也欢迎在 github 项目 issue 中留言反馈 bugs。

## 项目说明
本项目将以深度学习中的常用场景，介绍如何使用Amazon SageMaker进行模型训练和推理部署。

该项目包含以下内容：
* [图片分类image-classification](image-classification)，使用Amazon SageMaker内置的图片分类算法进行模型训练和部署
* [对象检测object-detection](object-detection)，使用YOLOv5算法演示对象检测
* [封装自定义算法encapsulation](encapsulation)，使用自定义算法，通过Amazon SageMaker进行封装在AWS平台上进行模型训练和部署
* [Java SDK2调用推理示例](Java2)(推荐)、[Java SDK1调用推理示例](Java)

## 准备工作
为了使用Amazon SageMaker您只需要拥有一个AWS的账号，我们就可以实践起来。

## 常见问题
### 升级相应Kernel中sagemaker版本
以升级mxnet_p36 kernal中sagemaker为例
```
source  activate mxnet_p36 
pip install sagemaker --upgrade
```
执行完以上命令重启kenrnel