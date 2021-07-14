# 训练数据输入
在有大量数据需要进行训练时，如果SageMaker直接从S3下载数据，耗时较长。可通过FSx for Lustre把S3数据作为SageMaker的训练数据输入，以解决直接从S3上下载训练数据耗时过长问题。也支持从EFS载入数据。  
[FSx](FSx.ipynb)  
[EFS](EFS.ipynb)