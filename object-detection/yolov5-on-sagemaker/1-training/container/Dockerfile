#ARG BASE_IMG=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04
ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG} 

ENV PATH="/opt/code:${PATH}"

COPY sources.list /etc/apt/

#RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add -

RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
    jq

## fix /usr/local/cuda-10.0/compat/libcuda.so
## RUN bash -c 'echo "/usr/local/cuda-10.0/compat" > /etc/ld.so.conf.d/cuda.conf'
RUN ldconfig -v


WORKDIR /opt/code
COPY dockersource ./
RUN pip install -r /opt/code/requirements.txt -i https://opentuna.cn/pypi/web/simple/
## https://github.com/aws/sagemaker-pytorch-training-toolkit/issues/143#issuecomment-566776288
## https://github.com/aws/sagemaker-pytorch-training-toolkit/blob/upgrade-training-toolkit/docker/build_artifacts/start_with_right_hostname.sh
## https://github.com/aws/deep-learning-containers/blob/v2.0-pt-1.5.1-py36/pytorch/training/docker/1.5.1/py3/Dockerfile.gpu#L181
COPY changehostname.c /opt/code
COPY start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh
COPY train /opt/code