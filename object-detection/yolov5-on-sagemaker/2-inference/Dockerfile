#ARG BASE_IMG=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.6.0-gpu-py36-cu101-ubuntu16.04
ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG} 

RUN apt-get update
RUN apt-get install -y --no-install-recommends nginx net-tools\
    && rm -rf /var/lib/apt/lists/*

RUN pip install flask gevent gunicorn boto3 -i https://opentuna.cn/pypi/web/simple/ && \
        rm -rf /root/.cache

# RUN mkdir /opt/ml/code
WORKDIR /opt/ml/code
COPY source ./

RUN pip install -r requirements.txt -i https://opentuna.cn/pypi/web/simple/
        
# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code/:${PATH}"

ENTRYPOINT ["python3"]