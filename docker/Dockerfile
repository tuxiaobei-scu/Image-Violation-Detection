FROM docker.io/bitnami/pytorch:2.1.1


WORKDIR /app

USER root

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y libgl1 zbar-tools && rm -rf /var/lib/apt/lists/*

RUN pip install onnx transformers onnxruntime easyocr flask pillow numpy requests pyzbar opencv-python -i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*
RUN pip install hnswlib -i http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

COPY . .

RUN mkdir -p /app/tmp
RUN chmod -R 777 /app/tmp

EXPOSE 5500

CMD ["python", "onnx_run.py"]