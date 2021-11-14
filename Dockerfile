FROM python:3.7.9-slim

WORKDIR /usr/app/src

RUN apt-get update -y && \
    apt-get install build-essential cmake pkg-config -y
RUN pip install dlib==19.22.1
RUN apt install -y libgl1-mesa-glx

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app ./

CMD ["sh", "-c", "streamlit run --server.port $PORT /usr/app/src/frace.py"]