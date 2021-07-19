FROM ubuntu:latest

RUN apt-get update && apt-get -y update
RUN apt-get install -y --no-install-recommends build-essential python3.6 python3-pip python3-dev
RUN pip3 -q install pip --upgrade

RUN mkdir src
WORKDIR src/
COPY ./requirements.txt .
COPY ./init.py .

RUN pip3 install -r requirements.txt --no-cache-dir

RUN python3 ./init.py
