FROM ubuntu:18.04

COPY . /root

WORKDIR /root

RUN apt-get update

RUN apt-get install -y libboost-dev libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-math-dev libboost-test-dev libboost-python-dev zlib1g-dev cmake
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y vowpal-wabbit

RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
