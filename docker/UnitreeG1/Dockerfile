FROM osrf/ros:humble-desktop-jammy

RUN mkdir -p /root/.pip
COPY pip.conf /root/.pip/pip.conf
COPY sources.list /etc/apt/sources.list

RUN apt update && apt install -y \
wget \
cmake \
make

WORKDIR  /root
RUN wget https://mirrors.huaweicloud.com/python/3.8.19/Python-3.8.19.tar.xz && tar -xvf Python-3.8.19.tar.xz
WORKDIR  /root/Python-3.8.19
RUN ./configure --prefix=/usr/local && make && make install

RUN ln -s pip3 pip

WORKDIR /workspace/unitree_sdk2_python