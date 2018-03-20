#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
CLIENT_IP=0.0.0.0
WEB_PORT=8000

BASE_DIR=/home/django/myml
cd $BASE_DIR

# Hadoop and Java env
export HADOOP_USER_NAME=hadoop
export HADOOP_HOME=/home/hadoop/hadoop_latest 
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export YARN_HOME=$HADOOP_HOME
export JAVA_HOME=/usr/java/default
export PATH=$PATH:$HADOOP_HOME/bin:$JAVA_HOME/bin:/home/hadoop/spark_latest/bin
#export _JAVA_OPTIONS="-Xmx2g"
export THEANO_FLAGS=mode=FAST_RUN,floatX=float32

# adding PATH for icc and its libs etc
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

# start web here
python manage.py runserver $CLIENT_IP:$WEB_PORT
