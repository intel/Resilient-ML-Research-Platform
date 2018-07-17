#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0

# this file is for Docker host
export HADOOP_USER=hadoop
export DJANGO_USER=django
# try setup users with higher uid in Docker Host
export HADOOP_UID=3001
export DJANGO_UID=3002
export HADOOP_DIR=/home/$HADOOP_USER/docker
export WEB_DIR=/home/$DJANGO_USER/myml
export WEB_SRC=..


# Check if root or sudoer
if ! [ $EUID -eq 0 ] || ! [ $(whoami)=="root" ]; then
    echo "ERROR: please execute as sudo or root!!"
    exit -1
fi
        # userdel -r django userdel -r hadoop
# Create users: hadoop django for container user mapping =============================
# Check if user '$HADOOP_USER' exist 
huid=`id -u $HADOOP_USER 2> /dev/null` 
no_hadoop=$?
if [ $no_hadoop -eq 1 ]; then
    echo "INFO: Create user '$HADOOP_USER' with uid=$HADOOP_UID"
    useradd -m $HADOOP_USER -u $HADOOP_UID > /dev/null
else
    echo "INFO: User '$HADOOP_USER' exists with uid=$huid"        
fi

# Check if user '$DJANGO_USER' exist "
duid=`id -u $DJANGO_USER 2> /dev/null ` 
no_djanog=$?
if [ $no_djanog -eq 1 ]; then
    echo "INFO: Create user '$DJANGO_USER' with uid=$DJANGO_UID"
    useradd -m $DJANGO_USER -u $DJANGO_UID > /dev/null
else
    echo "INFO: User '$DJANGO_USER' exists with uid=$duid"        
fi
# clean up: sudo userdel -r hadoop;  sudo userdel -r django; 

# Create data/config dirs for container web -v mapping =============================
#  web folder
if ! [ -d $WEB_DIR ]; then
    echo "INFO: Create $WEB_DIR"        
    mkdir $WEB_DIR
else
    echo "INFO: $WEB_DIR exists"        
fi

# Copy web files to $WEB_DIR
if ! [ -d $WEB_DIR/atdml ]; then
    echo "INFO: Copy web files to $WEB_DIR"        
    #cp -p -r $WEB_SRC/* $WEB_DIR/.
    rsync -ar $WEB_SRC/* $WEB_DIR/.
    # change owner to $DJANGO_USER
    echo "INFO: Change owner of $WEB_DIR to $DJANGO_USER"        
    chown $DJANGO_USER.$DJANGO_USER $WEB_DIR/ -R
fi


#  hadoop/spark/mongo data/config folders
if ! [ -d $HADOOP_DIR ]; then
    echo "INFO: Create $HADOOP_DIR"        
    mkdir $HADOOP_DIR
else
    echo "INFO: $HADOOP_DIR exists"        
fi

# copy config/data files to $HADOOP_DIR
if ! [ -d $HADOOP_DIR/hadoopdata ]; then
    echo "INFO: Copy config/data files to $HADOOP_DIR"        
    cp -p -r ./node_config/* $HADOOP_DIR/.
    # change owner to $HADOOP_USER
    echo "INFO: Change owner of $HADOOP_DIR to $HADOOP_USER"        
    chown $HADOOP_USER.$HADOOP_USER $HADOOP_DIR/ -R
fi







