#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0

# load config file
sed '/=/!d;/\[\.*/d;/^ *#/d;' < app.config > /tmp/$$.tmp
source /tmp/$$.tmp
rm -rf /tmp/$$.tmp

# cd to working dir
cd $WORKING_DIR
#WORKING_DIR=atdml/tasks

# load params
rid=$1
cmd=$2
arg01=$3
arg02=$4
logfile=$LOG_DIR/${rid}retrieve.log



# time stampe
date +"%m/%d/%Y %H:%M:%S $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)


# ? export for hdfs
export JAVA_HOME=$JAVA_HOME
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_USER_NAME=$HADOOP_USER_NAME

 
echo hdfs dfs $cmd $arg01 $arg02
echo hdfs dfs $cmd $arg01 $arg02 >> $logfile 2>&1
hdfs dfs $cmd $arg01 $arg02 >> $logfile 2>&1
ret=$?

# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "Warning!! $0 ret=${ret}"
    echo "Warning!! $0 ret=${ret}" >> $logfile  2>&1
    exit $ret
else
    echo " $0 done!  ret=${ret}"
    echo " $0 done!  ret=${ret}" >> $logfile  2>&1
fi

# time stampe
date +"%m/%d/%Y %H:%M:%S $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

