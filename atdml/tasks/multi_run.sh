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
filename=$2
mrun_numb=$3
uploadtype=$4
ml_lib=$5
opts=$6
ds_id=$7

spark_cmd=$SPARK_SUBMIT 
hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${rid}
# if ds_id is set, it is an option
if [[ $ds_id ]]
then
    hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${ds_id}
fi

out_dir=$TRAIN_DES_DIR/${rid}
sparkurl=$SPARK_URL
#MULTI_RUN=multi_run.py

logfile=$LOG_DIR/${rid}multi_run.log

fromweb="0"
if [ $rid -gt 0 ]
then
    fromweb="1"
fi

#WORKING_DIR=atdml/tasks

echo "============================ in multi_run.sh " 
# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
if [ "$ml_lib" == "mllib" ]
then
    train_alg=$MULTI_RUN_ML
else
    train_alg=$MULTI_RUN_SKL
fi
ret=-1
echo Invoked Spark: $spark_cmd $train_alg -f $hdfs_dir -n $filename -o $out_dir -r $rid -mn $mrun_numb -sp $sparkurl -u $uploadtype -w $fromweb -pm "$opts"
echo Invoked Spark: $spark_cmd $train_alg -f $hdfs_dir -n $filename -o $out_dir -r $rid -mn $mrun_numb -sp $sparkurl -u "$uploadtype" -w $fromweb -pm "$opts" >> $logfile

$spark_cmd $train_alg -f $hdfs_dir -n $filename -o $out_dir -r $rid -mn $mrun_numb -sp $sparkurl -u "$uploadtype" -w $fromweb -pm "$opts"  >> $logfile 2>&1
ret=$?



# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "Error executing $TMRUN_PYTHON_SCRIPT -f '${hdfs_dir}' !! ret=${ret}"
    exit $ret
else
    echo " *mrun script done!  ret=${ret}"
fi

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

