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
pca_k=$2
opts=$3   #TBD need to add for sklearn pca??
ds_id=$4
refresh=$5

spark_cmd=$SPARK_SUBMIT #/home/hadoop/spark-1.2.0-bin-hadoop2.4/bin/spark-submit


# set src here, need a flag in model to know starting from
hdfs_dir=$FEATURE_DES_DIR/${rid}
# web local output dir 
out_dir=$TRAIN_DES_DIR

# if ds_id is set, get hdfs file from parent dataset
if [[ $ds_id ]]
then
    hdfs_dir=$FEATURE_DES_DIR/${ds_id}
    #out_dir=$TRAIN_DES_DIR/${ds_id}
fi
# hdfs out data for mllib
pcaed_filename=$hdfs_dir/"libsvm_data_pca_"${pca_k}
# ml has extra extension
if [[ "$opts" == *"mllib"* ]]; then
    pcaed_filename=$pcaed_filename.ml
fi
libsvm_filename=$hdfs_dir/"libsvm_data"

sparkurl=$SPARK_URL
logfile=$LOG_DIR/${rid}pca.log
fromweb="0"

if [ $rid -gt 0 ]
then
    fromweb="1"
fi
ret=-1

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)


# if $out_dir exists, clean up pca file here
# ? export for hdfs
export JAVA_HOME=$JAVA_HOME
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_USER_NAME=$HADOOP_USER_NAME

#check if pca file exists, exit when refresh flag is off ========================
if [ "$refresh" == "0" ];
then
    echo INFO: Check if file "$pcaed_filename" exists
    echo INFO: hdfs dfs -ls $pcaed_filename 
    echo INFO: hdfs dfs -ls $pcaed_filename >> $logfile  2>&1
    out_list=`hdfs dfs -ls $pcaed_filename`
    # if not empty return, return
    if [[ ! -z  $out_list  ]]
    then
        echo "INFO: PCA file="$pcaed_filename" exists!! No action taken."
        echo "INFO: PCA file="$pcaed_filename" exists!! No action taken." >> $logfile 2>&1
        exit 0
    fi
fi

#upload feature extraction data to HDFS ========================
echo INFO: clean folder
echo INFO: hdfs dfs -rm -r $pcaed_filename
echo INFO: hdfs dfs -rm -r $pcaed_filename >> $logfile 2>&1
hdfs dfs -rm -r $pcaed_filename >> $logfile 2>&1
ret=$?
if [ $ret -ne 0 ]
then
    echo "WARNING: hdfs dfs -rm failed!! ret=${ret}"
fi

ret=-1

if [[ "$opts" == *"mllib"* ]]; then
    # by Spark ML
    pca_alg=$PCA_ML
    # write pca output to local before to hdfs
    echo "INFO: PCAed hdfs file by threshold at " $out_dir  
    echo "INFO: PCAed hdfs file by threshold at " $out_dir  >> $logfile
    echo INFO: Invoke Spark: $spark_cmd $pca_alg -f $hdfs_dir -o $out_dir -r $rid -sp $sparkurl -w $fromweb -pm "$opts" -dsid "$ds_id" 
    echo INFO: Invoke Spark: $spark_cmd $pca_alg -f $hdfs_dir -o $out_dir -r $rid -sp $sparkurl -w $fromweb -pm "$opts" -dsid "$ds_id" >> $logfile
    $spark_cmd $pca_alg -f $hdfs_dir -o $out_dir -r $rid -sp $sparkurl -w $fromweb -pm "$opts" -dsid "$ds_id" >> $logfile  2>&1
    ret=$?

else
    # sklearn
    pca_alg=$PCA_SKL
    # write pca output to local before to hdfs
    echo "INFO: PCAed hdfs file by threshold at " $out_dir  
    echo "INFO: PCAed hdfs file by threshold at " $out_dir  >> $logfile
    echo INFO: Invoke Spark: $spark_cmd $pca_alg -f $hdfs_dir -o $out_dir -r $rid -sp $sparkurl -w $fromweb -pm "$opts" -dsid "$ds_id" 
    echo INFO: Invoke Spark: $spark_cmd $pca_alg -f $hdfs_dir -o $out_dir -r $rid -sp $sparkurl -w $fromweb -pm "$opts" -dsid "$ds_id" >> $logfile
    $spark_cmd $pca_alg -f $hdfs_dir -o $out_dir -r $rid -sp $sparkurl -w $fromweb -pm "$opts" -dsid "$ds_id" >> $logfile  2>&1
    ret=$?
fi
# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "ERROR: PCA failed!! ret=${ret}" 
    echo "ERROR: PCA failed!! ret=${ret}" >> $logfile
    exit $ret
else
    echo "INFO: PCA done!! ret=${ret}" 
    echo "INFO: PCA done!! ret=${ret}" >> $logfile
fi


# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

