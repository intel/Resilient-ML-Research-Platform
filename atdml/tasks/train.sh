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
uploadtype=$3
algtype=$4
opts=$5
ds_id=$6

spark_cmd=$SPARK_SUBMIT #/home/hadoop/spark-1.2.0-bin-hadoop2.4/bin/spark-submit


# set src here, need a flag in model to know starting from
hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${rid}

# if ds_id is set, it is an option
if [[ $ds_id ]]
then
    hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR/${ds_id}
fi


#out_dir=$TRAIN_DES_DIR/$filename  # media/result
out_dir=$TRAIN_DES_DIR/$rid
#from_file=$FEATURE_SRC_DIR/$filename
sparkurl=$SPARK_URL
logfile=$LOG_DIR/${rid}train.log
#TRAIN=single_run.py
train_alg=$TRAIN
hdfs_model_dir=$HADOOP_MASTER$HDFS_MODEL_DIR/$rid


if [ "$algtype" == "mllib" ]
then
    train_alg=$TRAIN_ML
elif [ "$algtype" == "mllib_cv" ]
then
    train_alg=$TRAIN_ML_CV
elif [ "$algtype" == "mllib_clustering" ]
then
    train_alg=$TRAIN_ML_CLUSTERING
elif [ "$algtype" == "scikit" ]
then
    train_alg=$TRAIN_SKL
elif [ "$algtype" == "scikit_cv" ]
then
    train_alg=$TRAIN_SKL_CV
elif [ "$algtype" == "scikit_clustering" ]
then
    train_alg=$TRAIN_SKL_CLUSTERING
fi

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
echo INFO: File type: $uploadtype
echo INFO: Invoke Spark: $spark_cmd $train_alg -f $hdfs_dir -n $filename -o $out_dir -r $rid  -sp $sparkurl -w $fromweb -pm "$opts" -sl "1" -dsid "$ds_id"
echo INFO: Invoke Spark: $spark_cmd $train_alg -f $hdfs_dir -n $filename -o $out_dir -r $rid  -sp $sparkurl -w $fromweb -pm "$opts" -sl "1" -dsid "$ds_id" >> $logfile

$spark_cmd $train_alg -f $hdfs_dir -n $filename -o $out_dir -r $rid  -sp $sparkurl -w $fromweb -pm "$opts" -sl "1" -dsid "$ds_id" >> $logfile  2>&1
ret=$?

# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "ERROR: $train_alg failed!! ret=${ret}"
    exit $ret
else
    # save a copy of model in web server for ensemble
    # tbd: check linear model here?
    echo curl -X GET "$API_MODEL_URL/$rid/" -u '\$APIWD' -o "$out_dir/$rid"_model.json
    curl -X GET "$API_MODEL_URL/$rid/" -u "$APIWD" -o "$out_dir/$rid"_model.json
    echo "INFO: Done! ret=${ret}"
fi

# model chg. clean up prediction data
if [ "$fromweb" == "1" ];
then
    echo INFO: Invoke python: $EXEC_SQLITE -s "$DEL_PREDICT_SQL$rid"
    echo INFO: Invoke python: $EXEC_SQLITE -s "$DEL_PREDICT_SQL$rid" >> $logfile  2>&1
    python $EXEC_SQLITE -s "$DEL_PREDICT_SQL$rid" >> $logfile  2>&1
fi

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

