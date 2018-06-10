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
n_gram=$4
pattern=$5
label_arr=$6
data_field_list=$7
feat_threshold=$8
feat_cust=$9
feat_cust_params=${10}
filter_ratio=${11}

from_file=$FEATURE_SRC_DIR/$filename
to_hdfs_dir=$HADOOP_MASTER$FEATURE_DES_DIR
spark_cmd=$SPARK_SUBMIT #/home/hadoop/spark-1.2.0-bin-hadoop2.4/bin/spark-submit
ret=-1
# this folder will be dropped later, so put it to /tmp
OUT_DIR=/tmp/${rid}feature

fromweb="0"
if [ $rid -gt 0 ]
then
    fromweb="1"
fi
logfile=$LOG_DIR/${rid}feature.log

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo File type: $uploadtype


# dispatch to different feature scripts by type ###########
if [ "$uploadtype" == "$FTYPE_ATD" ]
then
    #for ATD
    echo Invoke: python $FEATURE_ATD -z $from_file -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb
    echo Invoke: python $FEATURE_ATD -z $from_file -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb >> $logfile 2>&1
    python $FEATURE_ATD -z $from_file -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb >> $logfile 2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_ATD failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_ATD failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
elif [ "$uploadtype" == "$FTYPE_ANDROID_D" ]
then
    #for Android
    IN_DIR=$HADOOP_MASTER$RETRIEVE_DATA_DIR/$filename/
    echo Invoke Spark: $spark_cmd $FEATURE_ANDROID_DYNAMIC -d $IN_DIR -o $OUT_DIR -r $rid -w $fromweb 
    echo Invoke Spark: $spark_cmd $FEATURE_ANDROID_DYNAMIC -d $IN_DIR -o $OUT_DIR -r $rid -w $fromweb   >> $logfile 2>&1

    $spark_cmd $FEATURE_ANDROID_DYNAMIC -d $IN_DIR -o $OUT_DIR -r $rid -w $fromweb  >> $logfile  2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_ANDROID_DYNAMIC failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_ANDROID_DYNAMIC failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
elif [ "$uploadtype" == "$FTYPE_IN_S" ]
then
    #for IN Static raw data
    echo Invoke: python $FEATURE_IN_STATIC -z $from_file -o $OUT_DIR -r $rid
    echo Invoke: python $FEATURE_IN_STATIC -z $from_file -o $OUT_DIR -r $rid >> $logfile 2>&1
    python $FEATURE_IN_STATIC -z $from_file -o $OUT_DIR >> $logfile -r $rid 2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_IN_STATIC failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_IN_STATIC failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
elif [[ "$uploadtype" == *"$FTYPE_IN_S"* ]] 
then
    #for IN Static: MD5 list or Query, get data from HDFS
    IN_DIR=$HADOOP_MASTER$RETRIEVE_DATA_DIR/$filename/
    echo Invoke Spark: $spark_cmd $FEATURE_IN_STATIC -d $IN_DIR -o $OUT_DIR -r $rid -w $fromweb 
    echo Invoke Spark: $spark_cmd $FEATURE_IN_STATIC -d $IN_DIR -o $OUT_DIR -r $rid -w $fromweb   >> $logfile 2>&1

    $spark_cmd $FEATURE_IN_STATIC -d $IN_DIR -o $OUT_DIR -r $rid -w $fromweb  >> $logfile  2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_IN_STATIC failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_IN_STATIC failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
elif [[ "$uploadtype" == *"$FTYPE_IN_D"* ]]
then
    #for IN Dynamic: MD5 list or Query, get data from HDFS
    IN_DIR=$HADOOP_MASTER$RETRIEVE_DATA_DIR/$filename/
    echo Invoke Spark: $spark_cmd $FEATURE_IN_DYNAMIC -d $IN_DIR -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb -ft $feat_threshold
    echo Invoke Spark: $spark_cmd $FEATURE_IN_DYNAMIC -d $IN_DIR -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb -ft $feat_threshold >> $logfile 2>&1

    $spark_cmd $FEATURE_IN_DYNAMIC -d $IN_DIR -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb -ft $feat_threshold >> $logfile  2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: Featuring failed. Please check log for details ret=${ret}" 
        echo "ERROR: Featuring failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
elif [[ "$uploadtype" == *"pattern"* ]] 
then
    #N-gram pattern text, get data from HDFS
    # filename may be a list
    IN_DIR=$filename
    idx=`expr index "$filename" '[,]'`
    # single folder found, not a list
    if [ $idx -eq 0 ]
    then
        IN_DIR=$RETRIEVE_DATA_DIR/$filename/
    fi
    
    # clean up pca data at /result/
    PCA_DIR=$TRAIN_DES_DIR
    OUT_DIR=""
    echo Invoke Spark: $spark_cmd $FEATURE_NG_PATTERN -d "$IN_DIR" -o $PCA_DIR -r $rid -ng $n_gram -w $fromweb -ptn "$pattern" -lba "$label_arr" -ft "$feat_threshold" -cf "$feat_cust" -cfp "$feat_cust_params" -fr "$filter_ratio"
    echo Invoke Spark: $spark_cmd $FEATURE_NG_PATTERN -d "$IN_DIR" -o $PCA_DIR -r $rid -ng $n_gram -w $fromweb -ptn "$pattern" -lba "$label_arr" -ft "$feat_threshold" -cf "$feat_cust" -cfp "$feat_cust_params" -fr "$filter_ratio" >> $logfile 2>&1

    $spark_cmd $FEATURE_NG_PATTERN -d "$IN_DIR" -o $PCA_DIR -r $rid -ng $n_gram -w $fromweb -ptn "$pattern" -lba "$label_arr" -ft "$feat_threshold" -cf "$feat_cust" -cfp "$feat_cust_params" -fr "$filter_ratio" >> $logfile  2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_NG_PATTERN failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_NG_PATTERN failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi

elif [[ "$uploadtype" == "Custom"* ]] 
then
    #N-gram pattern text, TBD
    # filename may be a list
    IN_DIR=$filename
    idx=`expr index "$filename" '[,]'`
    # single folder found, not a list
    if [ $idx -eq 0 ]
    then
        IN_DIR=$RETRIEVE_DATA_DIR/$filename/
    fi
    
    # clean up pca data at /result/
    PCA_DIR=$TRAIN_DES_DIR
    OUT_DIR=""
    echo Invoke Spark: $spark_cmd $FEATURE_NG_PATTERN -d "$IN_DIR" -o $PCA_DIR -r $rid -ng $n_gram -w $fromweb -ptn "$pattern" -lba "$label_arr" -ft "$feat_threshold" -cf "$feat_cust" -cfp "$feat_cust_params" -fr "$filter_ratio"
    echo Invoke Spark: $spark_cmd $FEATURE_NG_PATTERN -d "$IN_DIR" -o $PCA_DIR -r $rid -ng $n_gram -w $fromweb -ptn "$pattern" -lba "$label_arr" -ft "$feat_threshold" -cf "$feat_cust" -cfp "$feat_cust_params" -fr "$filter_ratio" >> $logfile 2>&1

    $spark_cmd $FEATURE_NG_PATTERN -d "$IN_DIR" -o $PCA_DIR -r $rid -ng $n_gram -w $fromweb -ptn "$pattern" -lba "$label_arr" -ft "$feat_threshold" -cf "$feat_cust" -cfp "$feat_cust_params" -fr "$filter_ratio" >> $logfile  2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_NG_PATTERN failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_NG_PATTERN failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
elif [[ "$uploadtype" == *"JSON"* ]] 
then
    #for N-gram JSON format, get data from HDFS
    IN_DIR=$HADOOP_MASTER$RETRIEVE_DATA_DIR/$filename/
    echo Invoke Spark: $spark_cmd $FEATURE_NG_PATTERN -d $IN_DIR -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb -dfl "$data_field_list" -lba "$label_arr"
    echo Invoke Spark: $spark_cmd $FEATURE_NG_PATTERN -d $IN_DIR -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb -dfl "$data_field_list" -lba "$label_arr" >> $logfile 2>&1

    $spark_cmd $FEATURE_NG_PATTERN -d $IN_DIR -o $OUT_DIR -r $rid -ng $n_gram -w $fromweb -dfl "$data_field_list" -lba "$label_arr" >> $logfile  2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $FEATURE_NG_PATTERN failed. Please check log for details ret=${ret}" 
        echo "ERROR: $FEATURE_NG_PATTERN failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
fi


# check return code and echo message
if [ "$ret" -ne 0 ]
then
    echo "ERROR: $0 failed!! ret=${ret}"
    exit $ret
else
    echo "INFO: spark job done!  ret=${ret}"
fi

# if $OUT_DIR exists, upload to hdfs
if [ -d "$OUT_DIR" ]; then
    # ? export for hdfs
    export JAVA_HOME=$JAVA_HOME
    export HADOOP_HOME=$HADOOP_HOME
    export HADOOP_USER_NAME=$HADOOP_USER_NAME

    #upload feature extraction data to HDFS ========================
    echo INFO: clean up folder $to_hdfs_dir/$rid
    #echo INFO: hdfs dfs -rm -r $to_hdfs_dir/$rid >> $logfile 2>&1
    hdfs dfs -rm -r $to_hdfs_dir/$rid >> /dev/null 2>&1

    echo INFO: upload to HDFS 
    echo INFO: hdfs dfs -put $OUT_DIR $to_hdfs_dir/$rid
    #echo INFO: hdfs dfs -put $OUT_DIR $to_hdfs_dir/$rid >> $logfile 2>&1
    hdfs dfs -put $OUT_DIR $to_hdfs_dir/$rid >> $logfile 2>&1
    ret=$?
fi

# check return code and echo message
if [ "$ret" -ne 0 ]
then
    echo "ERROR: Featuring failed!! ret=${ret}" 
    echo "ERROR: Featuring failed!! ret=${ret}" >> $logfile 2>&1
    exit $ret
else
    echo "INFO: Featuring done!  ret=${ret}"  
    echo "INFO: Featuring done!  ret=${ret}"  >> $logfile 2>&1
fi


#remove folder out
echo INFO: rm -rf $OUT_DIR
#echo INFO: rm -rf $OUT_DIR >> $logfile 2>&1
rm -rf $OUT_DIR >> $logfile 2>&1

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit $ret

