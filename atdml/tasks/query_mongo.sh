#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# load config file
sed '/=/!d;/\[\.*/d;/^ *#/d;' < app.config > /tmp/$$.tmp
source /tmp/$$.tmp

# cd to working dir
cd $WORKING_DIR
#WORKING_DIR=atdml/tasks

echo 'in query_mongo.sh'

# load params
rid=$1
filename=$2
uploadtype=$3
download_flag=$4

host=$5
port=$6
db=$7
tbl=$8
jstr_proj=$9
jstr_filter=${10}
usr=${11}
pwd=${12}
lb_field=${13}
lb_mapping=${14}
src_ds_id=${15}
src_fname=${16}
ret=-1

logfile=$LOG_DIR/${rid}retrieve.log
# zip file
from_file=$FEATURE_SRC_DIR/$filename
OUT_DIR=$RETRIEVE_DEC_DIR/${rid}retrieve
# parquet file dir
to_hdfs_dir=$RETRIEVE_DATA_DIR
#RETRIEVE_SCRIPT=query_mongo.py
spark_cmd=$SPARK_SUBMIT #/home/hadoop/spark-1.2.0-bin-hadoop2.4/bin/spark-submit

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo INFO: $uploadtype
#echo "jstr_proj="$jstr_proj
#echo "jstr_filter="$jstr_filter

if [[ $uploadtype == *"List IN-dynamic"* ]]
then
    if [ "$jstr_filter" == "" ]; then
        jstr_filter=$c_dynamic_filter
    fi
    if [ "$jstr_proj" == "" ]; then
        jstr_proj=$c_dynamic_proj
    fi
    sql=$c_dynamic_sql
elif [[ $uploadtype == *"List IN-static"* ]]
then
    if [ "$jstr_filter" == "" ]; then
        jstr_filter==$c_static_filter
    fi
    if [ "$jstr_proj" == "" ]; then
        jstr_proj=$c_static_proj
    fi
    sql=$c_static_sql
elif [[ $uploadtype == *"Query IN-dynamic"* ]]
then
    sql=$c_dynamic_sql
elif [[ $uploadtype == *"Query IN-static"* ]]
then
    sql=$c_static_sql
fi

echo "INFO: jstr_filter="$jstr_filter
echo "INFO: jstr_proj="$jstr_proj

# ? for hdfs
export JAVA_HOME=$JAVA_HOME
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_USER_NAME=$HADOOP_USER_NAME

if [ "$download_flag" == "Y" ];
then

    if [[ $uploadtype == *"Query IN"* ]]
    then 
        # query and output as zip file
        echo INFO: Invoke: python $RETRIEVE_SCRIPT -z $rid -o $OUT_DIR -m "query_outzip" -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "" -lb "$lb_field" -lm "$lb_mapping"
        echo INFO: Invoke: python $RETRIEVE_SCRIPT -z $rid -o $OUT_DIR -m "query_outzip" -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "" -lb "$lb_field" -lm "$lb_mapping" >> $logfile 2>&1
        
        python $RETRIEVE_SCRIPT -z $rid -o $OUT_DIR -m "query_outzip" -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "$pwd" -lb "$lb_field" -lm "$lb_mapping" >> $logfile 2>&1

        ret=$?
    else
        if [[ ! -z $src_ds_id ]]
        then # extract data from source dataset
            echo "INFO: Inherit data from dataset id=$src_ds_id"
            echo "INFO: Inherit data from dataset id=$src_ds_id" >> $logfile 2>&1
            echo INFO: Invoke Spark: $spark_cmd $EXTRACT_DS_SCRIPT -r $rid -shd $HADOOP_MASTER$to_hdfs_dir/$src_fname -ohd $HADOOP_MASTER$to_hdfs_dir/$filename -zfn $from_file -old $OUT_DIR -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "" 
            echo INFO: Invoke Spark: $spark_cmd $EXTRACT_DS_SCRIPT -r $rid -shd $HADOOP_MASTER$to_hdfs_dir/$src_fname -ohd $HADOOP_MASTER$to_hdfs_dir/$filename -zfn $from_file -old $OUT_DIR -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw ""  >> $logfile
            $spark_cmd $EXTRACT_DS_SCRIPT -r $rid -shd $HADOOP_MASTER$to_hdfs_dir/$src_fname -ohd $HADOOP_MASTER$to_hdfs_dir/$filename -zfn $from_file -old $OUT_DIR -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "$pwd" >> $logfile  2>&1
        else
            # get docs by hash list (_id) in a zip file
            echo INFO: Invoke: python $RETRIEVE_SCRIPT -z $from_file -o $OUT_DIR -m "hash_ziplist" -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "" 
            echo INFO: Invoke: python $RETRIEVE_SCRIPT -z $from_file -o $OUT_DIR -m "hash_ziplist" -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw ""  >> $logfile 2>&1
            python $RETRIEVE_SCRIPT -z $from_file -o $OUT_DIR -m "hash_ziplist" -jp "$jstr_proj" -jf "$jstr_filter" -ip "$host" -p $port -dn $db -t $tbl -un "$usr" -pw "$pwd"  >> $logfile 2>&1
        fi
        ret=$?
    fi
    
    if [ $ret -ne 0 ]; then
        echo "ERROR: $RETRIEVE_SCRIPT failed. Please check log for details ret=${ret}" 
        echo "ERROR: $RETRIEVE_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
    
    # create folder at HDFS ========================
    echo INFO: hdfs dfs -mkdir "$HADOOP_MASTER$to_hdfs_dir/$filename"
    echo INFO: hdfs dfs -mkdir "$HADOOP_MASTER$to_hdfs_dir/$filename" >> $logfile 2>&1
    hdfs dfs -mkdir $HADOOP_MASTER$to_hdfs_dir/$filename >> $logfile 2>&1
    # clean up *.gz at HDFS ========================
    echo INFO: clean up hdfs folder
    echo INFO: clean up hdfs folder >> $logfile 2>&1
    echo INFO: hdfs dfs -rm -r "$HADOOP_MASTER$to_hdfs_dir/$filename/*.gz"
    echo INFO: hdfs dfs -rm -r "$HADOOP_MASTER$to_hdfs_dir/$filename/*.gz" >> $logfile 2>&1
    hdfs dfs -rm $HADOOP_MASTER$to_hdfs_dir/$filename/*.gz >> $logfile 2>&1

    fcount=`ls $OUT_DIR/*.gz | wc -l`  
    #upload  *.gz to HDFS ========================
    if [ $fcount -gt 0 ]
    then
        echo INFO: upload to HDFS 
        echo INFO: hdfs dfs -put "$OUT_DIR/*.gz" $HADOOP_MASTER$to_hdfs_dir/$filename
        echo INFO: hdfs dfs -put "$OUT_DIR/*.gz" $HADOOP_MASTER$to_hdfs_dir/$filename >> $logfile 2>&1
        hdfs dfs -put $OUT_DIR/*.gz $HADOOP_MASTER$to_hdfs_dir/$filename >> $logfile 2>&1
        ret=$?
        #echo " *Upload to HDFS, $HADOOP_MASTER$to_hdfs_dir/$filename, done!! ret=${ret}"
        if [ $ret -ne 0 ]; then
            echo "ERROR: HDFS upload failed. Please check log for details ret=${ret}" 
            echo "ERROR: HDFS upload failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi
    else
        echo "WARNING: no .gz file in $OUT_DIR for HDFS upload"
        echo "WARNING: no .gz file in $OUT_DIR for HDFS upload"  >> $logfile 2>&1
    fi
    
    #remove folder out
    echo INFO: rm -rf $OUT_DIR
    echo INFO: rm -rf $OUT_DIR >> $logfile 2>&1
    rm -rf $OUT_DIR >> $logfile 2>&1
    
    # for creating parquet file  =======================================
    date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
    date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
    
else    
    # clean up parquet folder at HDFS ========================
    echo INFO: clean folder
    echo INFO: hdfs dfs -rm -r "$to_hdfs_dir/$filename/*.parquet"
    echo INFO: hdfs dfs -rm -r "$to_hdfs_dir/$filename/*.parquet" >> $logfile 2>&1
    hdfs dfs -rm -r $to_hdfs_dir/$filename/*.parquet >> $logfile 2>&1
fi


if [ $fcount -gt 0 ]
then
    # generate parquet file 
    echo INFO: Invoke Spark: $spark_cmd $RETRIEVE_PARQUET_CREATOR -hd $HADOOP_MASTER -d $to_hdfs_dir/$filename -sql "$sql" -r $rid 
    echo INFO: Invoke Spark: $spark_cmd $RETRIEVE_PARQUET_CREATOR -hd $HADOOP_MASTER -d $to_hdfs_dir/$filename -sql "$sql" -r $rid >> $logfile
    #sudo -u hadoop $spark_cmd $RETRIEVE_PARQUET_CREATOR -hd $HADOOP_MASTER -d $to_hdfs_dir/$filename -sql "$sql" -r $rid >> $logfile  2>&1
    $spark_cmd $RETRIEVE_PARQUET_CREATOR -hd $HADOOP_MASTER -d $to_hdfs_dir/$filename -sql "$sql" -r $rid >> $logfile  2>&1
    ret=$?
fi

# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "ERROR: $RETRIEVE_PARQUET_CREATOR failed !! ret=${ret}"
    echo "ERROR: $RETRIEVE_PARQUET_CREATOR failed !! ret=${ret}" >> $logfile  2>&1
    exit $ret
else
    echo "INFO: done query mongo !  ret=${ret}"
    echo "INFO: done query mongo !  ret=${ret}" >> $logfile  2>&1
fi

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

