#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0

BASE_DIR=/home/django/myml
# load config file
sed '/=/!d;/\[\.*/d;/^ *#/d;' < $BASE_DIR/app.config > /tmp/$$.tmp
source /tmp/$$.tmp
rm -rf /tmp/$$.tmp

# cd to working dir
cd $BASE_DIR/$WORKING_DIR
#WORKING_DIR=atdml/tasks
logfile=$LOG_DIR/android_prediction.log


# time stampe
date +"%m/%d/%Y %H:%M:%S $0 at $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $0 at $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo "logfile="$logfile
# ? export for hdfs
export JAVA_HOME=$JAVA_HOME
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_USER_NAME=$HADOOP_USER_NAME
nfs_rlt_dir=$APK_SANDBOX_OUT_DIR
nfs_arch_dir=$APK_SANDBOX_ARC_DIR
script_dir=$BASE_DIR/media/result
target_output="*.only.log.list"
status_output="status"
db_file="db.sqlite3"
sql_pred="select id, train_id from atdml_document where status='apk_queued'"
sql_apk="select id, train_id from atdml_document where status='submitted'"

while true
do
	#echo "$0..."
    # submit ML prediction only for status=='apk_queued'
    out=`sqlite3 $BASE_DIR/$db_file "$sql_pred" `
    for i in ${out// /} # separated by space
    do
        # get cid & dsid 
        farray=(${i//\|/ }) # separated by |
        cid="${farray[0]}"
        dsid="${farray[1]}"
        ret=-1
        # check status file
        sts_cnt=`ls $nfs_rlt_dir/$cid/$status_output 2> /dev/null | wc -l`
        # check log file ( will be rename to .failed if prediction failed )
        cnt=`ls $nfs_rlt_dir/$cid/$target_output 2> /dev/null | wc -l`
        if [ $cnt -eq 1 ] && [ $sts_cnt -eq 1 ]
        then
            echo "$0 $dsid, pred $cid"
            echo "$0 $dsid, pred $cid" >> $logfile 2>&1
            
            fnames=`ls $nfs_rlt_dir/$cid/*.only.log.list`
            fnames=(${fnames// /})
            fname=${fnames[0]}   
            echo "Invoke $script_dir/$dsid/$cid.sh $fname......"
            echo "Invoke $script_dir/$dsid/$cid.sh $fname......"  >> $logfile 2>&1
            
            # stdin, stderror to record's log file
            $script_dir/$dsid/$cid.sh $fname
            ret=$?
            if [ $ret -ne 0 ]
            then
                echo "Error!! $cid.sh failed!! ret=${ret}"
                echo "Error!! $cid.sh failed!! ret=${ret}" >> $logfile 2>&1
                # rename file
                mv $fname $fname.failed
                # update sqlite db?
                sql="update atdml_document set status='failed' where id="$cid
                echo Update sqlite db: $sql
                echo Update sqlite db: $sql >> $logfile  2>&1
                sqlite3 $BASE_DIR/db.sqlite3 "$sql" >> $logfile  2>&1

            else # success
                echo "$cid.sh done! ret=${ret}"
                echo "$cid.sh done! ret=${ret}" >> $logfile 2>&1
                #rm -f $script_dir/$dsid/$cid.sh
                # move script to result/
                mv $script_dir/$dsid/$cid.sh $nfs_rlt_dir/$cid/
                # move output to archive
                echo "mv $nfs_rlt_dir/$cid $nfs_arch_dir/"
                echo "mv $nfs_rlt_dir/$cid $nfs_arch_dir/" >> $logfile 2>&1
                mv $nfs_rlt_dir/$cid $nfs_arch_dir/
            fi

        fi  
        echo -n ":"
    done  # End prediction
    
    # update db  for status=='submitted' ====================================
    out=`sqlite3 $BASE_DIR/$db_file "$sql_apk" `
    for i in ${out// /} # separated by space
    do
        # get cid & dsid 
        farray=(${i//\|/ }) # separated by |
        cid="${farray[0]}"
        dsid="${farray[1]}"
        ret=-1
        # check status file
        sts_cnt=`ls $nfs_rlt_dir/$cid/$status_output 2> /dev/null | wc -l`
        # check log file ( will be rename to .failed if prediction failed )
        cnt=`ls $nfs_rlt_dir/$cid/$target_output 2> /dev/null | wc -l`
        if [ $cnt -eq 1 ] && [ $sts_cnt -eq 1 ]
        then
            sts_content=`cat $nfs_rlt_dir/$cid/$status_output`
            echo "Update status for cid=" $cid ", status=" $sts_content
            echo "Update status for cid=" $cid ", status=" $sts_content >> $logfile 2>&1
            upt_sql="update atdml_document set status='$sts_content', processed_date=datetime('now') where id=$cid"
            echo $upt_sql
            sqlite3 $BASE_DIR/db.sqlite3 "$upt_sql" >> $logfile  2>&1
        fi  
        echo -n "."
    done  # End status update
	sleep 7
done



ret=$?


#remove folder out
#rm -rf $OUT_DIR
# time stampe
date +"%m/%d/%Y %H:%M:%S $0 at $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $0 at $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

