#!/bin/bash
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# load params
source ./submit_apk.config

#web_host="xxx.com:8xxx"
#pred_dir="atdml/api/pred"
#ds_id=$1
fname=$1
df_dir="atdml/api/df"

#uid=$3
#passwd=$4
#total_wait_time=600
#sleep_interval=5
#wait_cycle=$[$total_wait_time/$sleep_interval]

#echo "ds_id="$ds_id
# check parameter count
if [ $# -lt 1 ]
then
    echo "APK filename is required !"
    exit -1
fi
# check if file exist
if [ ! -f $fname ]
then
    echo "File '"$fname"' not found!"
    exit -1
fi
econfig64=""
# check if $emulater_config defined and check if file exist
if [ ! -z $emulater_config_fname ]
then
    if [ -f $emulater_config_fname ]
    then
        econfig64=`base64 $emulater_config_fname`
        #echo $econfig64
    fi
fi

# submit APK =======================
#echo curl -X POST http://$web_host/$pred_dir/$ds_id/exec/ -u $uid:$passwd -H "Content-Type: multipart/form-data" -F "docfile=@$fname" --form-string "_file_type=apk&_emulater_config=$econfig64"
sub_json=`curl -s -S -X POST http://$web_host/$pred_dir/$ds_id/exec/ -u $uid:$passwd -H "Content-Type: multipart/form-data" -F "docfile=@$fname" --form-string "_file_type=apk" --form-string "_emulater_config=$econfig64"`
status=`echo $sub_json | grep -Po '"status":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`
# check if submission status
if [ ! "$status" == "apk_queued" ]
then
    echo "Error occured. Submit_status="$sub_json
    exit -2
#else
#    echo Submit_status=$sub_json
fi
# get prediction id =======================
#echo echo $sub_json | grep -Po '"id":.*?[^\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'
pid=`echo $sub_json | grep -Po '"id":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`

# polling apk execution =======================
echo Prediction id=$pid. Polling job status:
counter=0
# poll status every $sleep_interval seconds =======================
while [ $counter -lt $wait_cycle ]
do
    echo -n .
    # get status
    #echo curl -X GET  http://$web_host/$pred_dir/$ds_id/$pid/ -u $uid:$passwd
    sts_json=`curl -s -S -X  GET  http://$web_host/$pred_dir/$ds_id/$pid/ -u $uid:$passwd`
    #echo status_return=$sts_json
    #echo echo $sts_json | grep -Po '"status":.*?[^\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'
    status=`echo $sts_json | grep -Po '"status":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`
    counter=$[$counter+1]
    if [[ "$status" == "predicted" ]]
    then
        prediction=`echo $sts_json | grep -Po '"prediction":.*?[^\\\\]"' | awk '{split($0,a,":");print a[2]}' | tr -d '"'`
        echo Status=$status, Prediction=$prediction. id=$pid
        counter=$wait_cycle
    elif [[ "$status" == "failed" ]]
    then
        echo "Error!! Execution or prediction failed. id="$pid
        counter=$wait_cycle
    elif [ $counter -eq $wait_cycle ]
    then
        echo " Polling expired. Please visit ML Prediction page to check status for dataset id="$pid
    else
        sleep $sleep_interval
    fi
done

# retrieve execution log
if [[ "$status" == "predicted" ]]
then
    echo "Download execution log to '"$log_output_dir/$pid$log_suffix"'"
    sleep $sleep_interval
    mkdir -p $log_output_dir
    curl -s -S -X  GET  http://$web_host/$log_dir/$ds_id/_e_$pid/-1/ -u $uid:$passwd -o $log_output_dir/$pid$log_suffix

    # check if flag of zip is available
    has_zip=$(head -n 1 $emulater_config_fname)
    if [[ "$has_zip" == "SAVE_RUNTIME_FILES=1" ]]
    then
        echo "Download output to '"$log_output_dir/$pid"_output.zip'"
        curl -s -S -X GET http://$web_host/$df_dir/$pid/ -u $uid:$passwd -o $log_output_dir/$pid"_output.zip"
    fi
    
 fi
echo ""
