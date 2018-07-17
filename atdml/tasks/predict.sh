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
upload_filename=$1
rid=$2
filename=$3
cid=$4
feature_numb=$5
action_type=$6
tlabel=$7
uploadtype=$8


host=$9
#port=${10}
if [ -z "${10}" ]; then
    port=$port
else
    port=${10}
fi
db=${11}
tbl=${12}
hash=${13}
if [ -z "$host" ]; then
    host=$ip_address
fi
if [ -z "$db" ]; then
    db=$db_name
fi
if [ -z "$tbl"  ]; then
    tbl=$tb_name
fi
#n-gram
n_gram=${14}
# param string
opt_str=${15}
# spark mllib or scikit
ml_lib=${16}
usr=${17}
pwd=${18}
jstr_proj=${19}
ds_id=${20}
pattern=${21}
verbose=${22}
pca_opts=${23}
exe_type=${24}
# this is a base64-ed string
emulater_config=${25}
feat_threshold=${26}
ds_list=${27}
pert_flag=${28}
feat_cust=${29}
feat_cust_params=${30}


if [ -z "$usr"  ]; then
    usr=$username
fi
if [ -z "$pwd"  ]; then
    pwd=$password
fi
out_dir=$TRAIN_DES_DIR/$rid
src_dir=$FEATURE_SRC_DIR
logfile=$LOG_DIR/${cid}predict.log
spark_cmd=$SPARK_SUBMIT 

RETR_DEC_DIR=$RETRIEVE_DEC_DIR/${cid}retrieve
showlab="1"
tgt_filename=""

fromweb="0"
if [ $rid -gt 0 ]
then
    fromweb="1"
fi

# gzip the upload file (to save space)
function chk_gz {
    fname=$1
    # make sure file was gzip-ed by grep key word and then count it
    gz_chk=`file $fname | grep "gzip compressed data" | wc -l`
    # if 0; gzip it
    if [ "$gz_chk" -eq "0" ];
    then
        echo "Compress file..."
        # remove old file
        if [ -f $fname.gz ]; then
            rm $fname.gz
        fi
        gzip $fname
        tgt_filename=$fname.gz
        return 0
    fi
    return 1
}

# get script by $uploadtype and output to $UF_SINGLE_PREDICT_SCRIPT
function get_script_name {
    utype="$1"
    echo utype=$utype
    # set program for IN dynamic 
    UF_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_IN_DYNAMIC 
    # if static:
    if [[ $utype == *"-static"*  ]]; then
        UF_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_IN_STATIC
    elif [[ $utype == *"Format"*  ]]; then
        UF_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_GENERIC
    elif [[ $utype == *"pattern"*  ]]; then
        UF_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_PATTERN
    elif [[ $utype == *"ensemble"*  ]]; then
        UF_SINGLE_PREDICT_SCRIPT=$PREDICT_ENSEMBLE_PATTERN
    elif [[ $utype == *"image"*  ]]; then
        UF_SINGLE_PREDICT_SCRIPT=$PREDICT_IMAGE
    elif [[ $utype == *"Custom"*  ]]; then
        UF_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_CUSTOM 
    fi
    echo UF_SINGLE_PREDICT_SCRIPT=$UF_SINGLE_PREDICT_SCRIPT
    
    return 0
}

echo "INFO: ============================ in predict.sh " 
# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)
echo "INFO: uploadtype="$uploadtype ",act type="$action_type",ml_lib="$ml_lib
echo "INFO: uploadtype="$uploadtype ",act type="$action_type",ml_lib="$ml_lib >> $logfile 2>&1

if [[ $uploadtype == *"in-dynamic"* ]]
then
    jstr_proj=$c_dynamic_proj
    sql=$c_dynamic_sql
elif [[ $uploadtype == *"in-static"* ]]
then
    jstr_proj=$c_static_proj
    sql=$c_static_sql
fi
ret=-1

# upload_predict ############################
if [ "$action_type" == "upload_predict" ]; 
then
    echo "INFO: In upload prediction:" $upload_filename
    echo "INFO: In upload prediction:" $upload_filename >> $logfile 2>&1

    # TBD static not implemented. format md2.label; label is for internal usage only
    if [[ $exe_type == "apk-dynamic"  ]]; then 
        tgt_filename=$src_dir/$upload_filename
        # generate hash
        hash_fname=`sha256sum $tgt_filename  | awk -F' ' '{print $1}'`
        # generate config file from base64 encoded string to NFS
        #echo emulater_config=$emulater_config
        if [ ! -z "$emulater_config" ] 
        then
            echo "Invoke: echo 'emulater_config' | base64 -d > $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk.config"
            echo "Invoke: echo 'emulater_config' | base64 -d > $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk.config" >> $logfile 2>&1
            echo "$emulater_config" | base64 -d > $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk.config
        fi
        # move to NFS
        echo Invoke: mv $tgt_filename $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk
        echo Invoke: mv $tgt_filename $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk >> $logfile 2>&1
        mv $tgt_filename $APK_SANDBOX_IN_DIR/$cid-$hash_fname.apk
        ret=$?
        #  filename from android sandbox output
        out_filename=$APK_SANDBOX_OUT_DIR/$cid/$hash_fname.only.log.list
        # create a script file for prediction after execution
        get_script_name "$uploadtype"
        script_fname=$out_dir/$cid.sh
        cat ./template_prediction.sh > $script_fname
        chmod 750 $script_fname
        # script for prediction
        if [ "$ml_lib" == "mllib" ]; then
            echo Invoke: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d '\$fname' -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose -ft $feat_threshold ">> $logfile 2>&1"
            echo Invoke: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d '\$fname' -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose -ft $feat_threshold ">> $logfile 2>&1" >> $logfile 2>&1
            echo "$spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d \$fname -r $rid -o $out_dir -i '$cid' -fw '$fromweb'  -nb '$n_gram' -lb '$ml_lib' -sl '$showlab' -pm '$opt_str' -dsid '$ds_id' -ptn '$pattern' -pp '$pca_opts' -vb '$verbose' -ft '$feat_threshold' >> $logfile 2>&1" >> $script_fname
        else
            echo Invoke: $UF_SINGLE_PREDICT_SCRIPT -d '\$fname' -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose  -ft $feat_threshold ">> $logfile 2>&1"
            echo Invoke: $UF_SINGLE_PREDICT_SCRIPT -d '\$fname' -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose  -ft $feat_threshold ">> $logfile 2>&1" >> $logfile 2>&1
            echo "python $UF_SINGLE_PREDICT_SCRIPT -d \$fname -r $rid -o $out_dir -i '$cid' -fw '$fromweb' -nb '$n_gram' -lb '$ml_lib' -sl '$showlab' -pm '$opt_str' -dsid '$ds_id' -ptn '$pattern' -pp '$pca_opts' -vb '$verbose' -ft '$feat_threshold' >> $logfile 2>&1" >> $script_fname
        fi
        # save cmd to sqlite3
        
        if [ $ret -ne 0 ]
        then
            echo "Error!! $0 failed!! ret=${ret}"
            exit $ret
        else
            echo "$0 done! ret=${ret}"
            exit 205
        fi
    fi   # end apk dynamic
    
    if [[ $exe_type == "apk-static"  ]]; then 
        src_filename=$src_dir/$upload_filename
        # generate hash
        #hash_fname=`sha256sum $tgt_filename  | awk -F' ' '{print $1}'`
        
        # generate static data
        echo Invoke: $APK_STATIC_SW "$src_filename" "$out_dir"
        echo Invoke: $APK_STATIC_SW "$src_filename" "$out_dir"  >> $logfile 2>&1
        $APK_STATIC_SW $src_filename $out_dir
        tgt_filename=$out_dir/$upload_filename.only.log.static
        # replace .apk. from tgt_filename
        tgt_filename="${tgt_filename/.apk./.}"

        # move to $src_dir
        echo "INFO: mv -f $src_filename  $out_dir/$upload_filename"
        echo "INFO: mv -f $src_filename  $out_dir/$upload_filename" >> $logfile 2>&1
        mv -f $src_filename $out_dir/$upload_filename
        # zip it
        chk_gz $tgt_filename
        # get python program name
        get_script_name "$uploadtype" 
        
        # script for prediction
        if [ "$ml_lib" == "mllib" ]; then
            echo Invoke: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d "$tgt_filename" -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose -ft $feat_threshold ">> $logfile 2>&1"
            echo Invoke: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d "$tgt_filename" -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose -ft $feat_threshold ">> $logfile 2>&1" >> $logfile 2>&1
            $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i "$cid" -fw "$fromweb" -nb "$n_gram" -lb "$ml_lib" -sl "$showlab" -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb "$verbose" -ft "$feat_threshold" >> $logfile 2>&1 
        else
            echo Invoke: $UF_SINGLE_PREDICT_SCRIPT -d "$tgt_filename" -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose  -ft $feat_threshold ">> $logfile 2>&1"
            echo Invoke: $UF_SINGLE_PREDICT_SCRIPT -d "$tgt_filename" -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "'$opt_str'" -dsid "$ds_id" -ptn "$pattern" -pp "'$pca_opts'" -vb $verbose  -ft $feat_threshold ">> $logfile 2>&1" >> $logfile 2>&1
            python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i "$cid" -fw "$fromweb" -nb "$n_gram" -lb "$ml_lib" -sl "$showlab" -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb "$verbose" -ft "$feat_threshold" >> $logfile 2>&1 
                                                                              
        fi
        ret=$?
        
        if [ $ret -ne 0 ]
        then
            echo "Error!! $0 failed!! ret=${ret}"
            exit $ret
        else
            echo "$0 done! ret=${ret}"
            exit 205
        fi
    fi # end apk static

    if [[ $uploadtype == *"in-"* || $uploadtype == *"Format"*  || $uploadtype == *"pattern"*  ]]; then
        tgt_filename=$out_dir/$upload_filename
        # move file over
        echo "Invoke: mv -f $src_dir/$upload_filename $tgt_filename"
        mv -f $src_dir/$upload_filename $tgt_filename
        #echo "In upload_predict, tgt_filename=" $tgt_filename
        
        # make sure file was gzip-ed by grep key word and then count it
        chk_gz $tgt_filename

        get_script_name "$uploadtype"

        echo "INFO: In upload_predict, tgt_filename=" $tgt_filename
        # invoke predict 
        if [ "$ml_lib" == "mllib" ]; then
            echo Invoke: Spark $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold
            echo Invoke: Spark $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
            $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb  -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
        else
            echo Invoke: python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold
            echo Invoke: python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
            python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
        fi
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "ERROR!! $UF_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" 
            echo "ERROR!! $UF_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi
    elif [[ $uploadtype == *"Custom"* ]]; then
        tgt_filename=$out_dir/$upload_filename
        #tgt_filename=$UPLOAD_FULL_DIR/$upload_filename
        src_dir=
        echo "INFO: "$uploadtype":upload_filename="$upload_filename
        echo "INFO: "$uploadtype":upload_filename="$upload_filename  >> $logfile 2>&1

        
        # move file over
        echo "Invoke: mv -f $UPLOAD_FULL_DIR/$upload_filename $tgt_filename"
        mv -f "$UPLOAD_FULL_DIR/$upload_filename" "$tgt_filename"

        chk_gz "$tgt_filename"
        
        get_script_name "$uploadtype"

        echo "INFO: In Custom lib="$ml_lib", tgt_filename="$tgt_filename
        # invoke predict 
        if [ "$ml_lib" == "mllib" ]; then
            echo Invoke Spark: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params"
            echo Invoke Spark: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params" >> $logfile 2>&1
            $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb  -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params" >> $logfile 2>&1
       elif [ "$ml_lib" == "dnn" ]; then
            echo Invoke dnn: python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params"
            echo Invoke dnn: python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params" >> $logfile 2>&1
            python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params" >> $logfile 2>&1
       else
            echo Invoke: python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params"
            echo Invoke: python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params" >> $logfile 2>&1
            python $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" -ptn "$pattern" -pp "$pca_opts" -vb $verbose -ft $feat_threshold -cf "$feat_cust" -cfp "$feat_cust_params" >> $logfile 2>&1
        fi
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "ERROR!! $UF_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" 
            echo "ERROR!! $UF_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi
    elif [[ $uploadtype == "ATD"  ]]; then
        tgt_filename=$UPLOAD_FULL_DIR/$upload_filename

        # make sure file was gzip-ed by grep key word and then count it
        chk_gz $tgt_filename

        echo "INFO: upload_filename="$tgt_filename
        echo "INFO: upload_filename="$tgt_filename  >> $logfile 2>&1

        echo Invoke: Spark $spark_cmd $PREDICT_SINGLE_FILE_ATD -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id"
        echo Invoke: Spark $spark_cmd $PREDICT_SINGLE_FILE_ATD -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id" >> $logfile 2>&1
        $spark_cmd  $PREDICT_SINGLE_FILE_ATD -d $tgt_filename -r $rid -o $out_dir  -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -sl $showlab -pm "$opt_str" -dsid "$ds_id">> $logfile 2>&1
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "ERROR: $PREDICT_SINGLE_FILE_ATD failed. Please check log for details ret=${ret}" 
            echo "ERROR: $PREDICT_SINGLE_FILE_ATD failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi
    fi
# END upload_predict #########
# ensemble predict ############################
elif [[ "$action_type" == *"ensemble"*  ]];
then
    echo "INFO: In ensemble predict" 
    echo "INFO: In ensemble predict"  >> $logfile 2>&1
    # get python script
    get_script_name "$uploadtype"
    
    # create folder if not exist
    mkdir $out_dir
    
    # set tgt fname
    tgt_filename=$out_dir/$upload_filename

    # reset $out_dir point to result/ folder and uesed by multiple ensemble classifiers
    out_dir=$TRAIN_DES_DIR
    # move to $src_dir
    echo "INFO: mv -f $src_dir/$upload_filename $tgt_filename"
    mv -f $src_dir/$upload_filename $tgt_filename
    # get target fname
    chk_gz $tgt_filename
    
    echo Invoke Spark: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -ft $feat_threshold -dlist "$ds_list" 
    echo Invoke Spark: $spark_cmd $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -ft $feat_threshold -dlist "$ds_list" >> $logfile 2>&1
    $spark_cmd  $UF_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir  -i $cid -fw $fromweb -ft $feat_threshold -dlist "$ds_list" >> $logfile 2>&1
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $UF_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" 
        echo "ERROR: $UF_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi

# image predict ############################
elif [[ "$action_type" == *"image"*  ]];
then 
    echo "INFO: In image" 
    echo "INFO: In image"  >> $logfile 2>&1
    tgt_filename=$src_dir/$upload_filename

    echo "python $PREDICT_IMAGE -ifn $tgt_filename -r $rid -odn $out_dir -i $cid -fw $fromweb -fpt $pert_flag -mty $action_type " 
    echo "python $PREDICT_IMAGE -ifn \$tgt_filename -r $rid -odn $out_dir -i $cid -fw $fromweb -fpt $pert_flag -mty $action_type " >> $logfile
    python $PREDICT_IMAGE -ifn "$tgt_filename" -r "$rid" -odn "$out_dir" -i "$cid" -fw "$fromweb" -fpt "$pert_flag" -mty "$action_type" >> $logfile 2>&1
    
    if [ $ret -ne 0 ]
    then
        echo "Error!! $0 failed!! ret=${ret}"
        exit $ret
    fi

    
# hash_predict ############################
elif [ "$action_type" == "hash_predict" ]; 
then
    echo "INFO: In hash_predict" 
    echo "INFO: In hash_predict"  >> $logfile 2>&1
    # retrieve by query_mongo.py    
    echo Invoke: python $RETRIEVE_SCRIPT -sh $hash -o $RETR_DEC_DIR -m "hash" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw ""
    echo Invoke: python $RETRIEVE_SCRIPT -sh $hash -o $RETR_DEC_DIR -m "hash" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "" >> $logfile 2>&1
    python $RETRIEVE_SCRIPT -sh $hash -o $RETR_DEC_DIR -m "hash" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "$pwd" >> $logfile 2>&1
    ret=$?
    date +"INFO:%m/%d/%Y %H:%M:%S $HOSTNAME"
    date +"INFO:%m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1

    if [ $ret -ne 0 ]; then
        echo "ERROR: $RETRIEVE_SCRIPT failed. Please check log for details ret=${ret}" 
        echo "ERROR: $RETRIEVE_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
    ret=-1
    # predict
    tgt_filename=$RETR_DEC_DIR/$hash.gz
    echo "INFO: In hash_predict, tgt_filename=" $tgt_filename 
    echo "INFO: In hash_predict, tgt_filename=" $tgt_filename >> $logfile 2>&1
    # set program
    if [[ $uploadtype == *"in-dynamic"*  ]]; then
        H_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_IN_DYNAMIC
    else # static
        H_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_IN_STATIC
    fi
    # invoke predict 
    if [ "$ml_lib" == "mllib" ]; then
        echo Invoke: Spark $spark_cmd $H_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold  
        echo Invoke: Spark $spark_cmd $H_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
        $spark_cmd $H_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb  -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
    else
        echo Invoke: python $H_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold 
        echo Invoke: python $H_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
        python $H_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
    fi
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR: $H_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" 
        echo "ERROR: $H_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        exit $ret
    fi
# sample_predict? ############################
else
    echo "INFO: In sample prediction: uf=" $upload_filename
    echo "INFO: In sample prediction: uf=" $upload_filename >> $logfile 2>&1
    # TBD for IN
    if [[ $uploadtype == *"in-"*  ]]; then
        label=$(echo $upload_filename | awk -F . '{ print $2 }')
        md5=$(echo $upload_filename | awk -F . '{ print $1 }')

        # retrieve
        #echo python $RETR_SCRIPT -s $md5 -o $RETR_DEC_DIR -m 1 -l $label -fw $fromweb
        #echo python $RETR_SCRIPT -s $md5 -o $RETR_DEC_DIR -m 1 -l $label -fw $fromweb >> $logfile 2>&1
        #python $RETR_SCRIPT -s $md5 -o $RETR_DEC_DIR -m 1 -l $label -fw $fromweb  >> $logfile 2>&1
        echo Invoke: python $RETRIEVE_SCRIPT -sh $md5 -o $RETR_DEC_DIR -m "hash" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw ""
        echo Invoke: python $RETRIEVE_SCRIPT -sh $md5 -o $RETR_DEC_DIR -m "hash" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "" >> $logfile 2>&1
        python $RETRIEVE_SCRIPT -sh $md5 -o $RETR_DEC_DIR -m "hash" -ip "$host" -p $port -dn $db -t $tbl -jp $jstr_proj -un "$usr" -pw "$pwd" >> $logfile 2>&1
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "ERROR: $RETRIEVE_SCRIPT failed. Please check log for details ret=${ret}" 
            echo "ERROR: $RETRIEVE_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi

        #ret=-1
        # preproc
        #echo python $PREPROC_SCRIPT -d $RETR_DEC_DIR -o $PREP_DEC_DIR -fw $fromweb 
        #echo python $PREPROC_SCRIPT -d $RETR_DEC_DIR -o $PREP_DEC_DIR -fw $fromweb >> $logfile 2>&1
        #python $PREPROC_SCRIPT -d $RETR_DEC_DIR -o $PREP_DEC_DIR -fw $fromweb  >> $logfile 2>&1
        #ret=$?
        #if [ $ret -ne 0 ]; then
        #    echo "ERROR!! $PREPROC_SCRIPT failed. Please check log for details ret=${ret}" 
        #    echo "ERROR!! $PREPROC_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
        #    exit $ret
        #fi
        ret=-1
        # predict
        tgt_filename=$RETR_DEC_DIR/$md5.gz
        echo "INFO: In sample_predict, tgt_filename=" $tgt_filename
        echo "INFO: In sample_predict, tgt_filename=" $tgt_filename >> $logfile 2>&1
        # set program
        if [[ $uploadtype == *"in-dynamic"*  ]]; then
            SM_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_IN_DYNAMIC
        else # static
            SM_SINGLE_PREDICT_SCRIPT=$PREDICT_SINGLE_FILE_IN_STATIC
        fi
        # invoke ml_lib
        if [ "$ml_lib" == "mllib" ]; then
            echo Invoke: Spark $spark_cmd $SM_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold
            echo Invoke: Spark $spark_cmd $SM_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
            $spark_cmd $SM_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb  -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
        else
            echo Invoke: python $SM_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold 
            echo Invoke: python $SM_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
            python $SM_SINGLE_PREDICT_SCRIPT -d $tgt_filename -r $rid -o $out_dir -i $cid -fw $fromweb  -nb $n_gram -lb $ml_lib -pm "$opt_str" -dsid "$ds_id" -pp "$pca_opts" -vb $verbose -ft $feat_threshold >> $logfile 2>&1
        fi
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "ERROR: $SM_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" 
            echo "ERROR: $SM_SINGLE_PREDICT_SCRIPT failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi
    else # ATD
        #tgt_filename="${out_dir}/${rid}_sample_list/$tlabel/${upload_filename}"
        tgt_filename=${upload_filename}
        zip_fname=$UPLOAD_FULL_DIR/$filename
        echo "INFO: zip_fname="$zip_fname"<??="
        echo "INFO: zip_fname="$zip_fname"<??="  >> $logfile 2>&1
        echo "INFO: tgt_filename.="$tgt_filename
        echo "INFO: tgt_filename.="$tgt_filename >> $logfile 2>&1

        echo Invoke: python $PREDICT_SINGLE_FILE_ATD -zn $zip_fname -fn $tgt_filename -r $rid -o $out_dir  -i $cid -w $fromweb
        echo Invoke: python $PREDICT_SINGLE_FILE_ATD -zn $zip_fname -fn $tgt_filename -r $rid -o $out_dir  -i $cid -w $fromweb  >> $logfile 2>&1
        python $PREDICT_SINGLE_FILE_ATD -zn $zip_fname -fn $tgt_filename -r $rid -o $out_dir  -i $cid -w $fromweb >> $logfile 2>&1
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "ERROR: $PREDICT_SINGLE_FILE_ATD failed. Please check log for details ret=${ret}" 
            echo "ERROR: $PREDICT_SINGLE_FILE_ATD failed. Please check log for details ret=${ret}" >> $logfile 2>&1
            exit $ret
        fi
    fi
fi

#echo Invoke: python $PREDICT_SINGLE_FILE_ATD -d $tgt_filename -c $rid -o $out_dir  -s $feature_numb -i $cid
#echo Invoke: python $PREDICT_SINGLE_FILE_ATD -d $tgt_filename -c $rid -o $out_dir  -s $feature_numb -i $cid  > $logfile 2>&1
#python $PREDICT_SINGLE_FILE_ATD -d $tgt_filename -c $rid -o $out_dir -s $feature_numb -i $cid >> $logfile 2>&1
#ret=$?

# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "ERROR: $0 failed!! ret=${ret}"
    exit $ret
else
    echo "INFO: $0 done! ret=${ret}"
fi


#upload to HDFS
#echo Upload to HDFS 
#hdfs dfs -put out $to_hdfs_dir/$filename
#ret=$?

#echo " *Upload to HDFS, $to_hdfs_dir/$filename, done!! ret=${ret}"

#remove folder out
#rm -rf out

# time stampe
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME"
date +"INFO: %m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

