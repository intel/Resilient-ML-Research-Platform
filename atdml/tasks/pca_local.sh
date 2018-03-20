#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
# load config file. may start at different folder
ROOT_DIR=/home/django/myml

sed '/=/!d;/\[\.*/d;/^ *#/d;' < $ROOT_DIR/app.config > /tmp/$$.tmp
source /tmp/$$.tmp
rm -rf /tmp/$$.tmp

# cd to working dir
cd $ROOT_DIR/$WORKING_DIR
#WORKING_DIR=atdml/tasks

# load params
rid=$1
pca_k=$2
ds_id=$3
libsvm_str=$4  

spark_cmd=$SPARK_SUBMIT #/home/hadoop/spark-1.2.0-bin-hadoop2.4/bin/spark-submit


# web local output dir 
out_dir=$TRAIN_DES_DIR/${rid}
mkdir $out_dir
out_file=$out_dir/${rid}pca_local.txt

sparkurl=$SPARK_URL
logfile=$LOG_DIR/${rid}pca_local.log


ret=-1

# time stampe
date +"%m/%d/%Y %H:%M:%S $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $HOSTNAME" > $logfile 2>&1
STARTTIME=$(date +%s)


# run Scala PCA for MLlib
ret=-1

# by Scala
pca_jar=$PCA_ML  #"lib/ml_scala_util_2.10-1.0.0.jar"
pca_class="ML_PCA_TRANSFORM"
dep_jars=$PCA_DEP_JARS
pca_mdl_fname=$TRAIN_DES_DIR/${rid}/${rid}_model/${rid}_pca_$pca_k.spkl

if [ "$rid" != "$ds_id" ];
then
    pca_mdl_fname=$TRAIN_DES_DIR/${ds_id}/${ds_id}_model/${ds_id}_pca_$pca_k.spkl
fi

#pca_k="0"  # how to decide k? 
echo "INFO: PCAModel file= "$pca_mdl_fname", out file:"$out_file   
echo "INFO: PCAModel file= "$pca_mdl_fname", out file:"$out_file  >> $logfile
echo Invoke: Spark $spark_cmd --class $pca_class --master $sparkurl --jars $dep_jars $pca_jar $pca_mdl_fname "str" "$libsvm_str" $out_file
echo Invoke: Spark $spark_cmd --class $pca_class --master $sparkurl --jars $dep_jars $pca_jar $pca_mdl_fname "str" "$libsvm_str" $out_file >> $logfile
$spark_cmd --class $pca_class --master $sparkurl --jars $dep_jars $pca_jar $pca_mdl_fname "str" "$libsvm_str" $out_file >> $logfile  2>&1
ret=$?

# check return code and echo message
if [ $ret -ne 0 ]
then
    echo "Error!! $0 failed!! ret=${ret}" 
    echo "Error!! $0 failed!! ret=${ret}" >> $logfile
    exit $ret
else
    echo "$0 done! ret=${ret}" 
    echo "$0 done! ret=${ret}" >> $logfile
fi


# time stampe
date +"%m/%d/%Y %H:%M:%S $HOSTNAME"
date +"%m/%d/%Y %H:%M:%S $HOSTNAME" >> $logfile 2>&1
./elapse_time.sh $STARTTIME $logfile

exit 0

