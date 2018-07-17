'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

import sys,os, math
import numpy as np
from argparse import ArgumentParser
from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.ml.linalg import SparseVector as ml_SparseVector
from pyspark.sql import Row, SQLContext



# based on excl_feat_list to set value to 0 to exclude features ================
# excl_feat_list is a csv string
def exclude_feature(label, size, indices, values, excl_feat_list):
    excl_dict={}
    # fix "ValueError: assignment destination is read-only", np array is readonly
    if type(values) is np.ndarray:
        lvalues=values.tolist()
    # convert exclude list to dict
    for i in excl_feat_list.split(','):
        excl_dict[i]=1
        
    for idx,val in enumerate(indices):
        if str(val+1) in excl_dict:
            lvalues[idx]=0.0
            
    return (label,size,indices,lvalues)

    '''
    libsvm_arr=str_libsvm.split(' ')
    # save label
    ret_str= libsvm_arr[0]
    # scan all features
    for i in libsvm_arr[1:]:
        f=i.split(':')
        if f[0] in excl_dict:
            # set exclude item to zero
            ret_str=ret_str+' '+f[0]+':0'
        else: 
            # keep the same
            ret_str=ret_str+' '+i
    
    return ret_str
    '''

# convert SparseVector to dict ================
def sparseVector2dict(sparse_vector):
    i_arr=sparse_vector.toArray()
    #print "t=",type(f_arr),"f_arr=",f_arr
    dict_out={}
    for i,v in enumerate(i_arr):
        if v>0:
            dict_out[str(i+1)]=v
    #print "len=",len(dict_out),"dict_out=",dict_out
    return dict_out

# get a dict of fid:(coef,raw string) ================
def get_dict_coef_raw4feat(dict_src, jfeat_coef_dict):
    dict_out={}
    for k,v in dict_src.iteritems():
        try:
           dict_out[k]=jfeat_coef_dict[k]
           
        except Exception as e:
            print "WARNING: at get_dict_coef_raw4feat(). e=",e
    return dict_out        
    
# convert string to (labeledPoint, hash) ================
def str2LabeledPoint_hash(feature_count, str_sample, has_hash="y"):
    #print "str_sample=",str_sample
    if str_sample is None:
        return None 
    str_sample=str_sample.strip()
    if len(str_sample)==0:
        return None
    str_arr= str_sample.split(" ") 
    if len(str_arr)<=2:
        return None
    # check if 1st item is integer
    int_1st="y"
    try:
        int(str_arr[0])
    except ValueError:
        int_1st="n"
    # check if 2nd item is integer
    int_2nd="y"
    try:
        int(str_arr[1])
    except ValueError: 
        int_2nd="n"
    
    #print "int_1st=",int_1st,",int_2nd=",int_2nd
    
    if int_1st=="n" and int_2nd=="y":
        has_hash="y"
    elif int_1st=="y" and int_2nd=="n":
        has_hash="n"
    elif int_1st=="y" and int_2nd=="y":
        has_hash="y"
        
    if has_hash=="n": # no hash for pure libsvm
        #print "*** in libsvm old *** "
        hash="-"
        label=int(str_arr[0])
        f_arr=str_arr[1:]
    else:  # libsvm with hash in front of each row
        hash=str_arr[0]
        label=int(str_arr[1])
        f_arr=str_arr[2:]
    idx_arr=[]
    val_arr=[]
    for f in f_arr:
        i,v=f.split(":")
        # SparseVector is zero based
        idx_arr.append(int(i)-1)
        val_arr.append(float(v))
    #return (LabeledPoint(label, SparseVector(feature_count, idx_arr, val_arr)) ,hash  )
    return (label, feature_count, idx_arr, val_arr, hash  )

# for ML to get samples in dataframe: label, features, hash
def get_sample_dataframe(sc, libsvm_data_file, feature_count, excluded_feat_cslist):
    samples_rdd, feature_count=get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist)
    #print type(samples_rdd)
    sqlCtx = SQLContext(sc)
    rows_rdd = samples_rdd.map(lambda p: (p[0].label,p[0].features.asML(),p[1]) ) \
        .map(lambda i: Row(label=i[0], features=i[1], hash=i[2]))
    sample_df = sqlCtx.createDataFrame(rows_rdd).cache()    
    
    return sample_df,feature_count
    
    
# for ML lib to get samples_rdd: (LabeledPoint, hash)
def get_sample_rdd(sc, libsvm_data_file, feature_count, excluded_feat_cslist):
    #TBD how to read .gz file here?
    # load libsvm text, expect format: <hash> <label> <feature:value> <> ...
    if feature_count>0:
        samples_rdd = sc.textFile(libsvm_data_file) \
            .map(lambda p: str2LabeledPoint_hash(feature_count,p)) \
            .map(lambda p: ( LabeledPoint(p[0], SparseVector(p[1], p[2], p[3])), p[4]  ) ) \
            .filter(lambda p: not p is None).cache() 
    else: # for original libsvm format, no feature count available
        # get feature_count from data; no hash in 1st column
        feature_count=get_feature_count_from_data(sc,libsvm_data_file)
        print "INFO: feature_count=",feature_count,"--"
        samples_rdd = sc.textFile(libsvm_data_file) \
            .map(lambda p: str2LabeledPoint_hash(feature_count,p,"n")) \
            .map(lambda p: ( LabeledPoint(p[0], SparseVector(p[1], p[2], p[3])), p[4]  ) ) \
            .filter(lambda p: not p is None).cache() 
    
    # Exclude selected features in sample, exclude_feature() return (label,size,indices,values) ================ =====
    # 
    if not excluded_feat_cslist is None:
        samples_rdd = samples_rdd \
            .map(lambda p: (p[0].label, p[0].features, p[1])) \
            .map(lambda p: (exclude_feature(p[0], p[1].size, p[1].indices, p[1].values, excluded_feat_cslist), p[2]) ) \
            .map(lambda p: (LabeledPoint(p[0][0], SparseVector(p[0][1], p[0][2], p[0][3])), p[1] ) ).cache()  

    return samples_rdd, feature_count

# Convert Sample RDD into 3 arrays in RDD and find max/min of feat value. for RDD
# return: (nparray, label, info)
def get_sample_as_chunk(sc, dnn_data_file, feature_count=None, excluded_feat_cslist=None, chunk_size=3000 \
    , padding_flag='pre', padding_val=0):
    #feature_count=get_feature_count_from_data(sc,dnn_data_file)
    print "INFO: feature_count=",feature_count,"--"
    samples_rdd = sc.textFile(dnn_data_file) \
        .map(lambda p: get_list_from_string(p)) \
        .filter(lambda p: len(p)>0).cache() 
    # format [info, label, [data]]
    stat= samples_rdd.map(lambda p: len(p[2])).stats()
    feat_cnt_max=stat.max()
    feat_cnt_mean=stat.mean()
    feat_cnt_stdev=stat.stdev()
    feat_cnt_var=stat.variance()
    sample_cnt=stat.count()
    print "INFO: sample count=",sample_cnt
    print "INFO: feat_cnt_max=",feat_cnt_max,",mean=",feat_cnt_mean,",stdev=",feat_cnt_stdev,",variance=",feat_cnt_var
    
    # set to max count or padding
    if feat_cnt_max<chunk_size:
        chunk_size=int(feat_cnt_max)
    else:
        if feat_cnt_mean+feat_cnt_stdev < chunk_size:
            chunk_size=int(feat_cnt_mean+feat_cnt_stdev)
        
    print "INFO: chunk_size=",chunk_size
    samples_rdd=samples_rdd.flatMap(lambda p: break2chuckOrPad(p[2],chunk_size,p[0],p[1])).cache()
    chucked_sample_count=samples_rdd.count()
    print "INFO: chucked_sample_count=",chucked_sample_count
    for i in samples_rdd.take(3):
        print i
    all_list=samples_rdd.collect()   

    return all_list,chucked_sample_count

# input: data array
# return: an array in chucks; [(nparray, label, info)]
def break2chuckOrPad(arr, chunk_size, info,label,  padding_flag='pre', padding_val=0):
    count=0
    arr_len = len(arr)
    if arr_len == chunk_size:
        return [(np.array(arr), label, info)]
    elif arr_len > chunk_size:
        chunks = []
        body=arr
        n_chunk = arr_len / chunk_size
        for i in np.arange(n_chunk):
            chunks.append(( np.array(body[i*chunk_size : (i+1)*chunk_size]), label, info))
        if arr_len % chunk_size != 0:
            chunks.append(( np.array(body[(arr_len-chunk_size) : arr_len]), label, info) )
        return chunks
    else: #padding
        ret=[]
        z = [padding_val]*(chunk_size-arr_len)
        if padding_flag == 'pre':
            ret=z+arr
        else:
            ret=arr+z        
        return [(np.array(ret), label, info)]
    
# convert string [label, info, ..., [feat id,...]], assume end element is the data
# return [info, label, [feat list]]
def get_list_from_string(str, del_char="\t"):
    if str is None or len(str)==0:
        return []
    try:
        # expect [item1,item2,...,[numb1,numb2,...]]
        if str.index('[')==0:
            arr=eval(str)
        else:
            # expect item1\titem2\t...\tnumb1,numb2,...\t
            arr=str.split(del_char)
            arr[-1]=arr[-1].split(',')
    except:
        print "ERROR: data format error"
        return []
        
    if not type(arr[-1]) is list:
        print "ERROR: data format error"
        return []
    
    ret=[]
    # info
    ret.append(arr[1])
    # label
    ret.append(arr[0])
    # data
    ret.append(arr[-1])
    
    return ret    
    
# Convert Sample RDD into 3 arrays in RDD and find max/min of feat value. for RDD
# return: [(nparray, label, info)], int, int
def get_sample_as_arr(sc, libsvm_data_file, feature_count, excluded_feat_cslist):
    samples_rdd, feature_count=get_sample_rdd(sc, libsvm_data_file, None, excluded_feat_cslist)
    # find max and min of feature value
    feat_rdd=samples_rdd.map(lambda p: (p[0].features.toArray(), p[0].label, p[1]) ).cache()
    feat_max=feat_rdd.flatMap(lambda p: [np.amax(p[0])]).max() # for Normalization the values
    feat_min=feat_rdd.flatMap(lambda p: [np.amin(p[0])]).min()
    print "INFO: feat_max=",feat_max, "feat_min=",feat_min
    
    # TBD ?? need to collect 3 list in one shot...  
    all_list=feat_rdd.collect()    
    
    return all_list,feature_count,feat_max,feat_min
    '''
    feats,labels,infos=zip(*all_list)
    feat_list=list(feats)
    label_list=list(labels)
    info_list=list(infos)
    
    # convert each element to np array
    #feat_list=feat_rdd.map(lambda p: p[0]).collect()
    # find array size/ feature count
    #arr_size = feat_list[0].shape[0]
    feat_count=len(feat_list[0])
    # sample count
    sample_count=len(feat_list)
    # label list
    #label_list=samples_rdd.map(lambda p: p[0].label).collect()
    # info list
    #info_list=samples_rdd.map(lambda p: p[1]).collect()
    return feat_list, label_list, info_lis, feat_count, feat_max, feat_min, sample_count
    '''
# get feature count from hdfs
def get_feature_count(sc,feat_count_file):
    feature_count=None
    try:
        feat_count_rdd=sc.textFile(feat_count_file)
        if not feat_count_rdd is None:
            feature_count=feat_count_rdd.collect()[0]
    except Exception as e:
        #print "WARNING: at get_feature_count(). e=",e
        print "WARNING: feature count not found! "
     
    if not feature_count is None:
        return int(feature_count)
    return feature_count

# get feature count from hdfs
def get_feature_count_from_data(sc,libsvm_data_file):
    #print "in get_feature_count_from_data()=",libsvm_data_file
    feature_count=sc.textFile(libsvm_data_file) \
            .filter(lambda p: not p is None) \
            .filter(lambda p: len(p.strip())>1) \
            .map(lambda p: p.strip().split(" ")[-1].split(":")[0]) \
            .filter(lambda p: len(p)>0) \
            .max(key=int)
    #print "hihi feature_count=",feature_count
    return int(feature_count)
    

# calculate_hypothesis for prediction; sparse array version =================
def calculate_hypothesis(feat_arr, coef_arr, intercept, model_name):
    hypothesis_val=feat_arr.dot(coef_arr)+intercept
    if model_name and "logistic" in model_name.lower():
        hypothesis_val=sigmoid(hypothesis_val) 
    return hypothesis_val
    
# for logistic regression to calculate hypothesis output =================
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# for ML lib to get samples_rdd as tuple arr
def get_sample_tuple_arr(sc, libsvm_data_file):
    # load libsvm text, expect format: <hash> <label> <feature:value> <> ...
    ret = sc.textFile(libsvm_data_file) \
        .map(lambda p: libsvm2tuple_arr(p)) \
        .filter(lambda p: not p is None) \
        .collect()
    return ret
  
# convert libsvm to array of tuple [label,(f,v),...] or [hash, label, (f,v),(),...]
def libsvm2tuple_arr(str, has_hash="y", out_hash="n"):
    if str is None or len(str)==0:
        return None
        
    arr=str.split(' ')
    data_idx=1
    ret=[]
    if has_hash=="y": 
        data_idx=2
        if out_hash=="n":
            ret.append(int(arr[1])) # label only
        else:
            ret.append(int(arr[0])) # hash
            ret.append(int(arr[1])) # label       
    else: # no hash
        ret.append(int(arr[0])) # label
        
    for item in arr[data_idx:]:
        fv=item.split(':')
        ret.append( ( int(fv[0]), int(fv[1]) ) )
    return ret


 
# convert DenseVector to libsvm text with space in front
def dv2libsvm(dv_arr):
    ret=""
    for idx,val in enumerate(dv_arr):
        ret=ret+" "+str(idx+1)+":"+str(val)
    return ret


# convert list to libsvm, index starting by 1, ignore None/Null value
def list2libsvm(in_list, label_index=0, add_meta=True, label_dic=None):
    out=""
    delm=""
    meta=""
    idx=1
    for i,v in enumerate(in_list):
        if v:
            if label_index == i: #label
                if label_dic:
                    meta=str(label_dic[v])+" "
                else:
                    meta=str(v)+" "
            else:
                out=out+delm+str(idx)+":"+str(v)
                delm=" "
                idx+=1
    # calculate hash for all data
    meta=str(djb2_(out))+" "+meta
    return meta+out

# string hash function   ============= =============
def djb2_(key, max_feat_cnt=1442968193):
    hash = 5381 
    for k in key:
        hash = ((hash << 5) + hash) + ord(k) 
    hash_ret = hash % max_feat_cnt
    return hash_ret
    
# test only   ============= =============
def main():
    # parse arguments
    parser = ArgumentParser(description=__description__)
    parser.add_argument('-svm', '--str_libsvm', type = str, metavar = 'libsvm string', help = 'libsvm string', required =False)
    parser.add_argument('-ef', '--excl_feat_list', type = str, metavar = 'excluded feature list', help = 'excluded feature list', required =False)
    
    args = parser.parse_args()
    # 
    print exclude_feature(args.str_libsvm, args.excl_feat_list)

if __name__ == '__main__':
    __description__ = "feature extraction"
    main()  