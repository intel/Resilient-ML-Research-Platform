'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

##### ngram extraction ######
import os
from argparse import ArgumentParser
import numpy as np

import zip_preprocess_pattern


# Input:
#   all_arr- array with all data [meta-data1,meta-data2,...,str_arr]
#   data_idx- index for str_arr("log array") element
#   token_dict- {"str1":1,"str2":2,...}
# Return:
#   array: [meta-data1,meta-data2,..., [token1,token2,...]]
def tokenize_by_dict(all_arr, data_idx, token_dict,label_idx, label_dict):
    # get log array
    log_arr=all_arr[data_idx]
    ret=[]
    for i in log_arr:
        if i in token_dict:
            ret.append(token_dict[i])
    arr_ret=all_arr[:data_idx]
    arr_ret.append(ret)
    # convert to number
    if not label_dict is None and all_arr[label_idx] in label_dict:
        arr_ret[label_idx]=label_dict[all_arr[label_idx]]
    return arr_ret

# Input:
#   all_arr- array with all data [meta-data1,meta-data2,...,str_arr]
#   data_idx- index for str_arr("log array") element
#   MAX_FEATURES: prime for string hash function
#   num_gram: n-gram numb to generate feature
# Return:
#   array: [meta-data1,meta-data2,..., hash_cnt_dic, hash_str_dic]
#       hash_cnt_dic: {hash,hash:count),...}  hash_str_dic: {hash: 'str1',... }
def feature_extraction_ngram(all_arr, data_idx, MAX_FEATURES, num_gram, n_gram_1='Y', has_str_dic='Y'):
    
    hash_cnt_dic = {}  ### 
    hash_str_dic = {} ### 
    # don't modify all_arr
    ret_arr=all_arr[:data_idx]
    data_idx=int(data_idx)
    
    if not all_arr or len(all_arr)<data_idx:
        return []
    # get log array
    log_arr=all_arr[data_idx]
    # log array exists
    if not log_arr or len(log_arr)==0:
        return []
    
    # convert string array to hash array =============
    hash_indvul=[] 
    for t in log_arr:
        ht=djb2(t,MAX_FEATURES)
        hash_indvul.append(str(ht))
        if has_str_dic == 'Y' and not ht in hash_str_dic:
            hash_str_dic[ht]=t 
    # hash_str_dic: {hash:str,...}
        
    #print "hash_indvul=",hash_indvul
    #print "hash_str_dic=",hash_str_dic

    # n_gram =1; by default to include  ================================== ==================    
    if n_gram_1=='Y' and num_gram !=1:
        #print "xx=",hash_indvul
        for str_gen in ngram_str(hash_indvul, 1, 1):
            #print "str_gen=",str_gen
            if str_gen and len(str_gen)>0:
                if str_gen in hash_cnt_dic:
                    hash_cnt_dic[str_gen] = hash_cnt_dic[str_gen] +1
                else:
                    hash_cnt_dic[str_gen] = 1
                    
    # get num_gram to hash_cnt_dic ================================== ==================
    for str_gen in ngram_str(hash_indvul, num_gram, num_gram):
        #print "ngram string=",str_gen
        if str_gen and len(str_gen)>0:
            #hash_ngram_str(str_gen, hash_cnt_dic, MAX_FEATURES) # not hash the ngram hash string
            if str_gen in hash_cnt_dic:
                hash_cnt_dic[str_gen] = hash_cnt_dic[str_gen] +1
            else:
                hash_cnt_dic[str_gen] = 1
                

    #replace log arr with hash_cnt_dic, add hash_str_dic, hash_str_dic
    #all_arr[data_idx]=hash_cnt_dic
    #all_arr.append(hash_str_dic)
    ret_arr.append(hash_cnt_dic)
    ret_arr.append(hash_str_dic)
    
    #print "all_arr=", all_arr
    #return all_arr
    return ret_arr


# convert dict to np array ============= =============
def dict2nparr(dict_feat, feat_count):
    ret=np.zeros((feat_count,), dtype="int")
    # libsvm is one based
    for i in dict_feat:
        ret[int(i)-1]=dict_feat[i]
        
    return ret
    
# hash ngram string into number  ============= =============
#  hash_cnt_dic: {"hash":(1,str),...}
def hash_ngram_str(str_gen, hash_cnt_dic, MAX_FEATURES):
    hash_ret = djb2(str_gen, MAX_FEATURES)
    column_str = str(hash_ret+1) # add 1, libsvm didn't like 0
    
    if column_str in hash_cnt_dic:
        hash_cnt_dic[column_str] = (hash_cnt_dic[column_str][0] + 1,str_gen)
    else:
        hash_cnt_dic[column_str] = (1,str_gen)
        
# string hash function   ============= =============
def djb2(key, MAX_FEATURES):
    hash = 5381 
    for k in key:
        hash = ((hash << 5) + hash) + ord(k) 
    hash_ret = hash % MAX_FEATURES
    return hash_ret

# return ngram str array; ngram=2 ['str1','str2']    ============= =============
def ngrams(str_arr, MIN_N, MAX_N ):
    str_count = len(str_arr)
    for i in xrange(str_count):
        for j in xrange(i+MIN_N, min(str_count, i+MAX_N)+1):
            try:
                yield str_arr[i:j]
            except:
                yield []
            
# return n gram string 'hash1,hash2,'     ============= =============
def ngram_str(str_arr, MIN_N, MAX_N):
    for str_arr in ngrams(str_arr, MIN_N, MIN_N):
        str_gen = ",".join(str_arr)
        yield str_gen

# test only   ============= =============
def main():
    # parse arguments
    parser = ArgumentParser(description=__description__)
    parser.add_argument('-tf', '--tf', type = str, metavar = 'test file', help = 'test file', required =False)
    
    args = parser.parse_args()
    # prime for string hash
    MAX_FEATURES=1442968193 #120011
    print "args.tf=",args.tf
    hash_cnt_dic = {}  ### for current file
    hash_str_dic = {} ### mapping between key_str and column

    num_gram=2
    f=open(args.tf, 'r')
    for line in f:
        ret=zip_preprocess_pattern.preprocess_pattern(line, metadata_count=3
            , pattern_str=r'^I/AndroidATD\([ ]*[\d]+\): \[API\] (.*) \[.*')

        print "pre pro=",ret
        rp=feature_extraction_ngram(ret,3, MAX_FEATURES, num_gram)
        print "f ng=",rp
        
if __name__ == '__main__':
    __description__ = "feature extraction"
    main()  