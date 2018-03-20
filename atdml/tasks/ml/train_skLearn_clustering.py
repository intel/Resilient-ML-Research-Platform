#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
#
# python libraries
import os
import os.path, glob
import re
import random
import sys, ConfigParser
import math
import numpy as np
import datetime
import zipfile
import shutil
import json

from argparse import ArgumentParser
from scipy.sparse import *
from scipy import *
from time import time
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.stats import entropy
from sets import Set


####pyspark#####
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils

#####matplotlib###############
import matplotlib
matplotlib.use('Agg')
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import matplotlib.cm as cmx
'''
#####import for django database####
sys.path.append('./db')
import exec_sqlite

####import our own library####
import query_mongo
import ml_util
from ml_util import *
import zip_feature_util

CONF_FILE='../../app.config' # at the base dir of the web
config=ConfigParser.ConfigParser()
config.read(CONF_FILE)
portion = eval(config.get("machine_learning","training_portion"))
mtx_name_list = config.get("machine_learning","mtx_name_list")
mtx_libsvm = config.get("machine_learning","mtx_libsvm")
mtx_stat = config.get("machine_learning","mtx_stat")


def main():
    
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-f", "--folder", type=str, metavar="folder of features", help="hdfs folder contains features", required=False)
    parser.add_argument("-n", "--name", type=str, metavar="file name", help="filename for hdfs input data", required=False)
    parser.add_argument("-o", "--out", type=str, metavar="out figure folder", help="folder contains output", required=False)
    parser.add_argument("-r", "--row_id", type=str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-mf", "--modelfolder", type=str, metavar="model folder", help="model for prediction", required=False)
    parser.add_argument("-w", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument("-pm", "--parameter", type=str, metavar="parameters in json", help="json string contains learning alg and parameter selection", required=False)
    parser.add_argument("-sl", "--showlabelname", type=str, metavar="show label name", help="0: not shown; 1: show label name", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)

    parser.add_argument('-sp','--sp_master', type=str, dest='sp_master', help='spark.master'
                , default =config.get('spark', 'spark_master'))
    parser.add_argument('-em','--exe_memory', type=str, dest='exe_memory', help='spark.executor.memory'
                , default =config.get('spark', 'spark_executor_memory'))
    parser.add_argument('-cm','--core_max', type=str, dest='core_max', help='spark.cores.max'
                , default =config.get('spark', 'spark_cores_max'))

    #### get dic_name_label from mongo db
    parser.add_argument('-ip','--ip_address', type=str, dest='ip_address', help='mongodb ip address'
                , default =config.get('mongo', 'out_ip_address'))
    parser.add_argument('-p','--port', type=str, dest='port', help='mongodb port'
                , default =eval(config.get('mongo', 'out_port')))
    parser.add_argument('-dn','--db_name', type=str, dest='db_name', help='mongodb db name'
                , default =config.get('mongo', 'out_db'))
    parser.add_argument('-t','--tb_name', type=str, dest='tb_name', help='mongodb table name'
                , default =config.get('mongo', 'out_tb'))
    # auth
    parser.add_argument('-un','--username', type=str, dest='username', help='mongodb username'
                , default =config.get('mongo', 'out_username'))
    parser.add_argument('-pw','--password', type=str, dest='password', help='mongodb password'
                , default =config.get('mongo', 'out_password'))  
    # zip spark code
    parser.add_argument("-zo", "--zipout_dir", type=str, metavar="out figure folder", help="folder contains python code", required=False)
    parser.add_argument("-zc", "--zipcode_dir", type=str, metavar="out code", help="out code folder for python code", required=False)
    parser.add_argument("-zf", "--zipfilename", type=str, metavar="python code zip file", help="python code zip file for distribution to works", required=False)
                
    args = parser.parse_args()
    
    if args.folder:
        hdfs_feat_dir = args.folder
    else:
        hdfs_feat_dir  = config.get('app', 'HADOOP_MASTER')+'/user/hadoop/yigai/sality_virut_zbot_backdoor_hash_000'
    if args.name:
        src_filename = args.name
    else:
        src_filename  = 'aaaa'
    if args.out:
        local_out_dir = args.out
    else:
        local_out_dir  = 'out_result'
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = '88'
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = ''

    if args.modelfolder:
        model_data_folder = args.modelfolder
    else:
        model_data_folder  = os.path.join(local_out_dir , row_id_str + '_model/')

    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None
    if args.parameter:
        ml_opts_jstr = args.parameter
    else:
        ml_opts_jstr  = '{"learning_algorithm":"linear_svm", "c":"1", "regularization":"l2", "kernel":"rbf", "gamma":"0", "degree":"3", "nu":"0.1"}'
    if args.showlabelname: # mllib or scikit
        labelnameflag = eval(args.showlabelname)
    else:
        labelnameflag = 0
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None 
    if args.zipout_dir:
        zipout_dir = args.zipout_dir
    else:
        zipout_dir  = './ml'
    if args.zipcode_dir:
        zipcode_dir = args.zipcode_dir
    else:
        zipcode_dir  = './ml'
    if args.zipfilename:
        zip_file_name = args.zipfilename
    else:
        zip_file_name  = 'train_sklearn_clustering.zip'

    # mongo info for connection
    mongo_tuples=(args.ip_address, args.port, args.db_name, args.tb_name, username, password)

    return train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr
    , args.sp_master,config.get('spark', 'spark_rdd_compress'),config.get('spark', 'spark_driver_maxResultSize'), args.exe_memory, args.core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb, src_filename
    , 'train_sklearn_clustering:'+row_id_str, model_data_folder )

# ================================================================================== train () ============ 
def train(row_id_str, ds_id, hdfs_feat_dir, local_out_dir, ml_opts_jstr
    , sp_master, spark_rdd_compress, spark_driver_maxResultSize, sp_exe_memory, sp_core_max
    , zipout_dir, zipcode_dir, zip_file_name
    , mongo_tuples, labelnameflag, fromweb, src_filename
    , jobname, model_data_folder ): 
        

    # create zip files for Spark workers ================= ================
    zip_file_path = ml_build_zip_file(zipout_dir, zipcode_dir, zip_file_name, prefix='zip_feature_util')
    print "INFO: zip_file_path=",zip_file_path

    #data_folder = hdfs_feat_dir + "/"
    #local_out_dir = local_out_dir + "/"
    if not os.path.exists(local_out_dir):
        os.makedirs(local_out_dir)

    # ML model filename ====
    model_fname=os.path.join(model_data_folder , row_id_str+'.pkl')
    print "INFO: model_data_folder=",model_data_folder    
    # create out folders and clean up old model files ====
    ml_util.ml_prepare_output_dirs(row_id_str,local_out_dir,model_data_folder,model_fname)   

    # init Spark context ====
    sc=ml_util.ml_get_spark_context(sp_master
        , spark_rdd_compress
        , spark_driver_maxResultSize
        , sp_exe_memory
        , sp_core_max
        , jobname
        , [zip_file_path]) 

    # start here =================================================================== ===============
    t0 = time()
    
    
    ### load libsvm file: may or may not be PCA-ed ###
    libsvm_data_file = os.path.join(hdfs_feat_dir , src_filename)
    print "INFO: libsvm_data_file=",libsvm_data_file

    # feature count is a variable if PCA
    feature_count=0
    
    # samples_rdd may be from PCAed data
    # load sample RDD from text file   
    # format (LabeledPoint,hash) from str2LabeledPoint_hash() 
    samples_rdd, feature_count = zip_feature_util.get_sample_rdd(sc, libsvm_data_file, feature_count, '')

    # collect all data to local for processing ===============
    all_data = samples_rdd.collect()
    total_sample_count=len(all_data)
    # 2-D array, may be PCAed
    features_list = [x.features.toArray() for x,_ in all_data]
    # label array
    labels_list_all = [x.label for x,_ in all_data]
    # hash array
    hash_list_all = [x for _,x in all_data]
    # convert to np array
    features_array_reduced = np.array(features_list)
    hash_list_all=np.array(hash_list_all)
    labels_list_all = np.array(labels_list_all) 
    true_label_array = np.array(labels_list_all, dtype = np.int8)
    
    print "INFO: total_sample_count=",total_sample_count
    print "INFO: features_array_reduced.shape=",features_array_reduced.shape
    print "INFO: labels_list_all.shape=",labels_list_all.shape
    print "INFO: true_label_array.shape=",true_label_array.shape


    t1 = time()
    print 'INFO: data generating time: %f' %(t1-t0)

    
    ###############################################
    ########## build learning model ###############
    ###############################################
    
    ### parse parameters and generate the model ###
    (model, alg, n_clusters) = parse_para_and_get_model(ml_opts_jstr)
    if model is None:
        return
    
    labels_kmeans=None
    #### fit the model to training dataset ####
    try:
        model.fit(features_array_reduced)
        labels_kmeans = model.labels_  #'numpy.ndarray'

    except:
        print "ERROR: Error in model.fit(): ","model=",model,", sys.exc_info:", sys.exc_info()[0]
        return
        
    #### save clf for future use ####
    #joblib.dump(model, model_data_folder + row_id_str+'.pkl') 
    joblib.dump(model, model_fname) 
    
    #print "**model:intercept***"
    #print clf.intercept_

    print "INFO: model type=",type(model)," model=",model

    ###################################################
    ### generate label names (family names) ###########
    ### connect to database to get the column list which contains all column number of the corresponding feature####
    ###################################################
    
    if labelnameflag == 1:
        key = "dic_name_label"
        jstr_filter='{"rid":'+row_id_str+',"key":"'+key+'"}'
        jstr_proj='{"value":1}'
        
        # get parent dataset's data
        if ds_id != row_id_str:
            jstr_filter='{"rid":'+ds_id+',"key":"'+key+'"}'
        
        doc=query_mongo.find_one_t(mongo_tuples, jstr_filter, jstr_proj)
        dic_list = doc['value']
        
        label_dic = {}
        for i in range(0, len(dic_list)):
            for key in dic_list[i]:
                label_dic[dic_list[i][key]] = key.encode('UTF8')
        print "INFO: label_dic:", label_dic
    else:
        label_dic = {}
        label_set = set(labels_list_all)
        for label_value in label_set:
            label_dic[int(label_value)] = str(int(label_value))
        print "INFO: generated label_dic:", label_dic        
    
    labels_list = []
    for key in sorted(label_dic):
        labels_list.append(label_dic[key])
    print "INFO: labels_list=", labels_list   
    
    #Adjusted Mutual Information between two clusterings
    amis=adjusted_mutual_info_score(labels_list_all,labels_kmeans)
    print "INFO: Adjusted_mutual_info_score=", amis   
    #Similarity measure between two clusterings
    ars=adjusted_rand_score(labels_list_all,labels_kmeans)
    print "INFO: Adjusted_rand_score=", ars   
    
    
    ###################################################
    #######plot histogram                       ####
    ###################################################
    plot_col_num = int(math.ceil(math.sqrt(n_clusters)))
    figsize = (4*plot_col_num, 3*int(math.ceil(n_clusters*1.0/plot_col_num)))
    print "INFO: labels_list_all.shape=",labels_list_all.shape,"labels_kmeans.shape=",labels_kmeans.shape
    print "INFO: labels_list_all t=",type(labels_list_all),"labels_kmeans t=",type(labels_kmeans)
    print "INFO: n_clusters=",n_clusters,",label_dic=",label_dic
    print "INFO: plot_col_num=",plot_col_num,",figsize=",figsize,",local_out_dir=",local_out_dir
    
    # kmeans histogram
    _, p_true = ml_plot_kmeans_histogram_subfigures(labels_list_all, labels_kmeans, n_clusters, names = label_dic
                        , plot_col_num = plot_col_num, figsize=figsize, folder = local_out_dir, rid=row_id_str)
    # normalized kmeans histogram
    _, p_true_norm = ml_plot_kmeans_histogram_subfigures(labels_list_all, labels_kmeans, n_clusters, names = label_dic
                        , plot_col_num = plot_col_num, figsize=figsize, normalize = True, folder = local_out_dir, rid=row_id_str)

    ####plot "reverse" histogram with labels ####
    #num_bars = len(np.unique(labels_list_all))
    num_bars = max(labels_list_all) + 1
    figsize = (4*plot_col_num, 3*int(math.ceil(num_bars*1.0/plot_col_num)))
    
    _, p_cluster = ml_plot_kmeans_histogram_subfigures(labels_kmeans, labels_list_all, num_bars, names = label_dic
                        , plot_col_num = plot_col_num, figsize=figsize, reverse = True, folder = local_out_dir, rid=row_id_str)
    
    #### plot dot figures ####
    #mtx_label = model.labels_
    mtx_center = model.cluster_centers_
    # dot plot for Kmeans   ===========
    filename=os.path.join(local_out_dir ,row_id_str+'_cluster.png')   
    filename_3d=os.path.join(local_out_dir ,row_id_str+'_cluster_3d.json')  
    ml_plot_kmeans_dot_graph_save_file(features_array_reduced, labels_kmeans, mtx_center, n_clusters, figsize=(10,7), filename=filename
        , title='KMeans', filename_3d=filename_3d)
        
    # dot plot for True Labels  ===========
    filename=os.path.join(local_out_dir ,row_id_str+'_cluster_tl.png')      
    filename_3d=os.path.join(local_out_dir ,row_id_str+'_cluster_3d_tl.json')  
    ml_plot_kmeans_dot_graph_save_file(features_array_reduced, true_label_array, mtx_center, n_clusters, figsize=(10,7), filename=filename
        , title='True Labels', filename_3d=filename_3d)
         
    dataset_info={"training_fraction":1, "class_count":n_clusters,"dataset_count":total_sample_count}
    # only update db for web request   ===========
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set accuracy = '" \
            +"', status = 'learned', processed_date ='"+str(datetime.datetime.now()) \
            +"', total_feature_numb='"+str(feature_count) \
            +"', perf_measures='{}" \
            +"', dataset_info='"+json.dumps(dataset_info) \
            +"' where id="+row_id_str
        ret=exec_sqlite.exec_sql(str_sql)
        print "INFO: Data update done! ret=", str(ret)
    else:
        print "INFO: accuracy = '"+str(accuracy*100)+"%"
    
    t1 = time()
    print 'INFO: running time: %f' %(t1-t0)
    
    #print 'Finished!'
    return 0

def parse_para_and_get_model(ml_opts_jstr):
    
    para_dict = json.loads(ml_opts_jstr)
    alg = para_dict['learning_algorithm']  
    model=None
    
    ###parse and print print parameters###
    print "INFO: ============Learning Algorithm and Parameters============="    
    
    if alg == "kmeans":
        ### 1: linearSVM
        iteration = int(para_dict['iteration'])     
        k = int(para_dict['k'])   
        init = para_dict['init']     

        print "INFO: Learning Algorithm: ", alg
        print "INFO: k = ", k       
        print "INFO: init = ", init       
        print "INFO: ==================== Kmeans ============="
        model = KMeans(init=init, n_clusters=k,max_iter=iteration, n_init=10)

    return (model, alg, k)
    
if __name__ == '__main__':
    __description__ = "ML single run to show accuracy, roc, generate model, etc"
    main()
'''
    libsvm_data_file = os.path.join(data_folder , "libsvm_data")
    print "libsvm_data_file=",libsvm_data_file
    
    samples_rdd = MLUtils.loadLibSVMFile(sc, libsvm_data_file)
    
    #print samples_rdd.count()
    labels_and_features_rdd = samples_rdd.map(lambda p: (p.label, p.features))
    all_data = labels_and_features_rdd.collect()
    features_list = [x.toArray() for _,x in all_data]
    labels_list_all = [x for x,_ in all_data]
    labels_list_all = array(labels_list_all)
    
    #features_array = np.array(features_list)
    features_array = np.array(features_list, dtype = np.int8)
    true_label_array = np.array(labels_list_all, dtype = np.int8)

    # PCA here, TBD for removal
    print "features_array.shape=",features_array.shape
    # clean up?
    #features_array[features_array > 0.5] = 1
    #features_array[features_array != 1] = 0
    features_array = features_array.astype(np.int8, copy=False)
    #print "features_list=",features_list    
    #ml_pickle_save(features_array, 'mtx.pickle')
    #X=ml_pickle_load('mtx.pickle')    
    #print "XXXXXXXXXX=",X

    var_thresh = 0.9
    
    reduced_feat_filename=os.path.join(local_out_dir,row_id_str+"_feat_pca.pickle")
    print "reduced_feat_filename=",reduced_feat_filename
    # if not exist, create one
    if not os.path.exists(reduced_feat_filename):
        print "Doing PCA..., shape=", features_array.shape
        features_array_reduced = ml_sklearn_PCA_transform(features_array, var_thresh)
        #print "Done PCA"
        if features_array_reduced:
            ml_pickle_save(features_array_reduced, reduced_feat_filename)
        else: 
            return -1 # error occur
        #print "features_array_reduced created!"
    else:
        features_array_reduced=ml_pickle_load(reduced_feat_filename)  
        print "features_array_reduced loaded!"

    ### generate sparse matrix (csr) for all samples
    #features_sparse_mtx = csr_matrix(features_array)
    
    ### randomly split the samples into training and testing data
    #X_train_sparse, X_test_sparse, labels_train, labels_test = cross_validation.train_test_split(features_sparse_mtx, labels_list_all, test_size=0.4)
        
'''

    
'''
#######################
####plot functions ####
#######################
# also generate data files for 3D 
def ml_plot_kmeans_dot_graph_save_file(mtx_feat, mtx_label, mtx_center, n_clusters, figsize=(10,7), filename=None
        , title='KMeans', filename_3d=None): 
    PLOT_GRID_TH=111 # 1x1 grid, 1st subplot
    # 
    cmap = get_cmap(n_clusters+1)
    colors = []
    for i in range(n_clusters):
        col = cmap(i)
        colors.append(col)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(PLOT_GRID_TH)

    print "mtx_feat.shape=",mtx_feat.shape #,", mtx_feat=",mtx_feat
    print "mtx_center.shape=",mtx_center.shape #,", mtx_center=",mtx_center
    # plot for each color
    g_data={}
    id_arr=[]
    x_arr=[]
    y_arr=[]
    z_arr=[]
    style_arr=[]
    count=1
    
    cluster_color=zip(range(n_clusters), colors)
    print "**zip(range(n_clusters), colors)=", cluster_color
    for k, col in cluster_color: #zip(range(n_clusters), colors):
        my_members = mtx_label == k
        
        #print "my_members shape=",my_members.shape  #,", my_members=",my_members
        print "mtx_feat[my_members, 0].shape=",mtx_feat[my_members, 0].shape #,", mtx_feat[my_members, 0]=",mtx_feat[my_members, 0]
        #print "mtx_feat[my_members, 1].shape=",mtx_feat[my_members, 1].shape #,", mtx_feat[my_members, 1]=",mtx_feat[my_members, 1]
        #print "mtx_feat[my_members, 2].shape=",mtx_feat[my_members, 2].shape #,", mtx_feat[my_members, 2]=",mtx_feat[my_members, 2]
        cluster_center = mtx_center[k] 
        print " k=",k,",cluster_center[0]=",cluster_center[0],", cluster_center[1]=",cluster_center[1]
        g={}

        # add dots for 3d  here
        x_list=mtx_feat[my_members, 0].tolist()
        x_len=len(x_list)
        x_arr=x_arr+x_list
        y_arr=y_arr+mtx_feat[my_members, 1].tolist()
        z_arr=z_arr+mtx_feat[my_members, 2].tolist()
        style_arr=style_arr+([k]*x_len) # center share same style
        # fill out id_arr with sequential number?
        for i in range(count,x_len+count): 
            id_arr.append(i)
        count=count+x_len
        
        # plot data points
        ax.plot(mtx_feat[my_members, 0], mtx_feat[my_members, 1], 'w',
                markerfacecolor=col, marker='.' , markersize=6)
        # moved to below
        #ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #        markeredgecolor='k', markersize=10)
    #add centers
    for c, col in cluster_color: #range(n_clusters):
        x_arr.append(mtx_center[c][0])
        y_arr.append(mtx_center[c][1])
        z_arr.append(mtx_center[c][2])
        id_arr.append(c*-1)
        style_arr.append(n_clusters)
        # plot centre dots after data dots
        ax.plot(mtx_center[c][0], mtx_center[c][1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

    g_data["x"]=x_arr
    g_data["y"]=y_arr
    g_data["z"]=z_arr
    g_data["id"]=id_arr
    g_data["style"]=style_arr
    
    ax.set_title(title)
    #ax.set_xticks(())
    #### annotate cluster index ###
    for k in range(n_clusters):
        cluster_center = mtx_center[k]
        annotate_str = str(k)
        ax.annotate(annotate_str, xy=(cluster_center[0], cluster_center[1]), 
                rotation=15, 
                verticalalignment='bottom', 
                horizontalalignment='left')
    if filename:
        fig.savefig(filename)
        print "2D Figure saved=" ,filename


    if filename_3d:
        #### generate 3d data file###
        if os.path.exists(filename_3d):
            try:
                os.remove(filename_3d)
            except OSError, e:
                print ("Error: %s - %s." % (e.filename_3d,e.strerror))
        with open(filename_3d, 'w') as f:
            json.dump(g_data,f)
            print "filename_3d=",filename_3d

    #print "========== g_data =",g_data
    return fig    
    
#Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color.  ===========
def get_cmap(N):
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


#  ml_plot_kmeans_histogram_subfigures ==========================
def ml_plot_kmeans_histogram_subfigures(labels_true, labels_predicted, n_clusters, names = {}, num_names_to_show = 3
            , plot_col_num = 4, figsize=(16,12), reverse = False, normalize = False, folder = 'results', rid=""):
        
    K = int(math.ceil(n_clusters*1.0/plot_col_num))
    fig, ax_all = plt.subplots(K, plot_col_num, sharex=True, figsize=figsize)
    ax_all = ax_all.flatten()
    #true_label_num = len(np.unique(labels_true))
    true_label_num = int(max(labels_true) + 1)
    
    if num_names_to_show > true_label_num:
        num_names_to_show = true_label_num
        print "   warning: n_clusters < num_names_to_show, so set num_names_to_show = ", num_names_to_show
    
    if normalize == True:
        sample_num_vec = []
        for k in range(true_label_num):
            num = len(labels_true[labels_true == k])
            if num > 0:
                sample_num_vec.append(len(labels_true[labels_true == k]))
            else:
                sample_num_vec.append(0.0001)
        #print "sample_num_vec:", sample_num_vec
        
    for k in range(int(n_clusters)):
            
        n, bins = np.histogram(labels_true[labels_predicted == k], bins=range( true_label_num+1) )
        if np.sum(n) > 0:
            if normalize == True:
                nor = 1.0 * n / np.array(sample_num_vec) * sum(sample_num_vec)       
                #print "normalized n:", nor
                p = 1.0 * nor / np.sum(nor)
            else:
                p = 1.0 * n / np.sum(n)
        else:
            p = 1.0 * n

        for i in range(len(p)):
            if p[i] > 1:
                p[i] = 0

        pp = np.expand_dims(p, axis=0)
        if k == 0:
            all_p = pp
        else:
            all_p = np.concatenate((all_p, pp), axis=0)
        
        
        shownames = np.arange(len(p))
        shownames = map(str, shownames)
        
        index_according_to_order = [i[0] for i in sorted(enumerate(p), key=lambda x:x[1], reverse=True)]
        for i in range(num_names_to_show):
            idx = index_according_to_order[i]
            if reverse == False:
                if (len(names) > 0):
                    shownames[idx] = str(idx) + ":" + names[idx]
                else:
                    pass
            else:   #reverse == True
                shownames[idx] = str(idx) + ":" + str(idx)
                    
        
        if reverse == False:
            title_str = 'cluster %d (' % k + '%d)' % sum(n) + ', H:%.2f' % entropy(p)
        else:
            if np.sum(p) > 0:
                title_str = '%d: '%k + '%s (' % names[k] + '%d)' % sum(n) + ', H:%.2f' % entropy(p)
            else:
                title_str = '%d: '%k + '%s (' % names[k] + '%d)' % sum(n)
        
        plot_one_multinomial(p, bins, names = shownames, title = title_str, label = '1', color='b', ax=ax_all[k])
        #ax_all[k].legend(loc='upper right')
        plt.tight_layout()
    if (reverse == False) and (normalize == False):
        fig.savefig(os.path.join(folder ,rid+'_histogram_true_labels.png'))
        print "Figure saved: " ,os.path.join(folder ,rid+'_histogram_true_labels.png')
    elif (reverse == False) and (normalize == True):
        fig.savefig(os.path.join(folder ,rid+'_histogram_true_labels_normalized.png'))
        print "Figure saved: " , os.path.join(folder ,rid+'_histogram_true_labels_normalized.png')
    else:
        fig.savefig(os.path.join(folder ,rid+'_histogram_predicted_labels.png'))
        print "Figure saved: " ,os.path.join(folder ,rid+'_histogram_predicted_labels.png')
    return fig, all_p

# plot_one_multinomial ==========================
def plot_one_multinomial(vals, bins, names, title='probability distribution', label='', ax=None, color='b'):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    #ind = np.arange(len(vals))
    rects = ax.bar(bins[0:-1], vals, color=color, align='center', width=0.9)
    ax.set_xticks(bins[0:-1])
    ax.set_xlim((-1,len(vals)))
    ax.set_ylim((0,max(vals)+0.1))
    
    ax.set_title(title)
    
    idx_sorted = np.argsort(vals)
    for i in range(len(vals)):
        ii = idx_sorted[-i]
        ax.annotate(names[ii], xy=(ii, vals[ii]), 
                    rotation=15, 
                    verticalalignment='bottom', 
                    horizontalalignment='left')
'''    

