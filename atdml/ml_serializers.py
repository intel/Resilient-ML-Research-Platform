'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
# -*- coding: utf-8 -*-
#from rest_framework import serializers
#from atdml.models import *

#==== EOLed ... ========================================================= Doc2json ==================
def doc2json(documents):        
    ret=[]
    for doc in documents:
        jd={}
        jd['id']=doc.id
        jd['status']=doc.status
        jd['filename']=doc.filename
        jd['desc']=doc.desc
        jd['file_type']=doc.file_type
        jd['local_created_date']=doc.local_created_date()
        jd['local_processed_date']=doc.local_processed_date()
        jd['accuracy']=doc.accuracy
        jd['mean']=doc.mean
        jd['status_code']=doc.status_code
        jd['submitted_by']=doc.submitted_by
        jd['class_numb']=doc.class_numb
        
        jd['ml_n_gram']=doc.ml_n_gram
        jd['ml_lib']=doc.ml_lib
        jd['ml_opts']=doc.ml_opts
        jd['ml_feat_opts']=doc.ml_feat_opts
        
        jd['db_host']=doc.db_host
        jd['db_port']=doc.db_port
        jd['db_db']=doc.db_db
        jd['db_tbl']=doc.db_tbl
        jd['db_proj']=doc.db_proj
        jd['db_filter']=doc.db_filter
        jd['db_lb_field']=doc.db_lb_field
        jd['db_lb_mapping']=doc.db_lb_mapping
        jd['ds_list']=doc.ds_list
        jd['perf_measures']=doc.perf_measures
        jd['prediction']=doc.prediction
        ret.append(jd)
    return ret

#============================================================= pred2json ==================
def pred2json(documents):        
    ret=[]
    for doc in documents:
        jd={}
        jd['id']=doc.id
        jd['status']=doc.status
        jd['prediction']=doc.prediction
        jd['predict_val']=doc.predict_val
        jd['filename']=doc.filename
        jd['train_id']=doc.train_id
        jd['true_label']=doc.true_label
        jd['feat_list']=doc.feat_list
        jd['processed_date']=doc.local_processed_date()
        jd['submitted_by']=doc.submitted_by
        jd['db_host']=doc.db_host
        jd['db_db']=doc.db_db
        jd['file_type']=doc.file_type

        ret.append(jd)
    return ret    

''' compatibility issue with Django 1.10
class PredictSerializer(serializers.ModelSerializer):
    # for web api: prediction list, TBD not support json object datatype...
    class Meta:
        model = Document
        fields = ('id', 'status', 'prediction','predict_val', 'filename', 'train_id','true_label' \
            ,'feat_list','processed_date','submitted_by','db_host','db_db', 'file_type')

class DatasetSerializer(serializers.ModelSerializer):
    # for web api: dataset list
    class Meta:
        model = Document
        fields = ('id', 'status', 'filename', 'desc','file_type','local_created_date','local_processed_date'
                ,'accuracy','mean', 'status_code','submitted_by','class_numb'
                ,'ml_n_gram','ml_lib','ml_opts'
                ,'db_host','db_port','db_db','db_tbl','db_proj','db_filter','db_lb_field','db_lb_mapping','ds_list'
                )

class OptionSerializer(serializers.ModelSerializer):
    # for web api: dataset list
    class Meta:
        model = Document
        fields = ('id', 'status', 'local_processed_date', 'accuracy_short','desc','ml_n_gram'
                ,'ml_lib' #,'ml_alg'
                ,'ml_opts','status_code','ml_has_cv'
                )
        
'''

