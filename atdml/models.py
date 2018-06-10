'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
# -*- coding: utf-8 -*-

from django.db import models
import datetime, pytz, calendar, time
from django.utils.timezone import localtime
from django.conf import settings
from datetime import datetime
from pytz import timezone
import pytz, os
from django.contrib.auth.models import User


# extend the User model
#'''
class User_profile(models.Model):
    user = models.OneToOneField(User)
    acl_list = models.CharField(max_length=200, null=True) 
    # for apk upload 

    count_upload = models.IntegerField(default=0)
    count_upload_max = models.IntegerField(default=6)
    count_upload_period = models.IntegerField(default=24)
    count_upload_date = models.DateTimeField('started date',default=datetime.now)
#'''

# for feature selection page
class Feature_click(models.Model):
	fid=models.IntegerField(null=False) # feature id
	rid=models.IntegerField(null=False)	# row id - id FK to Document
	vote=models.IntegerField(default=0)

class Document(models.Model):
    docfile = models.FileField(upload_to=settings.UPLOAD_DIR)

    filename=models.CharField(max_length=200, null=True) # remove "upload/"
    status = models.CharField(max_length=30, default='new')
    status_code=models.IntegerField(default=0)

    created_date = models.DateTimeField('created date',default=datetime.now)
    processed_date = models.DateTimeField('processed date', null=True)
    submitted_by = models.CharField(max_length=200, null=True)

    # accuracy after train
    accuracy=models.CharField(max_length=20, null=True)
    roc_auc= models.CharField(max_length=20, null=True)
    fscore= models.CharField(max_length=20, null=True)
    perf_measures= models.CharField(max_length=500, null=True)

    # mean for mrun
    mean=models.CharField(max_length=20, null=True)
    # variance for mrun
    variance=models.CharField(max_length=20, null=True)
    # multi run count
    mrun_numb=models.CharField(max_length=20, null=True)

    # for train or predition
    file_type=models.CharField(max_length=50, null=True)
    # 
    total_feature_numb=models.CharField(max_length=30, null=True)
    class_numb= models.CharField(max_length=3, null=True)
    dataset_info= models.CharField(max_length=500, null=True)
    
    # id to link to dataset/model
    train_id=models.CharField(max_length=10, null=True)
    # predict outpout
    prediction=models.CharField(max_length=20, null=True)
    true_label=models.CharField(max_length=20, null=True)

    acl_list=models.CharField(max_length=200, null=True)
    
    # db info for query
    db_host=models.CharField(max_length=100, null=True)
    db_port=models.CharField(max_length=10, null=True)
    db_db=models.CharField(max_length=100, null=True)
    db_tbl=models.CharField(max_length=100, null=True)
    db_proj=models.CharField(max_length=1000, null=True)
    db_filter=models.CharField(max_length=1000, null=True)
    db_lb_field=models.CharField(max_length=100, null=True)
    db_lb_mapping=models.CharField(max_length=500, null=True)
    
    # ML related
    ml_n_gram= models.CharField(max_length=2, null=True)
    ml_lib= models.CharField(max_length=200, null=True)
    ml_opts= models.CharField(max_length=500, null=True)
    ml_has_cv= models.CharField(max_length=10, null=True)
    ml_pca_opts= models.CharField(max_length=100, null=True)
    ml_feat_threshold= models.CharField(max_length=10, null=True)
    ml_model= models.CharField(max_length=1000, null=True)
    ml_state= models.CharField(max_length=1000, null=True)
    ml_feat_opts= models.CharField(max_length=1000, null=True)
    
    # desc for dataset
    desc= models.CharField(max_length=1000, null=True)
    option_state= models.CharField(max_length=20, null=True)
    # pattern for ngram feature extraction
    pattern= models.CharField(max_length=200, null=True)
    # keys to get feature from json
    json_keys_arr= models.CharField(max_length=200, null=True)
    # feature extraction verification
    label_arr= models.CharField(max_length=2000, null=True)
    # evaluation numbers of model
    roc_auc= models.CharField(max_length=20, null=True)
    
    # output of hypothesis/prediction function
    predict_val=models.CharField(max_length=20, null=True)
    feat_list=models.CharField(max_length=1000, null=True)
    
    # ensemble ds id list
    ds_list=models.CharField(max_length=1000, null=True)


    def predict_val_short(self):
        if self.predict_val :
            try:
                pt_indx=self.predict_val.index('.')
                if pt_indx >0:
                    return self.predict_val[0:pt_indx+5]
            except ValueError:
                return self.predict_val
        return self.roc_auc
    def auc_short(self):
        if self.roc_auc :
            try:
                pt_indx=self.roc_auc.index('.')
                if pt_indx >0:
                    return self.roc_auc[0:pt_indx+5]
            except ValueError:
                return self.roc_auc
        return self.roc_auc
    def fscore_short(self):
        if self.fscore :
            try:
                pt_indx=self.fscore.index('.')
                if pt_indx >0:
                    return self.fscore[0:pt_indx+6]
            except ValueError:
                return self.fscore
        return self.fscore
    def accuracy_short(self):
        if self.accuracy :
            try:
                pt_indx=self.accuracy.index('.')
                if pt_indx >0:
                    return self.accuracy[0:pt_indx+3]+"%"
            except ValueError:
                return self.accuracy
        return self.accuracy
    def mean_short(self):
        if self.mean :
            try:
                pt_indx=self.mean.index('.')
                return self.mean[0:pt_indx+3]+"%"
            except ValueError:
                return self.mean
        return self.mean
    def variance_short(self):
        if self.variance :
            try:
                pt_indx=self.variance.index('.')
                return self.variance[0:pt_indx+3]+"%"
            except ValueError:
                return self.variance
        return self.variance

    def local_created_date(self):
        date_format='%Y-%m-%d %H:%M:%S %Z'
        pst_tz=timezone('US/Pacific')
        if self.created_date :
        #return pst_tz.localize(self.created_date).strftime(date_format)
            return self.created_date.astimezone(pst_tz).strftime(date_format)
        return self.created_date
  
    def local_processed_date(self):
        date_format='%Y-%m-%d %H:%M:%S %Z'
        pst_tz=timezone('US/Pacific')
        if self.processed_date :
            #return pst_tz.localize(self.processed_date).strftime(date_format)
            return self.processed_date.astimezone(pst_tz).strftime(date_format)
        return self.processed_date


