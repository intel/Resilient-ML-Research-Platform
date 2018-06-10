'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
# -*- coding: utf-8 -*-
from django.conf.urls import  url
from atdml.views import *

urlpatterns = [
    
    # list/main
    
    url(r'^list/$', list, name='list'),
    url(r'^list/(?P<rid>\d+)/(?P<msg_id>[0-9.]+)/$', list2, name='list_msg'),
    url(r'^list/(?P<rid>\d+)/$', train_opts, name='train_opts'),
    
    # android emulator main
    url(r'^ae_list/$', ae_list, name='ae_list'),
    url(r'^ae_list/(?P<msg_id>[0-9.]+)/$', ae_list_msg, name='ae_list_msg'),
    url(r'^ae_logs/$', ae_logs, name='ae_logs4tbl'),
    url(r'^ae_logs/(?P<cid>\d+)/$', ae_logs, name='ae_logs'),
    url(r'^api/apklist/$', api_get_apk_list, name='get_apk_list'),
    
    # RESTful APIs: ============================================
    # base url, for help only
    url(r'^api/$', list, name='api'),
    # download zip file from Android server to AWS
    url(r'^api/df/$', api_download_exezip, name='api_download_exezip4tbl'),
    url(r'^api/df/(?P<rid>\d+)/$', api_download_exezip, name='api_download_exezip'),
    url(r'^api/df/(?P<type>.+)/(?P<rid>\d+)/$', api_download_file, name='api_download_file'),
    url(r'^api/uf/(?P<type>.+)/(?P<rid>\d+)/$', api_upload_file, name='api_upload_file'),
    
    # result/json/log file
    url(r'^api/f/(?P<rid>\d+)/(?P<fname>.+)/(?P<ln>.+)/$', api_get_result_file, name='api_get_result_file'),
    url(r'^api/jf/(?P<rid>\d+)/(?P<fname>.+)/(?P<ln>.+)/$', api_get_result_jfile, name='api_get_result_jfile'),
    url(r'^api/jf/(?P<rid>\d+)/$', api_get_result_jfile, name='api_get_result_jfile4tbl'),
    url(r'^api/log/(?P<rid>\d+)/(?P<ltype>.+)/(?P<offset>-?\d+)/$', api_get_log_file, name='api_get_log_file'),
    
    # post apk to emulator or get status of emulator or for as_list page
    url(r'^api/apk/$', api_get_post_apk, name='api_post_apk'),
    url(r'^api/apk/(?P<cid>\d+)/$', api_get_post_apk, name='api_get_apk_status'),
    
    # 
    url(r'^api/list/$', api_get_dataset_list, name='get_dataset_list'),
    url(r'^api/optlist/(?P<rid>\d+)/$', api_get_optlist, name='get_optlist'),
    url(r'^api/eslist/$', api_get_eslist, name='api_get_eslist'),
    url(r'^api/model/(?P<rid>\d+)/$', api_get_model, name='api_get_model'),
    url(r'^api/ds/(?P<rid>\d+)/$', api_get_ds_info, name='get_ds_info'),

    # for ensemble
    url(r'^api/ds/$', api_create_ds, name='api_create_ds'),
    # set data
    url(r'^api/set/$', api_set_data, name='api_set_data4ui'),
    url(r'^api/set/(?P<type>[a-zA-Z0-9%_\-.]+)/(?P<rid>\d+)/$', api_set_data, name='api_set_data'),
    url(r'^api/__rm__/(?P<type>.+)/(?P<rid>\d+)/$', api_rm_data, name='api_rm_data'),
    url(r'^api/__rm__/ds/$', api_rm_data, name='rm_data4ds'),
    url(r'^api/__rm__/pred/$', api_rm_data, name='rm_data4pred'),
    
    # feature 
    url(r'^api/feat_extr/$', api_extract_feature, name='api_extract_feature'),
    url(r'^api/feat/(?P<rid>\d+)/$', api_get_feature_impo, name='get_feature_importance'),
    
    # train
    url(r'^api/train/$', api_train, name='api_train'),

    # query/action 
    url(r'^api/qry/$', api_query, name='api_query4script'),
    url(r'^api/qry/(?P<rid>\d+)/(?P<type>[a-zA-Z0-9%_\-.]+)/$', api_query, name='api_query'),
    
    # predict
    url(r'^api/pred/(?P<rid>\d+)/$', api_get_pred, name='get_pred'),
    url(r'^api/pred/$', api_get_pred, name='get_pred2'), #for javascript
    # exec: APK emulation; raw: log text; list/<hash> for IN only
    url(r'^api/pred/(?P<rid>\d+)/(?P<hash>[a-zA-Z0-9%_\-.]+)/$', api_get_post_predict, name='get_post_predict'),
    url(r'^api/predlist/(?P<rid>\d+)/$', api_get_all_predicts, name='get_all_predicts'),
    url(r'^api/predlist/(?P<rid>\d+)/(?P<count>\d+)/$', api_get_top_predicts, name='api_get_top_predicts'),
    
    # END RESTful APIs: ============================================
    # predict  
    url(r'^predict/$', predict, name='predict4oid'),
    url(r'^predict/(?P<rid>\d+)/$', predict, name='predict2'),
    url(r'^(?P<rid>\d+)/predict/$', predict, name='predict'),
    url(r'^(?P<rid>\d+)/predict/(?P<cid>\d+)/(?P<msg_id>[0-9.]+)/$', predict2, name='predict_msg'),

    # result graphs
    url(r'^graph/(?P<rid>\d+)/(?P<gname>[a-zA-Z0-9%_\-.]+)/(?P<tick>\d+)/', get_gdata_tick, name='get_gdata_tick'),
    url(r'^graph/(?P<rid>\d+)/(?P<gname>.+)', get_gdata, name='get_gdata'),

    # train opts/pipeline page
    url(r'^mlearning/opts/$', ml_opts , name='ml_opts'),
    url(r'^list/$', train_opts , name='train4oid'),
	# mlearning EOLed
    url(r'^(?P<rid>\d+)/(?P<filename>.+)/mlearning/$', mlearning , name='mlearning'),

    # multiple run
    url(r'^(?P<rid>\d+)/(?P<filename>.+)/mrun/$', mrun, name='mrun'),
    url(r'^(?P<rid>\d+)/(?P<filename>.+)/mrun/(?P<msg_id>\d+)/$', mrun2, name='mrun_msg'),

    # feature importance
    url(r'^feature_impo/$', feature_impo, name='feature_impo4oid'),
    url(r'^feature_impo/(?P<rid>\d+)/$', feature_impo, name='feature_impo2'),
    url(r'^(?P<rid>\d+)/feature_impo/$', feature_impo, name='feature_impo'),
    url(r'^(?P<rid>\d+)/set_feature/$', set_feature, name='set_feature'),
    url(r'^(?P<rid>\d+)/feature_impo_all/$', feature_impo_all, name='feature_impo_all'),
    url(r'^feature_impo_all/(?P<rid>\d+)/$', feature_impo_all, name='feature_impo_all2'),
    url(r'^feature_impo_combs/$', feature_impo_combs, name='feature_impo4oid'),
    url(r'^feature_impo_combs/(?P<rid>\d+)/$', feature_impo_combs, name='feature_impo_combs'),
    url(r'^exclude_feature/(?P<rid>\d+)/$', exclude_feature, name='exclude_feature'),

    # log page
    url(r'^(?P<rid>\d+)/job_logs/$', job_logs, name='job_logs2'),
    url(r'^job_logs/(?P<rid>\d+)/$', job_logs, name='job_logs'),
    # for predict list
    url(r'^job_logs/(?P<rid>\d+)/(?P<cid>\d+)/$', job_logs_pred, name='job_logs_pred'),
    url(r'^job_logs/$', job_logs, name='job_logs4oid'),
    
    # result page
    url(r'^results/(?P<rid>\d+)/(?P<oid>\d+)/$', result_opts, name='result_opts'),
    # old result
    url(r'^results/(?P<rid>\d+)$', results, name='results'),
    url(r'^results/$', results, name='results4oid'),
    
    #  for ajax echo status
    url(r'^(?P<rid>\d+)/status/$', status, name='status'),
    #  for ajax echo job log
    url(r'^log/(?P<rid>\d+)/(?P<ltype>.+)/(?P<ln>-?\d+)/$', getlog, name='getlog'),
    
    # test only
    url(r'^hello/$', hello, name='hello'),
    url(r'^test/(?P<rid>\d+)/$', get_test, name='get_test'),

    # default page
    url(r'^$', list, name='default'), 
    # not found. go default
    url(r'.+', list, name='default'),
]
