'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponse
from django.http import JsonResponse
from django.core.urlresolvers import reverse
from django.core import serializers
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import login
#rom django.middleware.csrf import CsrfViewMiddleware
import subprocess, datetime, pytz
from datetime import timedelta
import json, time, os,requests
from rest_framework.response import Response
from rest_framework.decorators import api_view
from atdml.models import Document  , User_profile
from atdml.forms import DocumentForm

# page specific
import _result
import _list
import _feature, _log, _api, _predict, _emulator
result_folder=os.path.join(settings.MEDIA_ROOT, 'result')
log_folder=os.path.join(settings.MEDIA_ROOT, 'log')

@api_view(['GET'])
@login_required
#============================================================= api_get_dataset_list==================
def api_get_dataset_list(request):
    print "in api_get_dataset_list, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_dataset_list(request, perm,disabled4reader)

@api_view(['GET'])
@login_required
#============================================================= api_get_dataset_list==================
def api_get_eslist(request):
    print "in api_get_eslist, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_eslist(request, perm,disabled4reader)
    
@api_view(['GET'])
@login_required
#============================================================= api_get_apk_list==================
def api_get_apk_list(request):
    print "in api_get_apk_list, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_apk_list(request, perm,disabled4reader)
    
@api_view(['GET'])
@login_required
#============================================================= api_get_feature_impo==================
def api_get_feature_impo(request, rid):
    print "in api_get_feature_impo, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_feature_impo(request, rid, perm,disabled4reader)

    
@api_view(['GET'])
@login_required
#============================================================= get_top_predicts ==================
def api_get_top_predicts(request,rid, count):
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_all_predicts(request, rid, perm,disabled4reader, count)    
    
@api_view(['GET'])
@login_required
#============================================================= api_get_all_predicts ==================
def api_get_all_predicts(request,rid):
    print "in api_get_all_predicts, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_all_predicts(request, rid, perm,disabled4reader)

@api_view(['GET'])
@login_required
#============================================================= api_get_optlist ==================
def api_get_optlist(request,rid):
    print "in api_get_optlist(), user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_optlist(request, rid, perm,disabled4reader)
    
@api_view(['GET'])
@login_required
#============================================================= api_get_ds_info ==================
def api_get_ds_info(request,rid):
    print "in api_get_ds_info, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_ds_info(request, rid, perm,disabled4reader)
    
@api_view(['GET'])
@login_required
#============================================================= api_get_pred ==================
def api_get_pred(request,rid):
    print "in api_get_pred, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_pred(request, rid, perm,disabled4reader)
    
@api_view(['GET'])
@login_required
#============================================================= api_get_model ==================
def api_get_model(request,rid):
    print "in api_get_optlist, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_model(request, rid, perm,disabled4reader)
    
@api_view(['GET'])
@login_required
#============================================================= api_get_result_file & jfile ==================
def api_get_result_file(request, rid, fname, ln):
    print "in api_get_result_file, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_result_file(rid, fname, ln, perm,disabled4reader,"text")
    
@api_view(['GET'])
@login_required
#============================================================= api_get_result_jfile ==================
def api_get_result_jfile(request, rid, fname, ln):
    print "in api_get_result_jfile, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)

    #special setup to get .json file; assume fname starting with rid
    if fname.index('_')==0: # add rid by default if starting with "_"
        fname=str(rid)+fname

    if "," in ln:
        arr=ln.split(",")
        print "arr=",arr
        ln=arr[0]
        type=arr[1]
        sort_col=None
        if len(arr)==3:
            sort_col=arr[2]
        return _api.get_result_file(rid, fname, ln, perm,disabled4reader, type, sort_col)
    else: # whole file
        return _api.get_result_file(rid, fname, ln, perm,disabled4reader,"json")
        
@api_view(['GET'])
@login_required
#============================================================= api_download_exezip ==================
def api_download_exezip(request, rid):
    print "in api_download_ezip, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    # special type to get zip for apk
    fname="output.zip"
    return _api.download_file(rid, fname, perm, disabled4reader, "apk_zip")

@api_view(['GET'])
@login_required
#============================================================= api_rm_data ==================
def api_rm_data(request, type, rid):
    print "in api_rm_data, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.rm_data(rid, type, perm, disabled4reader)

@api_view(['GET'])
@login_required
#============================================================= api_download_file ==================
# for downloading APK to emulator; hardcode for user django only
def api_download_file(request,type, rid):
    caller=str(request.user)
    print "in api_download_apk, user=", caller
    #uname,grp,perm,disabled4reader=get_perm(request)
    # hardcode for daemon download
    if caller == "django":
        return _api.download_file(rid=rid, fname=None, perm=None, disabled4reader=None, type="__apk__")
        
    return HttpResponse({"error":"file not found!"},status=404)

@api_view(['POST'])
@login_required
#============================================================= api_upload_file ==================
# for uploading result from emulator ; hardcode for user django only
def api_upload_file(request,type, rid):
    caller=str(request.user)
    print "in api_upload_file, user=", caller

    if caller == "django":
        return _api.upload_file(request, rid=rid, fname=None, perm=None, disabled4reader=None, type="__apk__")
    
    #print request.data
    #print request.data["_file_type"]
    #print request.FILES["docfile"]

    #handle_uploaded_file(request.FILES['docfile'])
        
    return HttpResponse({"error":"file not found!"},status=404)

@api_view(['POST','GET'])
@login_required
#============================================================= api_post_apk ==================
def api_get_post_apk(request, cid=None):
    print "in api_post_apk: user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    # check perm
    if perm is None or perm =="":
        return HttpResponse({"error":"no permission"},status=404)
    #return _api.get_post_apk(request,cid, perm,disabled4reader)

    # check max upload count, share with prediction for now
    if request.method == 'POST':
        count= increase_upload_count(request,"pred")
        print "count=",count
        if count<0:
            return Response({"error":"exceed max upload quota "+str(count*-1)+" !"},status=404)
            #return Response({"info":"ok for upload !","count":count})
            
    return _api.get_post_apk(request,cid, perm,disabled4reader)

    
@api_view(['GET'])
@login_required
#============================================================= api_get_log_file ==================
def api_get_log_file(request, rid, ltype, offset):
    print "in api_get_log_file, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    return _api.get_log_file(rid, ltype, offset, perm,disabled4reader)
        
@api_view(['POST'])
@login_required
#============================================================= api_create_ds ==================
def api_create_ds(request):
    print "in api_create_ds, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    
    return _api.create_ds(request, perm,disabled4reader)

@api_view(['POST'])
@login_required
#============================================================= api_extract_feature ==================
def api_extract_feature(request):
    print "in api_extract_feature, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    
    return _api.extract_feature(request, perm, disabled4reader)
    
@api_view(['POST'])
@login_required
#============================================================= api_train ==================
def api_train(request):
    print "in api_train, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    
    return _api.train(request, perm, disabled4reader)

@api_view(['POST'])
@login_required
def api_set_data(request, type, rid):
#============================================================= api_set_data ==================
    print "in api_set_data, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    
    return _api.set_data(request, type, rid, perm, disabled4reader)

@api_view(['POST'])
@login_required
def api_query(request, type, rid):
#============================================================= api_query  ==================
    print "in api_query, user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    
    return _api.api_query(request, type, rid, perm, disabled4reader)

    
#============================================================= TBD graph test only==================
def get_test(request, rid):
    print "in get_test"
    return render(request,
        'atdml/test.html',
        {},
        #context_instance=RequestContext(request)
    )
def get_table(request, rid):
    print "in get_table"
    return render(request,
        'atdml/test.html',
        {},
        #context_instance=RequestContext(request)
    )

#============================================================= Ajax to get graph data ==================
@api_view(['GET'])
@login_required
def get_gdata_tick(request, rid, gname, tick):
    #print "in views.get_gdata_tick, type=",gname,",tick=",tick
    # epoch/error graph
    return get_gdata(request, rid, gname, tick)
    
@api_view(['GET'])
@login_required
#============================================================= Ajax to get graph data ==================
def get_gdata(request, rid, gname, tick=None):
    #print "in views.get_gdata, type=",gname
    # ROC graph
    if gname=='roc':
        try:
            with open(os.path.join(result_folder,str(rid),str(rid)+'_roc.json')) as json_file:
                #print "after open"
                json_data = json.load(json_file)
                #print "json_data=",json_data
                return JsonResponse(json_data, safe=False)
        except Exception as e:
            print "Exception: ",e 
    # DNN graph
    if gname=='dnn':
        #epoch,acc,loss,val_acc,val_loss
        l_values=[]
        vl_values=[]
        a_values=[]
        va_values=[]
        #print type(tick)
        if not tick is None and int(tick)==0:
            tick=None
            #print "tick is 0"
        status=None
        try: 
            document = Document.objects.get(id=rid)
            if document:
                status=document.status
                #print "status=",status
        except : 
            print "Error in getting document"
            pass
        # find log and check epoch status
        epoch_state=""
        epoch_cnt=""
        if "processing" in status:
            log_fname=os.path.join(log_folder,str(rid)+'train.log')
            #print "log_fname=",log_fname
            # find last line
            with open(log_fname, "rb") as f:
                last_ln = f.readline()
                #seek(offset, from 2=end of file)
                f.seek(-2, 2)
                while f.read(1) != b"\r":
                    try: # from current
                        f.seek(-2, 1)
                    except:
                        pass
                last_ln = f.readline()
                # found the state row
                if "ETA:" in last_ln and "loss:" in last_ln:
                    epoch_state=last_ln
                    f.seek(-2, 1)
                    # find prior 2nd \n
                    while f.read(1) != b"\n":
                        f.seek(-2, 1)
                    f.seek(-2, 1)
                    while f.read(1) != b"\n":
                        f.seek(-2, 1)
                    # read the "Epoch n/n" line 
                    epoch_cnt = f.readline()
                    #print "epoch_cnt=",epoch_cnt
                    if "Epoch " in epoch_cnt:
                        epoch_state=epoch_cnt.replace("\n",": ")+epoch_state
        try:
            fname=os.path.join(result_folder,str(rid),str(rid)+'_logger.csv')
            #print "cnn fname=",fname
            with open(fname) as log_file:
                #print "in dnn"
                title=None
                cnt=0
                for line in log_file:
                    #print "line=",line
                    if title is None:
                        title=1 #line.split(',')
                        #print "csv title=", title
                    elif len(line)>0:
                        vals=line.split(',')
                        if len(vals)>2:
                            epoch=int(vals[0])
                            if not tick is None:
                                tick=int(tick)
                                # append item only when epoch > tick
                                if epoch >tick:
                                    #print "epoch=",epoch, ",tick=",tick
                                    l_values.append([epoch,float(vals[2])])
                                    vl_values.append([epoch,float(vals[4])])  
                                    a_values.append([epoch,float(vals[1])])
                                    va_values.append([epoch,float(vals[3])])  
                                    cnt +=1
                            else:
                                l_values.append([epoch,float(vals[2])])
                                vl_values.append([epoch,float(vals[4])])
                                a_values.append([epoch,float(vals[1])])
                                va_values.append([epoch,float(vals[3])])
                                cnt +=1
            #print "data row cnt=",cnt 
            # add status in 1st element to stop polling 
            ret=[{"color": "#FF4000","key":"Training Loss","values":l_values,"status":status,"epoch_state":epoch_state}
                ,{"color": "#0080FF","key":"Validation Loss","values":vl_values}
                ,{"color": "#FF4000","key":"Training Accuracy","values":a_values}
                ,{"color": "#0080FF","key":"Validation Accuracy","values":va_values}
            ]        
                
            return JsonResponse(ret, safe=False)
        except Exception as e:
            print "Exception: ",e 
    # mrun graph        
    elif gname=='mrun': #mrun
        try:
            with open(os.path.join(result_folder,str(rid),str(rid)+'_mrun.json')) as json_file:
                #print "after mrun opened"
                json_data = json.load(json_file)
                #print "json_data=",json_data
                return JsonResponse(json_data, safe=False)
        except Exception as e:
            print "Exception: ",e 
    # prediction distribution graph
    elif gname=='score': #score
        try:
            with open(os.path.join(result_folder,str(rid),str(rid)+'_score_graph.json')) as json_file:
                #print "after score opened"
                json_data = json.load(json_file)
                #print "json_data=",json_data
                return JsonResponse(json_data, safe=False)
        except Exception as e:
            print "Exception: ",e 
    # clusater 3-D graph
    elif gname in ('cluster_3d','cluster_3d_tl'): #mrun
        try:
            with open(os.path.join(result_folder,str(rid),str(rid)+'_'+gname+'.json')) as json_file:
                #print "after cluster opened"
                json_data = json.load(json_file)
                #print "json_data=",json_data
                return JsonResponse(json_data, safe=False)
        except Exception as e:
            print "Exception: ",e 

    ret=None
    
    return JsonResponse(ret, safe=False)
    
    
#============================================================= get_perm ==================
def get_perm(request):

    grp=None
    uname=""
    gname=""
    perm="" 
    disabled4reader=""

    if request.user:
        uname=request.user.username
        #print "Here=", uname
    else: 
        return None, None, None, None

    if uname and uname !="" :
        #print "uname exists"
        try: 
            grp=request.user.groups.get()
            if grp:
                gname=grp.name
                if gname:
                    perm=grp.name.split('-')[0]
            else:
                #gname=""
                # set gname as username
                gname=uname
                perm=uname
                
        except Exception as e:
            print "Exception: ",e 
            gname=uname
            perm=uname
    #'''
    try:
        user_profile = User_profile.objects.get(user_id=request.user.id)
        #print "count_upload=",user_profile.count_upload,",started_date=",user_profile.count_upload_date
    except Exception as e:
        print "Exception: ",e 
        # create profile
        try:
            user_profile= User_profile(user=request.user)
            user_profile.save()
        except Exception as e1:
            print "Exception: ",e1 
    #'''
    # disable tag for reader group
    if perm=="" or perm == "1":
        disabled4reader = "disabled"
    #print uname,", ",gname, ", ",perm,", ",disabled4reader   
    return uname, gname, perm, disabled4reader

#============================================================= about ==================
def about_mlaas(request):
    #get perms 
    uname,grp,perm,disabled4reader=get_perm(request)

    # Render list page with the documents and the form
    return render(request,
        'atdml/about_mlaas.html',
        {  'perm':perm},
        #context_instance=RequestContext(request)
    )
    
def about_ae(request):
    #get perms 
    uname,grp,perm,disabled4reader=get_perm(request)

    # Render list page with the documents and the form
    return render(request,
        'atdml/about_ae.html',
        {  'perm':perm},
        #context_instance=RequestContext(request)
    )
#============================================================= help ==================
def help_mlaas(request):
    #get perms 
    uname,grp,perm,disabled4reader=get_perm(request)

    # Render list page with the documents and the form
    return render(request,
        'atdml/help_mlaas.html',
        {  'perm':perm, "request":request },
        #context_instance=RequestContext(request)
    )

#============================================================= list ==================
# msg_id: success=101
#         fail=9101
@login_required
def list(request): 
    return list2(request,0,0)

@login_required
def list2(request,rid,msg_id): # msg_id ============= ======
    #get perms 
    print 'before _list.list2()'
    uname,grp,perm,disabled4reader=get_perm(request)    
    print 'perm=',perm
    if settings.RECAPTCHA_PREDICT=="Y" and request.method == 'POST':
        #form = DocumentForm(request.POST, request.FILES)
        ok2go=verify_recaptcha(request)
        print "ok2go=",ok2go
        if not ok2go:
            msg_id="90901"
            return HttpResponseRedirect(reverse('list_msg', kwargs={'rid': str(rid),'msg_id':msg_id}))

    # check max upload count, share with prediction for now
    if request.method == 'POST':
        count= increase_upload_count(request,"pred")
        print "count=",count
        if count<0:
            # append count to msg_id
            msg_id="90902."+str(count*-1) 
            # redirect will drop POST data and become a GET
            return HttpResponseRedirect(reverse('list_msg', kwargs={'rid': str(rid),'msg_id':msg_id}))

    return _list.list2(request, rid, msg_id,perm,disabled4reader)
    

#============================================================= APK emulator ==================
@login_required
def ae_list(request): 
    return ae_list2(request, rid=None, cid=None, msg_id=None)
    
@login_required
def ae_list_msg(request, msg_id): 
    return ae_list2(request, rid=None, cid=None, msg_id=msg_id)

@login_required
def ae_list2(request, rid, cid, msg_id): 
    print "in ae_list2(), rid=",rid,",msg_id=",msg_id
    uname,grp,perm,disabled4reader=get_perm(request)
            
    # check verify_recaptcha only for POST
    if settings.RECAPTCHA_PREDICT=="Y" and request.method == 'POST':
        #form = DocumentForm(request.POST, request.FILES)
        ok2go=verify_recaptcha(request)
        #print "ok2go=",ok2go
        # recaptcha failed:
        if not ok2go:
            msg_id="90901"
            return HttpResponseRedirect(reverse('ae_list_msg', kwargs={'msg_id':msg_id}))
    # check max upload count
    if request.method == 'POST':
        count= increase_upload_count(request,"pred")
        print "count=",count
        if count<0:
            # append count to msg_id
            msg_id="90902."+str(count*-1) 
            #print "URL============",reverse('ae_list_msg', kwargs={'msg_id':msg_id}),"msg_id=",msg_id
            # redirect will drop POST data and become a GET
            return HttpResponseRedirect(reverse('ae_list_msg', kwargs={'msg_id':msg_id}))

    # continue with prediction
    return _emulator.emulate(request, rid, cid, msg_id, perm, disabled4reader)

    
#============================================================= list.train_opts ==================
@login_required
def train_opts(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _list.train_opts(request, rid, perm,disabled4reader)

#============================================================= list.ml_opts ==================
@login_required
def ml_opts(request):
    print 'at ml_opts()'
    uname,grp,perm,disabled4reader=get_perm(request)
    return _list.ml_opts(request,perm,disabled4reader)
    
#============================================================= list.learnPredict ==================
@login_required
def mlearning(request, rid, filename):
    # EOLed; get perm
    print 'before _list.learnPredict()'
    uname,grp,perm,disabled4reader=get_perm(request)
    return _list.learnPredict(request, rid, filename,perm,disabled4reader)
 
#============================================================= result.mrun ==================
# msg_id: success=211
#         fail=9211
@login_required
def mrun(request, rid, filename):
    return  mrun2(request, rid, filename, 0)

@login_required
def mrun2(request, rid, filename, msg_id):
    print "in mrun2"
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    # get sample file list

    return _result.mrun2(request, rid, filename, msg_id,perm,disabled4reader)

    
@api_view(['GET','POST'])
@login_required
#@authentication_classes((SessionAuthentication, BasicAuthentication))
#@permission_classes((IsAuthenticated,))
#============================================================= for api predict==================
def api_get_post_predict(request, rid, hash):
    print "in api_get_post_predict: user=", request.user
    uname,grp,perm,disabled4reader=get_perm(request)
    count=0
    if request.method == 'POST':
        count= increase_upload_count(request,"pred")
    if count<0:
        return Response({"error":"exceed max upload quota "+str(count*-1)+" !"},status=404)
    else:
        return _api.get_post_predict(request, rid, hash, perm,disabled4reader)

# allow max upload within the time period= started_date + upload_period
#============================================================= increase_upload_count==================
# check limit/add upload count
def increase_upload_count(request, type="pred"):
    if not settings.LIMIT_UPLOAD_PREDICT == "Y":
        # allowed
        return 0
    #TBD if type=="ds" and not settings.LIMIT_UPLOAD_DATASET == "Y":
    #    # allowed
    #    return 0

    user_profile = User_profile.objects.get(user=request.user)
    
    # current upload count
    cnt=user_profile.count_upload
    max_cnt=user_profile.count_upload_max
    # in hour, time window for max upload
    upload_period=user_profile.count_upload_period
    # start time for time window
    started_date=user_profile.count_upload_date
    
    # get timezone
    tz=started_date.tzinfo
    # current datetime
    now=datetime.datetime.now(tz)
    # end time for time window
    reset_dtime=started_date+ timedelta(hours=upload_period)
    print "rt=",reset_dtime,"st=",started_date,"now=",now
    
    # if started_date + upload_period =< now, reset count
    if now >= reset_dtime:
        user_profile.count_upload=1
        user_profile.count_upload_date=now
        cnt=0
    elif reset_dtime > now:
        if cnt < max_cnt :
            user_profile.count_upload=user_profile.count_upload+1
        # else return error
        else:
            # -1 is the flag of hitting max upload
            return cnt*-1

    user_profile.save()
    return cnt+1
    
#============================================================= result.predict ==================
# msg_id: success=201
#         fail=9201
@login_required
def predict(request, rid):
    return predict2(request, rid, 0,0)

@login_required
def predict2(request, rid, cid, msg_id):
    print 'before _result.predict2()'
    # check if upload action
    action_type=request.POST.get('_action_type')
    if settings.RECAPTCHA_PREDICT=="Y" and action_type == 'upload_predict':
        #form = DocumentForm(request.POST, request.FILES)
        ok2go=verify_recaptcha(request)
        #print "ok2go=",ok2go
        # recaptcha failed:
        if not ok2go:
            msg_id="90901"
            #print reverse('predict_msg', kwargs={'rid': str(rid),'cid':str(cid),'msg_id':msg_id})
            return HttpResponseRedirect(reverse('predict_msg', kwargs={'rid': str(rid),'cid':str(cid),'msg_id':msg_id}))

    # check max upload count
    if request.method == 'POST':
        count= increase_upload_count(request,"pred")
        print "count=",count
        if count<0:
            # append count to msg_id
            msg_id="90902."+str(count*-1) 
            #print "URL============",reverse('ae_list_msg', kwargs={'msg_id':msg_id}),"msg_id=",msg_id
            # redirect will drop POST data and become a GET
            return HttpResponseRedirect(reverse('predict_msg', kwargs={'rid': str(rid),'cid':str(cid),'msg_id':msg_id}))

        # continue with prediction
    uname,grp,perm,disabled4reader=get_perm(request)
    return _predict.predict(request, rid, cid, msg_id,perm,disabled4reader)


   
    
#============================================================= status ==================
@login_required
def status(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #print rid
    try: 
        document = Document.objects.get(id=rid)
    except : 
        document = None
    sts="n/a"
    if document:
        sts=document.status
    #return HttpResponse(sts) 
    
    ret ={"status":sts, "id":rid}
    #time.sleep(2)   
    return HttpResponse(json.dumps(ret), content_type="application/json")        

#============================================================= feature_impo ==================
# msg_id: success=224
#         fail=
# get feat impo data only
@login_required
def feature_impo(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    return _feature.feature_impo(request, rid, perm,disabled4reader)

# get feat impo data only
@login_required
def feature_impo_all(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    return _feature.feature_impo_all(request, rid, perm,disabled4reader)
    
@login_required
def feature_impo_combs(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    return _feature.feature_impo_combs(request, rid, perm,disabled4reader)

#============================================================= set_feature ==================
#         
@login_required
def set_feature(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _feature.set_feature(request, rid, perm,disabled4reader)

#============================================================= exclude_feature ==================
#         
@login_required
def exclude_feature(request,rid):
    print "in views.exclude_feature()"
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _feature.exclude_feature(request, rid, perm,disabled4reader)
    
    
#============================================================= joblog for pred ========
#         
@login_required
def job_logs_pred(request,rid,cid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _log.job_logs(request, rid, perm,disabled4reader, cid)

#============================================================= job_logs ==================
#         
@login_required
def job_logs(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _log.job_logs(request, rid, perm,disabled4reader)

#============================================================= ae_logs ==================
#         
@login_required
def ae_logs(request,cid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _log.ae_logs(request, cid, perm,disabled4reader)
    
#============================================================= results ==================
#         
@login_required
def results(request,rid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _result.result(request, rid, perm,disabled4reader)
    
@login_required
def result_opts(request,rid,oid):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #return _list.setfeature(request, rid, perm,disabled4reader)
    return _result.result_opts(request, rid, oid, perm,disabled4reader)

    
#============================================================= getlog ==================
@login_required
def getlog(request,rid,ltype,ln):
    # get perm
    uname,grp,perm,disabled4reader=get_perm(request)
    #print "++++++++++++++ in getlog,dis=",disabled4reader
    return _log.getlog(rid,ltype,ln, perm,disabled4reader)

 
#============================================================= msg in login ==================
def login_msg(request, msg_id, template_name):
    msg_error=""
    if msg_id=="90901":
        msg_error=settings.MSG_RECAPTCHA_FAILED
    if settings.RECAPTCHA_PREDICT=="Y":
        use_recaptcha='Y'
    else:
        use_recaptcha='N'
    # render msg
    return render(request, template_name, {'msg_error':msg_error,'use_recaptcha':use_recaptcha} )

#============================================================= custom login RECAPTCHA ==================
def login_recaptcha(request, template_name):
    print "in login_recaptcha ==================="
    if request.method == 'POST':
        if settings.RECAPTCHA_PREDICT=="Y":
            ok2go=verify_recaptcha(request)
            #print "ok2go=",ok2go
            if not ok2go:
                #print "Not ok"
                msg_id="90901"
                return HttpResponseRedirect(reverse('login_msg', kwargs={'msg_id':msg_id,'template_name':template_name}))
    
            #print "pass ok"
            # ok for login() at django.contrib.auth.views
            return login(request, template_name)
        else:
            return login(request, template_name)

    # redirect for msg rendering
    return login_msg(request, msg_id="", template_name=template_name)

#============================================================= verify_recaptcha ==================
#@login_required
def verify_recaptcha(request,form=None):
    # get reCAPTCHA input and query google
    #print "req=", request.method # TBD form invalid?, ", form valid=",form.is_valid()
    
    if request and request.method == 'POST': #and form  and form.is_valid():
        ''' Begin reCAPTCHA validation '''
        # get recaptcha_response from POST
        recaptcha_response = request.POST.get('g-recaptcha-response')
        if recaptcha_response is None or len(recaptcha_response)==0:
            return False
        print "recaptcha_response len=",len(recaptcha_response)
        # data to verify
        data = {
            'secret': settings.GOOGLE_RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
        #print "before post to google"
        proxyDict=None
        if len(settings.PROXY)>0:
            proxyDict = { "https": settings.PROXY}
        ret = requests.post(settings.RECAPTCHA_URL, data=data, proxies=proxyDict, verify=False)
        print "recapcha r=", ret
        if ret is None:
            return False
        result={}
        try:
            result = ret.json()
        except Exception as e:
            print "Exception: ",e
            return False
        ''' End reCAPTCHA validation '''
        if 'success' in result and result['success']:
            return True
        else:
            return False
    # is not POST:
    return True
 
    
#============================================================= test only ==================
def hello(request):
    return HttpResponse("Hello World")        


