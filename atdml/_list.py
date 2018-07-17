'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''
#from django.shortcuts import render_to_response
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from django.core import serializers
from atdml.models import *
from atdml.forms import DocumentForm
import subprocess, datetime, sys
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q
import json, time, os, string, random
import _result
import _feature, _log, _api
#sys.path.append('./ml')
#import ml_util

#============================================================= list ==================
# msg_id: success=101
#         fail=9101
@login_required
def list2(request,rid,msg_id,perm,disabled4reader): # msg_id ============= ======
    #get perms 
    print "in _list.list2(), rid=",rid
    msg_error=""
    msg_success=""
    msg_info=""
    msg_warning=""
    new_id=""

    #msg_id for message after POST and avoid re-POST
    if msg_id=="101":
        msg_success=settings.MSG_UPLOAD_SUCCESS+" Id="+str(rid)
        new_id=rid
    elif msg_id=="102":
        msg_success=settings.MSG_ADD_DATASET_SUCCESS+" Id="+str(rid)
        new_id=rid
    elif msg_id=="221":
        msg_info=settings.MSG_FEATURE_SUCCESS+" Id="+str(rid)
    elif msg_id=="225":
        msg_success=settings.MSG_PCA_SUCCESS+" Id="+str(rid)
    elif msg_id=="222":
        msg_success=settings.MSG_TRAIN_SUCCESS+" Id="+str(rid)
    elif msg_id=="231":
        msg_success=settings.MSG_FEATURE_IMPO_SUCCESS+" Id="+str(rid)
    elif msg_id=="90231":
        msg_error=settings.MSG_FEATURE_IMPO_FAILED+" Id="+str(rid)
    elif msg_id=="90101":
        msg_error=settings.MSG_UPLOAD_FAILED+" Id="+str(rid)
    elif msg_id=="8101":
        msg_warning=settings.MSG_UPLOAD_WARNING+" Id="+str(rid)
    elif msg_id=="90901":
        msg_error=settings.MSG_RECAPTCHA_FAILED
    elif msg_id and "90902" in msg_id:
        arr=msg_id.split('.')
        if len(arr)>1: # append count to the end
            msg_error=settings.MSG_UPLOAD_OVER_MAX+" "+arr[1]
        else:
            msg_error=settings.MSG_UPLOAD_OVER_MAX
    elif msg_id=="0":
        print "for back button"
        # pre-select 
        new_id=rid
        
    # check recaptcha flag
    recaptcha=settings.RECAPTCHA_PREDICT
    if recaptcha is None:
        recaptcha="N"

    # Handle file upload
    if request.method == 'POST':
        action=request.POST.get('action')
        #pre_action=request.POST.get('pre_action')
        print "action=",action
        #print "pre_action=",pre_action

        if action == 'query': # query db
            #if pre_action == 'insert_empty_record': # query db
            print "in query mongodb"
            return insert_empty_record(request,perm,action)
            #else:
            #    return query_db(request,rid,perm)
        elif 'hdfs' in action:
            print "in hdfs action"
            return insert_empty_record(request,perm,action)
        elif 'ensemble' in action:
            print "in ensemble action"
            return insert_empty_record(request,perm,action)
        else: 
            form = DocumentForm(request.POST, request.FILES)
            return uploadFile(request,form,perm)

    else: # not POST ========== ====
        form = DocumentForm() # A empty, unbound form


    # Load documents for the list page
    # filter by type and acl_list <= group & not predict & not train_id>0
    documents = get_ds_doclist(perm)
    
    if (not documents or len(documents)==0) and rid !="0":
        rid="0"  # set to init page
        return HttpResponseRedirect('/atdml/list/'+str(rid)+'/0/')

    # Render list page with the documents and the form
    #return render_to_response(
    return render( request,
        'atdml/list.html',
        {'documents': documents, 'form': form, 'disabled4reader':disabled4reader, 'perm':perm
            , 'msg_error':msg_error, 'msg_success': msg_success, 'msg_info': msg_info, 'msg_warning': msg_warning
            , 'new_id': new_id #, 'options': options
            , "use_recaptcha": recaptcha
        }  
    )
#============================================================= insert_empty_record ==================
def insert_empty_record(request,perm,action):
    print "in insert_empty_record(), action=",action
    newdoc = Document()
    newdoc.submitted_by=request.user.username
    newdoc.acl_list=perm
    newdoc.submitted_by=request.user.username
    newdoc.desc=request.POST.get('hf_desc')

    folder=""
    if action == "query":
        newdoc.filename='q_'+random_string_generator()
        newdoc.file_type=request.POST.get('hf_querytype')
        newdoc.docfile=newdoc.filename
    elif "ensemble" in action: #tbd
        ds_list=request.POST.get("_es_list")
        # for api
        if ds_list is None or len(ds_list)==0:
            ds_list=request.POST.get("ds_list")
        es_name=request.POST.get('_es_name')
        if es_name is None or len(es_name)==0:
            es_name=request.POST.get("es_name")
        uploadtype=request.POST.get("hf_uploadtype")
        newdoc.file_type=uploadtype.lower()

        newdoc.filename=ds_list
        newdoc.filename=es_name
        newdoc.status="new ensemble"
        newdoc.status_code="-101" # avoid pipeline
        newdoc.ds_list=ds_list
    elif "hdfs" in action:
        folder=request.POST.get("hdfs_path")
        uploadtype=request.POST.get("hf_uploadtype")
        newdoc.file_type=uploadtype
        newdoc.filename=folder
        newdoc.docfile=folder
        newdoc.status_code=settings.STS_100_RETRIEVE
        newdoc.status="new hdfs data"
        print "newdoc.file_type=",newdoc.file_type
        print "request.POST.get('label_arr')=",request.POST.get("label_arr")
        
        if "pattern" in uploadtype:
            print "request.POST.get('ptn_str')=",request.POST.get("ptn_str")
            newdoc.status="new hdfs pattern"
            newdoc.pattern=request.POST.get("ptn_str")
        elif "Custom" in uploadtype:
            # custom type
            cust=request.POST.get("hf_customtype")
            print "cust=",cust
            if uploadtype.lower()=="custom" and cust and cust > "":
                newdoc.ml_feat_opts='{"custom":"'+cust.lower()+'"}'
            newdoc.status="new custom"
        else:
            print "request.POST.get('json_keys_arr')=",request.POST.get("json_keys_arr")
            newdoc.status="new hdfs json"
            newdoc.json_keys_arr=request.POST.get("json_keys_arr")

        if len(request.POST.get("label_arr"))>0:
            newdoc.label_arr=request.POST.get("label_arr")

    # no n_gram for static data
    if "static" in newdoc.file_type or "Format" in newdoc.file_type:
        newdoc.ml_n_gram="-1"
    elif not "ensemble" in newdoc.file_type:
        newdoc.ml_n_gram="2" 
        
    newdoc.save()  
    # get new id
    if action == "query":
        sdoc = Document.objects.get(filename=newdoc.filename)
        rid=sdoc.id
        #set filename for query type
        sdoc.filename='q_'+str(rid)
        sdoc.save()
    elif "hdfs" in action:
        rid=newdoc.id
        # do nothing for folder list
        if not ',' in folder:
            create_hdfs_folder(rid, folder)
        # for api adding dataset for hdfs
        if action =="hdfs_api":
            # back to api
            return rid
    elif "ensemble" in action:    
        rid=newdoc.id
        if newdoc.filename is None or len(newdoc.filename)==0:
            newdoc.filename=str(rid)
            newdoc.save()
        if action =="ensemble_api":
            return rid
    
    ret_data ={"id":rid}
    #time.sleep(2)   
    print "insert_empty_record json=",json.dumps(ret_data)
    return HttpResponseRedirect('/atdml/list/'+str(rid)+'/102/')

#============================================================= create_hdfs_folder ==================
def create_hdfs_folder(rid, folder):
    print "in create_hdfs_folder, dir=",settings.HDFS_RETR_DIR+"/"+folder
    ret=subprocess.call([settings.TASK_EXE,      #bash
                settings.HDFS_SCRIPT,      #hdfs_mkdir.sh
                str(rid),
                "-mkdir",
                settings.HDFS_RETR_DIR+"/"+folder,
    ])
    return ret
    
#============================================================= random_string_generator ==================
def random_string_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
#============================================================= query db ==================
def query_db(request,rid,perm):
    print "in query_db()"
    uploadtype=request.POST.get("hf_querytype")
    jstr_filter=request.POST.get("hf_query_filter")
    jstr_proj=request.POST.get("hf_query_proj")
    dns=request.POST.get("_dns")
    port=request.POST.get("_port")
    db=request.POST.get('_db')
    tbl=request.POST.get('_tbl')
    filename='q_'+rid
    
    # call script to query
    print "before call ",settings.RETRIEVE_SCRIPT
    print rid
    print filename
    print uploadtype
    print dns
    print port
    print db
    print tbl
    print jstr_proj
    print jstr_filter
    
    document = Document.objects.get(id=rid)
    #print rid,document 
    document.filename=filename
    document.submitted_by=request.user.username
    document.acl_list=perm
    document.file_type=uploadtype
    document.db_host=dns
    document.db_port=port
    document.db_db=db
    document.db_tbl=tbl
    document.db_filter=jstr_filter
    document.db_proj=jstr_proj
    document.status='processing query'
    #print 'before save'
    document.save()  

    try:
        ret=subprocess.call([settings.TASK_EXE,         #bash
                            settings.RETRIEVE_SCRIPT,   #query_mongo.sh
                            str(rid),
                            filename,
                            uploadtype,
                            "Y",  # flag to download data
                            dns, port, db, tbl,
                            jstr_proj,
                            jstr_filter, 
        ])
    except Exception as e:
        print str(e)

    print "after script"
    ret_msg="ret_msg"
    msg_id="888"
    document = Document.objects.get(id=rid)
    print "doc=",document
    # allow feature button here
    if settings.STS_300_INIT > document.status_code:
        document.status_code="310" # allow feature and log buttons
    document.processed_date=datetime.datetime.now()
    document.status='retrieved'
    document.save()
    
    # get data again to allow document.local_processed_date() to work...
    document = Document.objects.get(id=rid)
  
    ret_data ={"status":document.status, "id":rid, "pdate": document.local_processed_date()
            , "msg": ret_msg+" Id="+str(rid)
            , "msg_id": msg_id, "file_type": document.file_type
            , "status_code": document.status_code
                }
    #time.sleep(2)  
    print "query json=",json.dumps(ret_data)
        
    return HttpResponse(json.dumps(ret_data), content_type="application/json")        

#============================================================= uploadFile ==================
def uploadFile(request,form,perm):
    print "in file upload POST"
    form = DocumentForm(request.POST, request.FILES)
    if form.is_valid():
        uploadtype=request.POST.get("hf_uploadtype")
        desc=request.POST.get('_desc')
        print "uploadtype=",uploadtype
        newdoc = Document(docfile = request.FILES['docfile'])
        newdoc.filename=request.FILES['docfile'] #hardcode to remove "upload/"
        newdoc.submitted_by=request.user.username
        newdoc.acl_list=perm
        newdoc.file_type=uploadtype
        newdoc.desc=desc
        # no n_gram for static data and 
        if "static" in uploadtype or "Libsvm" in uploadtype:
            newdoc.ml_n_gram="-1"
        elif 'custom' in uploadtype.lower():
            newdoc.ml_n_gram="-"
        else:
            newdoc.ml_n_gram="2"
                
        # for featured data ============
        if 'libsvm' in uploadtype.lower():
            #print '++++++++++++ in featured upload'
            newdoc.status='featured'
            newdoc.status_code=settings.STS_400_FEATURE
        elif not 'List' in uploadtype:
            newdoc.status_code=settings.STS_300_INIT # for non-list upload
        
        # db params for different file_type ==========
        if 'List' in uploadtype:
            newdoc.db_host=settings.MONGO_DNS
            newdoc.db_port=settings.MONGO_PORT
            newdoc.db_db=settings.MONGO_DB
            newdoc.db_tbl=settings.MONGO_TBL     
        elif "n-gram pattern gz"==uploadtype.lower():
            newdoc.status_code=settings.STS_100_RETRIEVE
            if "pattern" in uploadtype:
                print "ptn_str=",request.POST.get("ptn_str")
                newdoc.pattern=request.POST.get("ptn_str")
            else:
                print "json_keys_arr=",request.POST.get("json_keys_arr")
                newdoc.json_keys_arr=request.POST.get("json_keys_arr")

            if len(request.POST.get("label_arr"))>0:
                newdoc.label_arr=request.POST.get("label_arr")
        newdoc.save()

        realname=newdoc.docfile.name
        dir_indx=realname.index(settings.UPLOAD_DIR) 

        if dir_indx == 0:
            realname=realname[len(settings.UPLOAD_DIR)+1:]

        print "before Save ========="
        # filename may be different if filename duplicated
        if realname != newdoc.filename:
            newdoc.filename=realname
            newdoc.save()
        print "After Save =========="

        new_id=newdoc.id

		
        if 'libsvm' in uploadtype.lower() or "n-gram pattern gz"==uploadtype.lower() \
            or 'custom' in uploadtype.lower():
            print 'upload to HDFS !! fname=',realname
			# upload to HDFS; call upload_hdfs.sh
            ret=subprocess.call([settings.TASK_EXE,
                                settings.UPLOAD_HDFS_SCRIPT,
                                str(new_id),
                                realname,
                                uploadtype,
            ])
			
        # Redirect to the document list after POST
        return HttpResponseRedirect('/atdml/list/'+str(new_id)+'/101/')
    else:
        print "upload form invalid!"

# response the training opt page ========================================== train_opts ==================
def train_opts(request, rid, perm,disabled4reader):
    print 'in train_opts, rid=', rid,", r_list",reverse('list')
    
    document=get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))
        
    # redirect ensemble to list page
    if "ensemble" in document.file_type:
        return HttpResponseRedirect(reverse('list')+"#"+str(rid))
    
    # set to parent dataset if is an id from option
    #print "document=",document
    if document.train_id > "":
        options =get_opt_docs(document.train_id, perm)
        document =get_ds_doc(document.train_id, perm)
    else:
        # get option list
        options =get_opt_docs(rid, perm)

    jopts=""
    if document:
        jopts=document.ml_opts
    #print "jopts",jopts
    
    has_pca="n"
    if document and not document.ml_pca_opts is None and "threshold" in document.ml_pca_opts:
        has_pca="y"
    print "has_pca=",has_pca #,document.ml_pca_opts
    
    # for custom featuring
    '''
    custom=""
    custom_jstr=document.ml_feat_opts
    if custom_jstr > "":
        try:
            jfeat_opts=json.loads(custom_jstr)
            if jfeat_opts and "custom" in jfeat_opts:
                custom=jfeat_opts["custom"]
        except:
            e = sys.exc_info()[0]
            print ("Exception at train_opts(). e=%s" % e)
    else:
        custom_jstr=""
    '''
    
    if options is None:
        options=[]
    if options:
        print "len of options=",len(options), 
        if options[0]:
            print ",options[0].id=",options[0].id

    return render(request,
        'atdml/train_opts.html',
        {'document': document
            ,'disabled4reader':disabled4reader, 'perm':perm
            ,"jopts":jopts, "has_pca":has_pca
            ,"options":options
        }, 
    ) 
# train opts page's action handler======================================= Learn&Predict ==================
def ml_opts(request,perm,disabled4reader):
    rid=request.POST.get('hf_w_id')
    oid=request.POST.get('hf_w_oid')
    option_state=request.POST.get('hf_w_option_state')
    if option_state:
        option_state=option_state.lower().replace(' ','_')
    print "in _list.ml_opts, rid=",rid,", oid=",oid,",option_state=",option_state
    odoc=None
    ret_data=None
    
    # get option doc if exists
    try:
        document=get_ds_doc(rid, perm)
        if oid and oid > "" and rid != oid:
            #odoc = Document.objects.get(id=oid)
            odoc=get_ds_doc(oid, perm)
            print "odoc.id=" ,odoc.id
        if not document :
            return HttpResponseRedirect(reverse('list'))
    except : 
        e = sys.exc_info()[0]
        print ("Exception at ml_opts(). e=%s" % e)
        return HttpResponseRedirect(reverse('list'))

    http_cde=400
    new_id=-1 
    
    if request.method == 'POST':
        action=request.POST.get('action')
        print "at ml_opts() action=",action
        # add new option entry =======================================
        if action=="new_opt":
            desc=request.POST.get('hf_w_desc')
            # add new option; copy data from dataset
            newdoc = Document()
            newdoc.filename=document.filename
            newdoc.docfile=document.docfile
            newdoc.file_type=document.file_type
            
            newdoc.ml_n_gram=document.ml_n_gram
            newdoc.ml_feat_threshold=document.ml_feat_threshold
            newdoc.ml_lib=document.ml_lib
            newdoc.ml_opts=document.ml_opts
            newdoc.ml_feat_opts=document.ml_feat_opts
            
            newdoc.class_numb=document.class_numb
            newdoc.db_host=document.db_host
            newdoc.db_port=document.db_port
            newdoc.db_db=document.db_db
            newdoc.db_tbl=document.db_tbl
            newdoc.pattern=document.pattern
            
            newdoc.train_id=rid
            newdoc.submitted_by=request.user.username
            newdoc.acl_list=perm
            newdoc.desc=desc
            newdoc.option_state=option_state
            newdoc.status=option_state
            if option_state =="new_training":
                newdoc.status_code="400" # for new training
                newdoc.total_feature_numb=document.total_feature_numb
            else:
                newdoc.status_code="300"  # for new featuring
            newdoc.save()
            new_id=newdoc.id
            print "new doc saved, new_id=",new_id
      
            msg_id="103"
            ret_msg, http_cde=get_msg(msg_id)
            print "http_cde=",http_cde,",ret_msg=",ret_msg 

            # return a new table row
            opt_row=[_api.get_row_4_opt(newdoc)]  
            print "opt_row=",opt_row
            ret_data ={
                 "msg": ret_msg+" for DataSet Id="+str(rid)+", Option Id="+str(new_id), "msg_id":str(msg_id), "new_id": str(new_id), "id":rid
                , "opt_rows": opt_row
            }
            print "end new_opt"
        # feature =======================================
        elif action in ("feature","feature_api"):
            n_gram=request.POST.get('hf_w_ml_n_gram')
            feat_threshold=request.POST.get('hf_w_ml_feat_threshold')
            filter_ratio=request.POST.get('hf_w_ml_filter_ratio')
            if filter_ratio is None or filter_ratio=="":
                filter_ratio="0"
            print "n_gram=",n_gram,",feat_threshold=",feat_threshold,"filter_ratio=",filter_ratio
            
            # custom featuring
            cust=request.POST.get('hf_w_ml_cust')
            cust_params=request.POST.get('hf_w_ml_cust_params')
            print "cust=",cust,",cust_params=",cust_params
            
            ptn=""
            lbl=""
            json_keys_str=""
            if "pattern" in document.file_type:
                ptn=request.POST.get('hf_w_pattern')
                lbl=request.POST.get('hf_w_label_arr')
                print "ptn=",ptn, "lbl=",lbl
            elif "JSON" in document.file_type:
                json_keys_str=request.POST.get('hf_w_json_keys_arr')
                lbl=request.POST.get('hf_w_label_arr')
                print "k_list=",json_keys_str, "lbl=",lbl
            tgt_doc=document
            # for option record
            if odoc:
                tgt_doc=odoc
                option_state=odoc.option_state
                if option_state == "new_featuring":
                    # having featuring output, not depends on source dataset id.
                    ds_id=str(oid)
                # set  option_state to new_featuring 
                if odoc.option_state != 'new_featuring':
                    odoc.option_state='new_featuring'
                    odoc.pattern=ptn
                    odoc.json_keys_arr=json_keys_str
                    odoc.label_arr=lbl
                    odoc.ml_feat_threshold=feat_threshold
                    odoc.ml_feat_opts=cust_params
                    odoc.save()
                    
            # FEATURING here  ===================   
            print "before featuring()..."
            msg_id=featuring(tgt_doc, n_gram, ptn, lbl,json_keys_str, feat_threshold, cust,cust_params,filter_ratio ) 
            # TBD odoc has the upload filename, need to review for query, atd and libsvm type...            
            ret_msg, http_cde=get_msg(msg_id)
            
            if action =="feature_api":
                return rid, msg_id, ret_msg
            
            #print "oid=",oid,"msg_id=",msg_id,", ret_msg=",ret_msg
            # refresh doc
            tgt_doc = Document.objects.get(id=tgt_doc.id)
            ret_data=get_ret_json(tgt_doc, msg_id)
            ret_data["msg"]=ret_msg+" for DataSet Id="+str(rid)+", Option Id="+ str(oid)

            print "feature ret_data=",ret_data

        # PCA =======================================
        elif action=="pca":
            pca_jstr=request.POST.get("hf_w_ml_pca_opts")
            #print "pca_jstr=",pca_jstr
            ds_id=str(rid)
            #print "ds_id=",ds_id  , ",odoc", odoc, ",rid=",rid, ",oid=",oid
            if odoc is None: # for parent dataset
                pca_requester_doc=document
                refresh="1"
            else:# for option entry, reuse parent's pca output
                pca_requester_doc=odoc
                refresh="0"
                
            print "pca_jstr=",pca_jstr,",rid",rid, ",ds_id=" ,ds_id,",refresh=", refresh
            
            #msg_id="225"  #MSG_PCA_SUCCESS=225
            msg_id=pca(pca_requester_doc, pca_jstr, ds_id, refresh) 
            ret_msg, http_cde=get_msg(msg_id)
            #print "ret_msg=",ret_msg, ",http_cde=" ,http_cde
            # refresh doc
            pca_requester_doc = Document.objects.get(id=pca_requester_doc.id)
            # get 
            ret_data=get_ret_json(pca_requester_doc, msg_id)
            print "pca ret_data=",ret_data
            if not odoc is None:
                ret_data["msg"]= ret_msg+" for DataSet Id="+ds_id+", Option Id="+str(oid)
            else:
                ret_data["msg"]= ret_msg+" for DataSet Id="+ds_id
            #print " PCA ret_data=",ret_data
            
        # train / single run=======================================
        elif action in ('mllib_api' ,'mllib', 'scikit', 'scikit_api','dnn'): 
            print "in train action=",action
            opt_jstr=request.POST.get("hf_w_ml_opts")
            ml_has_cv=request.POST.get("hf_w_ml_has_cv") 
            ml_alg=request.POST.get("learning_algorithm").lower() # for classification or clustering
            tgt_doc=document
            ds_id=str(rid)
            option_state=""
            print "in _list.ml_opts.train: opt_jstr=",opt_jstr,",ml_alg=",ml_alg
            #pca_jstr=request.POST.get("hf_w_ml_pca_opts")
             
            # for option
            if odoc:
                tgt_doc=odoc
                option_state=odoc.option_state
                if option_state == "new_featuring":
                    # having net featuring by itself, not depends on source dataset id.
                    ds_id=str(oid)
                    
            # check if exclude feature flag is on, make sure flag is carried forward
            jopts={}
            if not tgt_doc.ml_opts is None and len(tgt_doc.ml_opts)>0:
                jopts=json.loads(tgt_doc.ml_opts)
            if "has_excluded_feat" in jopts:
                if jopts["has_excluded_feat"]==1:
                    # insert this key
                    new_jopts = json.loads(opt_jstr)
                    new_jopts["has_excluded_feat"]=1
                    opt_jstr=json.dumps(new_jopts)
            #print "rid=",tgt_doc.id,",jopts=",jopts
            #print "opt_jstr=",opt_jstr
            
            ml_model=None
            if action=="dnn":
                ml_model=request.POST.get("hf_w_ml_model")
                
            # remove _api for api caller
            tr_action=action.replace("_api","")
            # TRAINING here =================== ==========================================================
            msg_id=training(tgt_doc, tr_action, ml_has_cv, opt_jstr, ds_id,ml_alg,ml_model)
            ret_msg, http_cde=get_msg(msg_id)
            
            # refresh tgt_doc
            tgt_doc=Document.objects.get(id=tgt_doc.id)
            ret_data=get_ret_json(tgt_doc, msg_id)

            if odoc:
                ret_data["msg"]= ret_msg+" for DataSet Id="+ds_id+", Option Id="+str(tgt_doc.id)
            else:
                ret_data["msg"]= ret_msg+" for DataSet Id="+ds_id
            print " train ret_data2=",ret_data, ",http_cde=",http_cde," ,ret_msg=",ret_msg
            
            # for api
            if '_api' in action:
                return rid, msg_id, ret_msg
        # feature_importance =======================================
        elif action in ('feature_importance'): 
            tid=rid
            if rid != oid:
                tid=oid
            print 'In ml_opts.feature_importance'

            # Feature importance here ===================
            # check option_state inside
            msg_id=_feature.calculate_feature_impo(request,tid, perm,disabled4reader)
            ret_msg, http_cde=get_msg(msg_id)
            
            # refresh tgt_doc
            doc=Document.objects.get(id=tid)
            ret_data=get_ret_json(doc, msg_id)
            
            if doc.train_id: 
                ret_data["msg"]= ret_msg+" for DataSet Id="+tid+", Option Id="+str(oid)
            else:
                ret_data["msg"]= ret_msg+" for DataSet Id="+tid
            print " feature importance ret_data=",ret_data
        # goto predict =======================================
        elif action in ('predict'): 
            print 'In ml_opts.predict'
        # multiple_run =======================================
        elif action in ('multiple_run'): 
            print 'In ml_opts.multiple_run:'
            # for option
            tgt_doc=document
            ds_id = rid
            tid=rid
            if not odoc is None:
                tgt_doc=odoc
                tid=oid
                print "odoc.option_state=",odoc.option_state
                # having net featuring by itself, not depends on source dataset id.
                if not odoc.option_state is None and odoc.option_state == "new_featuring":
                    ds_id=str(oid)
            print 'In ml_opts.multiple_run, tid=',tid
           
            # get mrun numb
            mrun_numb=request.POST.get('hf_w_mrun_numb')
            
            # MRUN here =======================
            msg_id=mrun(tgt_doc,mrun_numb,ds_id)
            ret_msg, http_cde=get_msg(str(msg_id))
            
            # refresh tgt_doc
            tgt_doc=Document.objects.get(id=tid)
            ret_data=get_ret_json(tgt_doc, msg_id)
            # add option info
            if tgt_doc.train_id: 
                ret_data["msg"]= ret_msg+" for DataSet Id="+ds_id+", Option Id="+str(oid)
            else:
                ret_data["msg"]= ret_msg+" for DataSet Id="+ds_id
            #print " multiple_run ret_data=",ret_data
        
        if not ret_data:     #??    
            ret_data ={}
            
        # return here
        if request.is_ajax():
            #print "ret_data=",ret_data
            print "http_cde=",http_cde,",ret_msg=",ret_msg 
            return HttpResponse(json.dumps(ret_data), content_type="application/json",status=http_cde)
        # error ?
        return HttpResponse(json.dumps({"msg":"Error","msg_id":-1}), content_type="application/json",status=http_cde)
            
    return HttpResponseRedirect(reverse('list'))
#============================================================= get_ret_json ==================
#@login_required;
# return dataset info in json ( for opt table)
def get_ret_json(doc, msg_id):
    if not doc:
        return {}
    rid=str(doc.id)
    jopts=""
    # get algorithm 
    if doc.ml_opts:
        jopts=json.loads(doc.ml_opts) 
    alg=""
    if jopts and "learning_algorithm" in jopts:
        alg=jopts["learning_algorithm"]

    ret_data={
          "id": rid
        , "msg_id": str(msg_id)
        , "status":doc.status 
        , "status_code":doc.status_code
        , "local_processed_date": doc.local_processed_date()
        , "file_type": doc.file_type
        , "ml_n_gram": str(doc.ml_n_gram) if str(doc.ml_n_gram)!="-1" else "N/A"
        , "ml_feat_threshold": doc.ml_feat_threshold if doc.ml_feat_threshold else "5"
        , "ml_lib": doc.ml_lib if doc.ml_lib else ""
        , "ml_opts":doc.ml_opts if doc.ml_opts else ""
        , "ml_pca_opts":doc.ml_pca_opts if doc.ml_pca_opts else ""
        , "ml_has_cv": doc.ml_has_cv
        , "ml_alg" :alg
        , "ml_model" :doc.ml_model
        , "ml_feat_opts" :doc.ml_feat_opts
        , "class_numb": str(doc.class_numb) 
        , "accuracy_short":str(doc.accuracy_short())
        , "mrun_numb":str(doc.mrun_numb)
        , "desc":doc.desc
        , "pattern":doc.pattern
        , "label_arr":doc.label_arr
        , "json_keys_arr":doc.json_keys_arr
        , "option_state":doc.option_state if doc.option_state else ""
            }
    return ret_data
#============================================================= mrun ==================
def mrun(document,mrun_numb, ds_id):
    print 'In mrun()'
    
    document.status='processing mrun'
    document.mrun_numb=mrun_numb
    document.save()

    uploadtype=document.file_type
    ml_lib=document.ml_lib
    opt_jstr=document.ml_opts
    rid=str(document.id)
    filename=document.filename
    ret=-1
    ret=subprocess.call([settings.TASK_EXE,    #bash
                        settings.MRUN_SCRIPT,  #multi_run.sh
                        rid,
                        filename,
                        mrun_numb,
                        uploadtype,
                        ml_lib,
                        opt_jstr,
                        str(ds_id), 
    ])
    # refresh document
    document = Document.objects.get(id=rid)
    
    if ret==0:
        if settings.STS_800_MRUN > document.status_code:
            document.status='mruned'
            document.status_code=settings.STS_800_MRUN
            print '*** updated document.status_code=', document.status_code
            document.processed_date=datetime.datetime.now()
            document.save()
            
        msg_id="211"
    else:
        msg_id="90211"
    print "after mrun subproc. ret=", ret

    return msg_id
    
#============================================================= featuring ==================
def featuring(document, n_gram, pattern, label_arr,json_keys_str, feat_threshold=5
        , cust=None, cust_params=None, filter_ratio=None):
    #print 'In featuring()'
    
    rid=str(document.id)
    filename=document.filename
    uploadtype=document.file_type
    
    # check if custom 
    jf_opts={}
    if cust.lower() == "predefined":
        cust=""
        cust_params=""
    elif cust and cust>"":
        try: # add cust to opts
            if cust_params and cust_params>"":
                jf_opts=json.loads(cust_params)
            jf_opts["custom"]=cust
            document.ml_feat_opts=json.dumps(jf_opts)
        except:
            pass

        
    # set default if null, default should be set in upload...
    if 'n-gram' in jf_opts:
        n_gram=jf_opts["n-gram"]
    elif not n_gram:
        n_gram=settings.FEATURE_N_GRAM
    elif n_gram=="N/A":
        n_gram="-1"
    #print "n_gram=",n_gram
    
    #update db 
    document.status='processing feature'
    document.processed_date=datetime.datetime.now()
    document.ml_n_gram=n_gram
    document.ml_feat_threshold=feat_threshold
    document.pattern=pattern
    document.label_arr=label_arr
    document.json_keys_arr=json_keys_str
    document.save()
    
    
    print "rid=",rid,",filename=",filename,",uploadtype=",uploadtype,",pattern=",pattern
    print ",n_gram=",n_gram,",filter_ratio=",filter_ratio,",feat_threshold=",feat_threshold,",json_keys_str=",json_keys_str
    
    #execute shell script here
    ret=subprocess.call([settings.TASK_EXE,         #bash
                        settings.FEATURE_SCRIPT,    #feature.sh
                        rid,
                        filename,
                        uploadtype,
                        str(n_gram),
                        pattern,
                        label_arr
                        ,json_keys_str
                        ,str(feat_threshold)
                        ,cust if not cust is None else ""
                        ,cust_params if not cust is None else ""
                        ,str(filter_ratio)
    ])

    # set status
    document = Document.objects.get(id=rid)
    str_ml_opts=document.ml_opts
    ml_opts={}
    if str_ml_opts and len(str_ml_opts)>0:
        ml_opts=json.loads(str_ml_opts)
        

    if ret ==0:
        if settings.STS_400_FEATURE > document.status_code:
            document.status_code=settings.STS_400_FEATURE
        #document.processed_date=datetime.datetime.now()
        document.status='featured'
        msg_id="221" 
        # clean up key "has_excluded_feat"
        if "has_excluded_feat" in  ml_opts:
            ml_opts["has_excluded_feat"]=0
    else:
        document.status='feature failed'
        msg_id="90221" 
    
    #document.status='featured'
    #if settings.STS_400_FEATURE > document.status_code:
    #    document.status_code=settings.STS_400_FEATURE
    document.processed_date=datetime.datetime.now()
    document.ml_opts=json.dumps(ml_opts)
    document.save()
    # feature submitted
    #msg_id="221" 
    print '* end Feature: rc=', ret, '; id=', rid,', fname=', filename 
    return msg_id
 
#============================================================= training ==================
def training(document, action, ml_has_cv, opt_jstr, ds_id, ml_alg, ml_model=None):
    rid=str(document.id)
    hdfs_fname=document.filename
    uploadtype=document.file_type
    pca_opts=document.ml_pca_opts
    print 'In training action=',action," ,rid=",rid,",ml_has_cv=",ml_has_cv,",ml_alg=",ml_alg,",ds_id=",ds_id,",rid=",document.id,"pca_opts=",pca_opts
    hdfs_fname='libsvm_data'
    #check if PCAed
    jopt_jstr=json.loads(opt_jstr)
    has_pca="N"
    jpca_opts=None
    
    # construct hdfs data filename ============= ==========
    # has pca flag on
    if "pca" in jopt_jstr and jopt_jstr["pca"]=="Y":
        # find pca info
        if not pca_opts is None and len(pca_opts)> 0:
            jpca_opts=json.loads(pca_opts)
            if "lib" in jpca_opts:
                if jpca_opts["lib"]=="mllib" and "threshold" in jpca_opts:
                    hdfs_fname=hdfs_fname+'_pca_'+str(jpca_opts["threshold"])+".ml"
                    has_pca="Y"
                elif jpca_opts["lib"]=="scikit" and "threshold" in jpca_opts:
                    hdfs_fname=hdfs_fname+'_pca_'+str(jpca_opts["threshold"])
                    has_pca="Y"

    if action=='mllib':
        #opt_jstr=request.POST.get("hf_w_ml_opts")
        if ml_alg == "kmeans":
            action="mllib_clustering"
            
            
            # clean up data
            document.perf_measures="{}"
            document.fscore=None
            document.roc_auc=None
            document.accuracy=None
            #print "hihi"
            # clean up training_fraction
            jdataset_info=document.dataset_info
            if not jdataset_info is None:
                jdataset_info=json.loads(jdataset_info)
            else:
                jdataset_info={}
            if "training_fraction" in jdataset_info:
                jdataset_info["training_fraction"]=None
            document.dataset_info=json.dumps(jdataset_info)            
        elif ml_has_cv=="yes":
            action="mllib_cv"
        document.ml_lib='mllib'
        #print 'mllib=',opt_jstr

    elif action=='scikit': #scikit-learn
        #opt_jstr=request.POST.get("hf_w_ml_opts")
        #ml_has_cv=request.POST.get("hf_w_ml_has_cv") 
        if ml_alg=="kmeans":
            action="scikit_clustering"
            
            
            # clean up data
            document.perf_measures="{}"
            document.fscore=None
            document.roc_auc=None
            document.accuracy=None
            # clean up dataset_info
            jdataset_info={}
            if not document.dataset_info is None and len(document.dataset_info)>1:
                try:
                    jdataset_info=json.loads(document.dataset_info)
                    if "training_fraction" in jdataset_info:
                        jdataset_info["training_fraction"]=None
                        document.dataset_info=json.dumps(jdataset_info)
                except:
                    pass
        elif ml_has_cv=="yes":
            action="scikit_cv"
        document.ml_lib='scikit'
        #print 'scikit=',opt_jstr
    elif action=='dnn':
        print 'ml_model=',ml_model
        document.ml_model=ml_model
        document.ml_lib='dnn'
        ml_has_cv='no'
        #hdfs_fname=rid+'_dnn_*.gz'
    else:
        document.ml_lib='scikit'
        ml_has_cv='no'
        print 'Warning: action not found:',action
    print 'In training action=',action,",ml_has_cv=",ml_has_cv,",ml_alg=",ml_alg
    
    #update db 
    document.ml_has_cv=ml_has_cv
    document.ml_opts=opt_jstr
    document.status='processing training'
    document.processed_date=datetime.datetime.now()
    document.save()
    #execute shell script here

    if action in ('dnn'):
        #opt_jstr=''
        hdfs_fname_prefix=rid+"_dnn_"
        if rid != ds_id :
            hdfs_fname_prefix=ds_id+"_dnn_"
        print "in submit dnn hdfs_fname_prefix=",hdfs_fname_prefix,",uploadtype=",uploadtype,",action=",action,",opt_jstr=",opt_jstr ,",ds_id=",ds_id 
        ret=subprocess.call([settings.TASK_EXE,         #bash
                            settings.TRAIN_SUBMIT_DNN_SCRIP,      #submit_dnn.sh
                            rid,
                            hdfs_fname_prefix,
                            uploadtype,
                            action, #
                            opt_jstr,
                            ds_id, # flag for train opt
                            ml_model
        ])
        if ret ==0:
            msg_id="226"
        else:
            msg_id="90226"
            document.status='dnn training submitted'
            document.save()
        print '* DNN Train submitted: rc=', ret, '; id=', rid," ,msg_id=",msg_id #,', hdfs_fname=', hdfs_fname 
        return msg_id    
    else:
        ret=subprocess.call([settings.TASK_EXE,         #bash
                            settings.TRAIN_SCRIPT,      #train.sh
                            rid,
                            hdfs_fname,
                            uploadtype,
                            action, # key for mllib/scikit w/o cv
                            opt_jstr,
                            ds_id, # flag for train opt
        ])
    # traing submitted
    print "end train. ret=",ret
    # will be updated in training script
    document = Document.objects.get(id=rid)
    
    if ret ==0:
        if settings.STS_500_SRUN > document.status_code:
            document.status_code=settings.STS_500_SRUN
        #document.processed_date=datetime.datetime.now()
        document.status='learned'
        msg_id="222" 
    else:
        document.status='learn failed'
        msg_id="90222" 
        
    document.save()

    print '* end Train: rc=', ret, '; id=', rid,', hdfs_fname=', hdfs_fname 
    return msg_id

#============================================================= pca ==================
# 1st param: document, should be the PCA requester doc 
def pca(document, pca_jstr, ds_id, refresh="0"):
    print '*** In PCA, pca_jstr=',pca_jstr, ",refresh=", refresh
    
    rid=document.id
    #uploadtype=document.file_type
    opt_state=document.option_state
    
    # for opt_state=new featuring, all PCA related stuff goes to doc itself
    # for opt_state=new training, pca opt string goes to itself, PCA model and output go to parent doc
    if opt_state == "new_featuring":
        ds_id = str(rid)      
        
    #update db 
    document.ml_pca_opts=pca_jstr
    document.status='processing pca'
    document.processed_date=datetime.datetime.now()
    document.save()
    #execute shell script here
    # get k
    pca_k="0"
    pca_opt=json.loads(pca_jstr)
    
    if pca_opt and "threshold" in pca_opt:
        # for filename sufix
        pca_k=pca_opt["threshold"]
    else:
        pca_k="0.9"

    # check if PCA file exists, update ml_pca_opts

    #print "before pca, script=",settings.PCA_SCRIPT,",id=",rid,",pca_k=",pca_k,",pca_jstr=",pca_jstr,",ds_id=",ds_id,",refresh=",refresh
    ret=subprocess.call([settings.TASK_EXE,         #bash
                        settings.PCA_SCRIPT,      #pca.sh
                        str(rid),
                        str(pca_k), # pca_k
                        pca_jstr, # pca opts
                        ds_id, # flag for train opt
                        str(refresh),
    ])
    # traing submitted
    #print "return PCA. ret=",ret
    # will be updated in training script
    document = Document.objects.get(id=rid)
    #print "after doc.get ret=",ret
    
    if ret ==0:
        if settings.STS_400_FEATURE > document.status_code:
            document.status_code=settings.STS_400_FEATURE
        #document.processed_date=datetime.datetime.now()
        document.status='pcaed'
        msg_id="225" # pca
    else:
        document.status='pca failed'
        msg_id="90225"  # TBD
    #print "before doc save"
    
    # can't save?
    document.save()
    

    print '*** end PCA: rc=', ret, '; id=', rid
    return msg_id
     
#============================================================= Learn&Predict ==================
#@login_required;
# EOLed
def learnPredict(request, rid, filename,perm,disabled4reader):

    # get perm
    #uname,grp,perm,disabled4reader=get_perm(request)
    document=get_ds_doc(rid, perm)
    if not document:
        return HttpResponseRedirect(reverse('list'))

    #verify document not predict
    uploadtype=document.file_type
    if uploadtype=='predict':
        return HttpResponseRedirect(reverse('list'))

    form = DocumentForm() # A empty, unbound form

    if request.method == 'POST':
        action=request.POST.get('action')
 
        print '*** mlearning action=', action, ' rid=', rid

        if document and action=='retrieve': #============================== Retrieve data =============
            print 'In retrieve action'
            #update db 
            host=request.POST.get("_w_host")
            port=request.POST.get("_w_port")
            db=request.POST.get("_w_db")
            tbl=request.POST.get("_w_tbl")
            jstr_filter=request.POST.get("_w_filter")
            jstr_proj=request.POST.get("_w_proj")
            usr=request.POST.get("_w_username")
            pwd=request.POST.get("_w_password")
            lb_field=request.POST.get("_w_lb_field")
            lb_mapping=request.POST.get("_w_lb_mapping")
            download_flag=request.POST.get("hf_download_flag")
            src_ds_id=request.POST.get("_w_ds_id")
            
            print "conn str=",host, port, db, tbl, jstr_filter, jstr_proj,"usr=",usr ,"pwd=" 
            print "lb=",lb_field,"lb_mapping=",lb_mapping
            document.status='processing data retrieval'

            document.processed_date=datetime.datetime.now()
            document.db_host=host
            document.db_port=port
            document.db_db=db
            document.db_tbl=tbl
            document.db_proj=jstr_proj
            document.db_filter=jstr_filter
            if "Query" in document.file_type:
                document.db_lb_field=lb_field
                document.db_lb_mapping=lb_mapping
            document.save()
            
            # inherited dataset
            if not src_ds_id is None and len(src_ds_id) > 0:
                src_doc=get_ds_doc(src_ds_id, perm)
                src_fname=src_doc.filename
                print "src_ds_id=",src_ds_id,",src_fname=",src_fname
            else:
                src_fname=""
                src_ds_id=""
                
            #execute shell script here

            #print settings.TASK_EXE, settings.RETRIEVE_SCRIPT
            #print settings.TASK_SRC_DIR+"/"+filename
            ret=subprocess.call([settings.TASK_EXE,         #bash
                                settings.RETRIEVE_SCRIPT,   #query_mongo.sh
                                rid,
                                filename,
                                uploadtype,
                                download_flag,  # flag to download data; may allow to  re-create parquet only
                                host, str(port), db, tbl,
                            jstr_proj,
                            jstr_filter, 
                            usr,
                            pwd,
                            lb_field,lb_mapping
                            , str(src_ds_id)
                            , src_fname
            ])
            
            # will be updated in training script
            document = Document.objects.get(id=rid)
            if ret ==0:
                if settings.STS_500_SRUN > document.status_code:
                    document.status_code=settings.STS_100_RETRIEVE
                #document.processed_date=datetime.datetime.now()
                document.status='retrieved'
                msg_id="223" 
            else:
                document.status_code=-1
                document.status='retrieval failed'
                msg_id="90223"
                
            document.processed_date=datetime.datetime.now()
            document.save()

            #msg_id="223" 
            print '* end Retrieve: rc=', ret, '; id=', rid,', fname=', filename 
        #  ================= Featuring =====  =================   =================  ================= 
        elif document  and action=='feature': #and document.status=='new'
        
            n_gram=request.POST.get("hf_w_ml_n_gram")
            msg_id=featuring(document,n_gram);
        elif document and  action in ('train' ,'mllib', 'scikit'): #============= TRAIN / single run =============
            opt_jstr=request.POST.get("hf_w_ml_opts")
            ml_has_cv=request.POST.get("hf_w_ml_has_cv") 
            
            msg_id=training(document, action, ml_has_cv, opt_jstr, None)          
        # result ===========================================================================
        elif document and action.lower()=='result':
            print 'In result action'
            return _result.result(request, rid, perm,disabled4reader)
        # feature importance result ==================================================================
        elif document and action.lower()=='feat_result_all':
            print 'In feat_result_all action'
            return _feature.feature_impo_all(request,rid, perm,disabled4reader)
        elif document and action.lower()=='feat_result':
            print 'In feat_result action'
            return _feature.feature_impo(request,rid, perm,disabled4reader)
        # log page ==================================================================
        elif document and action.lower()=='viewlog':
            print 'In viewlog action'
            return _log.job_logs(request,rid, perm,disabled4reader)
        # feature importance ===========================================================================
        elif document and action.lower()=='feature_importance':
            print 'In list.feature_importance'
            msg_id=_feature.calculate_feature_impo(request,rid, perm,disabled4reader)

        # Invalid action ===========================================================================
        else:
            print '*** Invalid status or action! id=', rid,', fname=', filename 
        # END POST



    # for viewing result only
    if request.method == 'GET' and document:
        print 'In result GET'
        predictions = Document.objects.all().filter(file_type="predict", train_id=rid).order_by('-id')[0:10]
        # get sample file list
        sflist=_result.get_sfile_list(document.filename, document.id, document.file_type); # how to get dir?
        return render(request,
            'atdml/result.html',
            {'document': document, 'form': form, 'predictions':predictions
                        ,'disabled4reader':disabled4reader, 'perm':perm, 'sflist':sflist},  
            #context_instance=RequestContext(request)
        )
    if request.is_ajax():
        #print "Ajax at mlearning"
        #sdoc = serializers.serializer('json', [document])
        #print "sdoc="+sdoc
        document = Document.objects.get(id=rid)
        ret_msg=""

        ret_msg, sts_cde=get_msg(msg_id)
        
        #print "before mlearning ret_data"
        ret_data ={"status":document.status, "id":rid, "pdate": document.local_processed_date()
            , "msg": ret_msg+" Id="+rid, "msg_id": msg_id, "file_type": document.file_type
            , "status_code":document.status_code
            
            #, "db_host":document.db_host , "db_port":document.db_port , "db_db":document.db_db
            #, "db_tbl":document.db_tbl , "db_proj":document.db_proj , "db_filter": document.db_filter
            , "ml_n_gram":document.ml_n_gram, "ml_lib":document.ml_lib
            , "class_numb": document.class_numb, 
            "ml_opts":document.ml_opts
            
        }
        #time.sleep(2)   
        #print "mlearning json=",json.dumps(ret_data)
        return HttpResponse(json.dumps(ret_data), content_type="application/json",status=sts_cde)
    # default redirect back
    #return HttpResponseRedirect(reverse('atdml.views.list'))
    return HttpResponseRedirect('/atdml/list/'+rid+'/'+msg_id+'/')
 
#============================================================= get_msg ==================
def get_msg(msg_id):
    http_cde=200
    ret_msg="null"
    if msg_id=="101":
        ret_msg=settings.MSG_UPLOAD_SUCCESS
    elif msg_id=="102":
        ret_msg=settings.MSG_ADD_DATASET_SUCCESS
    elif msg_id=="103":
        ret_msg=settings.MSG_ADD_OPTION_SUCCESS
    elif msg_id=="211":
        ret_msg=settings.MSG_MRUN_SUCCESS
    elif msg_id=="221":
        ret_msg=settings.MSG_FEATURE_SUCCESS
    elif msg_id=="222":
        ret_msg=settings.MSG_TRAIN_SUCCESS
    elif msg_id=="223":
        ret_msg=settings.MSG_RETRIEVE_SUCCESS
    elif msg_id=="224":
        ret_msg=settings.MSG_PREPROCESS_SUCCESS
    elif msg_id=="225":
        ret_msg=settings.MSG_PCA_SUCCESS
    elif msg_id=="226":
        ret_msg=settings.MSG_TRAIN_SUBMIT_SUCCESS
    elif msg_id=="231":
        ret_msg=settings.MSG_FEATURE_IMPO_SUCCESS
    elif msg_id=="90223":
        ret_msg=settings.MSG_RETRIEVE_FAILED
        http_cde=400 # error
    elif msg_id=="90221":
        ret_msg=settings.MSG_FEATURE_FAILED
        http_cde=400 # error
    elif msg_id=="90222":
        ret_msg=settings.MSG_TRAIN_FAILED
        http_cde=400 # error
    elif msg_id=="90225":
        ret_msg=settings.MSG_PCA_FAILED
        http_cde=400 # error
    elif msg_id=="90226":
        msg_error=settings.MSG_TRAIN_SUBMIT_FAILED
        http_cde=400 # error
    elif msg_id=="90231":
        ret_msg=settings.MSG_FEATURE_IMPO_FAILED
        http_cde=400 # error
    elif msg_id=="90211":
        msg_error=settings.MSG_MRUN_DUPLICATED
        http_cde=400 # error
    return ret_msg, http_cde

#============================================================= get_ds_doc ==================
# check perm and then get dataset record ============= 
def get_ds_doc(rid, perm):
    print "in get_ds_doc. rid=",rid ,',perm=',perm
    document=None
    try: #acl_list__lte=perm
        if perm=="5": #developer see all records
            document = Document.objects.all().filter(~Q(file_type='predict'),id=rid)
        else: # other see own records
            document = Document.objects.all().filter(~Q(file_type='predict'),acl_list=perm,id=rid)
        if document is None or len(document)==0:
            return None
        document=document[0]
        print "document.id=",document.id
        return document
    except : 
        print "Can't find record. id=", rid

    return None

#============================================================= get_doc ==================
# check perm and then get any record ============= 
def get_doc(rid, perm):
    print "in get_doc. rid=",rid ,',perm=',perm
    document=None
    try: #acl_list__lte=perm
        if perm=="5": #developer see all records
            document = Document.objects.all().filter(id=rid)
        else: # other see own records
            document = Document.objects.all().filter(acl_list=perm,id=rid)
        if document is None or len(document)==0:
            return None
        document=document[0]
        #print "document.id=",document.id
        return document
    except : 
        print "Can't find record. id=", rid

    return None    
    
#============================================================= get_ds_doclist ==================
# for list page's datset list  
def get_ds_doclist(perm):
    print "in get_ds_doclist."," perm=",perm
    documents=None
    try: #acl_list__lte=perm
        if perm=="5": #developer see all records
            #exclude predict and ensemble recordc
            documents = Document.objects.all().filter(~Q(file_type__contains='predict'),train_id__isnull=True).order_by('-id')[0:1000]
        else: # other see own records
            documents = Document.objects.all().filter(~Q(file_type__contains='predict'),train_id__isnull=True,acl_list=perm).order_by('-id')[0:1000]
        if not documents or len(documents)==0:
            return None
        return documents
    except Exception as e:
        print "ERROR: Can't find record. perm=", perm, str(e)

    return None


#============================================================= get_shared_doc ==================
# if acl_list < perm, return rid doc ============= 
# 	allow shared ML model for prediction; based on acl_list=1 as shared record
def get_shared_doc(rid, perm):
    print "in get_shared_doc. rid=",rid ,',perm=',perm
    document=None
    try: 
        if perm=="5": #developer see all records
            document = Document.objects.all().filter(id=rid)
        else: # other see own records
            document = Document.objects.all().filter(acl_list__lte=perm,id=rid)
            #document = Document.objects.all().filter(acl_list="00",id=rid)
        if document is None or len(document)==0:
            return None
        document=document[0]
        #print "document.id=",document.id
        return document
    except : 
        print "Can't find shared record. id=", rid

    return None    

    
#============================================================= get_docs4ensemble ==================
# get doc list for ensemble; allow opt records    
def get_docs4ensemble(perm):
    print "in get_docs4ensemble."," perm=",perm
    documents=None
    try: 
        if perm=="5": #developer see all records
            #exclude predict and ensemble recordc
            documents = Document.objects.all().filter(~Q(file_type__contains='predict')).order_by('-id')[0:1000]
        else: # other see own records
            documents = Document.objects.all().filter(~Q(file_type__contains='predict'),acl_list=perm).order_by('-id')[0:1000]
        if not documents or len(documents)==0:
            return None
        return documents
    except : 
        print "Can't find record. perm=", perm

    return None
    

#============================================================= get_opt_docs ==================
# get opt list for _api    
def get_opt_docs(rid, perm):
    documents=None
    try:
        if perm=="5": #developer see all
            documents = Document.objects.all().filter(~Q(file_type__icontains='predict'),train_id=rid).order_by('-id')[0:500]
        else: # other see own records
            documents = Document.objects.all().filter(~Q(file_type__icontains='predict'),acl_list=perm,train_id=rid).order_by('-id')[0:500]
        if not documents or len(documents)==0:
            return None
        return documents
    except : 
        print "Can't find records. id=", rid
    return None
    
    