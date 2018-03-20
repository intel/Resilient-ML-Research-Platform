#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

# standard library imports
from argparse import ArgumentParser
import sys, ConfigParser, pickle, datetime
import os, glob
import re
import collections, gzip, mimetypes
import ujson, json, math, zipfile
import numpy as np
from scipy.sparse import csr_matrix
import subprocess

from scipy.stats import entropy
#import pydoop.hdfs as hdfs
from time import time


#####matplotlib###############
import matplotlib, math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import matplotlib.cm as cmx

#from scipy.misc import imread, imresize
import tensorflow as tf
from tensorflow.python.platform import gfile
#from PIL import Image
import ml_image_util
sys.path.append('./db')
import exec_sqlite


# may chg to local dir for offline prediction ########################
CONF_FILE='/home/django/myml/app.config' # at the base dir of the web

config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

# device for tensorflow
DEVICE = '/cpu:0'
IMAGENET_DIR='/home/django/myml/media/result/imageNet'
TMODEL_FNAME='tensorflow_inception_graph.pb'
PERT_FNAME='universal.npy'
LABEL_FNAME='labels.txt'

IMG_MDL_INCEPTION="image-inception"
IMG_MDL_YOLO="image-yolo"
   
def main():
    parser = ArgumentParser(description=__description__)
    parser.add_argument("-r", "--row_id", type= str, metavar="row id", help="row_id number in the db", required=False)
    parser.add_argument("-i", "--cid", type=str, metavar="child row id", help="child row id for prediction", required=False)
    parser.add_argument("-dsid", "--ds_id", type=str, metavar="source dataset id", help="source dataset id for training option", required=False)
    
    parser.add_argument("-ifn", "--input_fname", type=str, metavar="input_fname name", help="input filename for prediction", required=False)
    parser.add_argument("-idn", "--input_dirname", type=str, metavar="input_dirname name", help="input dirname for prediction", required=False)
    parser.add_argument("-ofn", "--local_out_fname", type=str, metavar="local output filename", help="out files for prediction", required=False)
    parser.add_argument("-odn", "--local_out_dirname", type=str, metavar="local output dirname", help="out dir for prediction", required=False)
    
    parser.add_argument("-mfn", "--model_fname", type=str, metavar="model filename", help="model filename for prediction", required=False)
    parser.add_argument("-lfn", "--label_fname", type=str, metavar="label filename", help="label filename", required=False)
    parser.add_argument("-upfn", "--univ_pert_fname", type=str, metavar="universal perturbation filename", help="universal perturbation", required=False)
    parser.add_argument("-mty", "--model_type", type=str, metavar="model type", help="model type for prediction", required=False)
    
    parser.add_argument("-fw", "--fromweb", type=str, metavar="flag for web", help="flag for web", required=False)
    parser.add_argument('-fpt','--flag_perturbation', type=str, dest='flag_perturbation', help='flag for image perturbation'
                , default ="Y")
    #### database for output
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

    args = parser.parse_args()
    
    if args.row_id:
        row_id_str = args.row_id
    else:
        row_id_str  = ""
    if args.ds_id:
        ds_id = args.ds_id
    else:
        ds_id  = row_id_str 
    if args.cid:
        cid_str = args.cid
    else:
        cid_str  = '01'
    if args.fromweb:
        fromweb = args.fromweb
    else:
        fromweb  = None  
    if args.input_dirname:
        input_dirname = args.input_dirname
    else:
        input_dirname  = '.'
    if args.input_fname:
        input_fname = args.input_fname
    else:
        input_fname  = 'input.png'
    if args.local_out_dirname:
        local_out_dirname = args.local_out_dirname
    else:
        local_out_dirname  = '.'        
    if args.local_out_fname:
        local_out_fname = args.local_out_fname
    else:
        local_out_fname  = 'predict_out'
    if args.model_type:
        model_type = args.model_type
    else:
        model_type  = "image-inception"        
    if args.model_fname:
        model_fname = args.model_fname
    else:
        if model_type == IMG_MDL_INCEPTION:
            model_fname  = os.path.join(IMAGENET_DIR,TMODEL_FNAME)
        elif model_type == IMG_MDL_YOLO:
            model_fname  = ""
    if args.label_fname:
        label_fname = args.label_fname
    else:
        if model_type == IMG_MDL_INCEPTION:
            label_fname  = os.path.join(IMAGENET_DIR,LABEL_FNAME)
        elif model_type == IMG_MDL_YOLO:
            label_fname  = ""
    if args.univ_pert_fname:
        univ_pert_fname = args.univ_pert_fname
    else:
        univ_pert_fname  = os.path.join(IMAGENET_DIR,PERT_FNAME)        

    ######database########################################
    if len(args.username)>0:
        username = args.username
    else:
        username  = None
    if len(args.password)>0:
        password = args.password
    else:
        password  = None     
        
    return predict(row_id_str, ds_id, cid_str, input_fname, model_fname, univ_pert_fname, label_fname
        , local_out_dirname, local_out_fname, model_type
        , ip_address=args.ip_address, port=args.port, db_name=args.db_name, tb_name=args.tb_name
        , username=username, password=password, flag_perturbation=args.flag_perturbation
        , fromweb=fromweb
        )
        
#   ================================= =======================
def predict(row_id_str, ds_id, cid_str, input_fname, model_fname, univ_pert_fname, label_fname
        , local_out_dirname, local_out_fname, model_type
        , ip_address=config.get('mongo', 'out_ip_address'), port=eval(config.get('mongo', 'out_port'))
        , db_name=config.get('mongo', 'out_db'), tb_name=config.get('mongo', 'out_tb')
        , username=config.get('mongo', 'out_username'), password=config.get('mongo', 'out_password')
        , flag_perturbation="Y", fromweb="1"
        ):
    t0 = time()

    # model from input string ============ Load Model ==============
    #if model_fname is None:
    # default model_fname?
    print "INFO: model_fname=",model_fname
    
    
    if model_type == IMG_MDL_INCEPTION:
        status,str_label_original,predict_val=predict_inception(
        row_id_str, ds_id, cid_str, input_fname, model_fname, univ_pert_fname, label_fname
        , local_out_dirname, local_out_fname, flag_perturbation
        )
    elif model_type == IMG_MDL_YOLO:
        status,str_label_original,predict_val=predict_yolo(
        row_id_str, ds_id, cid_str, input_fname, model_fname, univ_pert_fname, label_fname
        , local_out_dirname, local_out_fname, flag_perturbation
        )

        
    # update sqliteDB ===============
    # only update db for web request
    if fromweb=="1": 
        #print "database update"
        str_sql="UPDATE atdml_document set status = '"+status+"', processed_date ='" \
            +str(datetime.datetime.now())+"', prediction = '"+ str(str_label_original)  \
            +"', predict_val = '"+str(predict_val) \
            +"' where id="+cid_str
        ret=exec_sqlite.exec_sql(str_sql)
        #print "Data update done! ret=", str(ret)
    
    t1 = time()
    print 'INFO: total running time: %f' %(t1-t0)
    return 0  

#   ================================= predict_yolo =======================
def  predict_yolo(row_id_str, ds_id, cid_str, input_fname, model_fname, univ_pert_fname, label_fname
        , local_out_dirname, local_out_fname, flag_perturbation
    ):
    print "INFO: in predict_yolo()"
    str_label_original=None
    status="predicted"
    predict_val=0
    
    # Draw original and perturbed image  ===========
    out_dname=os.path.join(local_out_dirname,cid_str)
    print "INFO: input_fname=",input_fname
    print "INFO: out_dname=",out_dname
    
    # create thumbnail ===========
    ico_fname=os.path.join(local_out_dirname,cid_str+"_ico.png")
    ml_image_util.resize_image2file(input_fname, ico_fname, tgt_size_tuple=(30,30))
    
    # resize image to default 256,256 ===========
    re_fname=os.path.join(local_out_dirname,cid_str+"_re.png")
    print "INFO: resize_fname=",re_fname
    ml_image_util.resize_image2file(input_fname, re_fname, tgt_size_tuple=(256,256))     
    
    # classify origin image here ===========
    cmd = "cd /usr/bin/darknet && ./darknet detect '"+re_fname+"' '"+out_dname+"'"
    print "cmd=",cmd
    ret=os.popen(cmd).read()
    ret_str = re.search('===== Result(.*)===== End Result.', ret)
    if ret_str and ret_str.group and ret_str.group(1) and len(ret_str.group(1).strip())>0:
        str_label_original=ret_str.group(1).strip()
    else:
        str_label_original="not_found"
    print "ret=",ret 
    sys.stdout.flush()    
    
    
    if flag_perturbation=="Y": 
        # create perturbed image  ===========
        pert_fname=os.path.join(local_out_dirname,cid_str+"_p.png")
        y_pert_fname=os.path.join(local_out_dirname,cid_str+"_pert.png")
        print "INFO: pert_fname=",pert_fname
        perturbed_image(re_fname, pert_fname, cid_str, local_out_dirname, img_size=(256, 256), crop_size=(224, 224) 
                ,univ_pert_fname=univ_pert_fname, color_mode="rgb", str_label_perturbed="")
                
        # classify perturbed image here ===========
        cmd = "cd /usr/bin/darknet && ./darknet detect '"+pert_fname+"' '"+out_dname \
            +"_y' && mv -f '"+out_dname+"_y.png' '"+y_pert_fname+"'"
        print "cmd=",cmd
        ret=os.popen(cmd).read()
        print "ret=",ret
    
    return status, str_label_original, predict_val

#   ================================= predict_inception =======================
def perturbed_image(input_fname, pert_fname, cid_str, local_out_dirname, img_size=(256, 256), crop_size=(224, 224) 
        ,univ_pert_fname=None, color_mode="rgb", str_label_perturbed=""):
        
    image_original = ml_image_util.preprocess_image_batch([input_fname], img_size=img_size, crop_size=crop_size \
            , color_mode=color_mode)
            
    pert_v=None
    if not os.path.isfile(univ_pert_fname) and flag_perturbation=="Y":
        print "ERROR: file for perturbation not found!"
        return -2
    else:
        pert_v = np.load(univ_pert_fname)
            
    # Clip the perturbation to make sure images fit in uint8  =================
    clipped_v = np.clip(ml_image_util.undo_image_avg(image_original[0,:,:,:] + pert_v[0,:,:,:]), 0, 255) \
              - np.clip(ml_image_util.undo_image_avg(image_original[0,:,:,:]), 0, 255)
    image_perturbed = image_original + clipped_v[None, :, :, :]
    
    if pert_fname is None:
        pert_fname=os.path.join(local_out_dirname,cid_str+"_pert.png")
        print "INFO: pert_fname=",pert_fname
    ml_image_util.save_image(image_perturbed[0, :, :, :], str_label_perturbed, pert_fname)
    
#   ================================= predict_inception =======================
def  predict_inception(row_id_str, ds_id, cid_str, input_fname, model_fname, univ_pert_fname, label_fname
        , local_out_dirname, local_out_fname, flag_perturbation
    ):
    str_label_original=None
    status="failed"
    predict_val=0
    #print "INFO: Start device..."
    with tf.device(DEVICE):
        persisted_sess = tf.Session()
        
        if not os.path.exists(model_fname):
            print "ERROR: model not found!"
            return -1
        
        # Load the Inception model tensorflow_inception_graph.pb
        with gfile.FastGFile(model_fname, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')    

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")


        #univ_pert_fname = ./universal.npy ==============
        pert_v=None
        if not os.path.isfile(univ_pert_fname) and flag_perturbation=="Y":
            print "ERROR: file for perturbation not found!"
            return -2
        else:
            pert_v = np.load(univ_pert_fname)

        # get labels ================
        labels = open(label_fname, 'r').read().split('\n')

        # resize to 224x224 ==============
        #img_in=Image.open(input_fname)
        #print "INFO: img_in.size=",img_in.size
        
        re_fname=os.path.join(local_out_dirname,cid_str+"_re.png")
        print "INFO: resize_fname=",re_fname
        ml_image_util.resize_image2file(input_fname, re_fname)
        
        # create thumbnail  ==============
        ico_fname=os.path.join(local_out_dirname,cid_str+"_ico.png")
        ml_image_util.resize_image2file(input_fname, ico_fname, tgt_size_tuple=(30,30))

        # func for run() =================
        def f_run(image_inp): 
            return persisted_sess.run(persisted_output \
                                    , feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})

        # predict original image here  =================
        image_original = ml_image_util.preprocess_image_batch([re_fname], img_size=(256, 256), crop_size=(224, 224) \
                , color_mode="rgb")
        orig_out=f_run(image_original)
        #print "orig_out t=",orig_out.shape,orig_out
        # get index of top five predictions
        ind = sorted(range(len(orig_out[0])), key=lambda x: orig_out[0][x])[-5:]
        #print "ind=",ind
        for i in reversed(ind):
            print "++",labels[i-1].strip(), orig_out[0][i]

        #label_original = np.argmax(orig_out, axis=1).flatten()
        #str_label_original = labels[np.int(label_original)-1].split(',')[0].strip()
        str_label_original =labels[ind[-1]-1].split(',')[0].strip()
        predict_val=orig_out[0][ind[-1]]
        print "RESULT: label_original=",str_label_original
        status="predicted"
        
        # generate image
        #im = Image.fromarray(ml_image_util.undo_image_avg( image_original[0, :, :, :]).astype(dtype='uint8'))
        #im.save("./orig.png") 

        # Draw original and perturbed image
        org_fname=os.path.join(local_out_dirname,cid_str+".png")
        print "INFO: org_fname=",org_fname
        ml_image_util.save_image(image_original[0, :, :, :], str_label_original, org_fname)

        if flag_perturbation=="Y":
            # Clip the perturbation to make sure images fit in uint8  =================
            clipped_v = np.clip(ml_image_util.undo_image_avg(image_original[0,:,:,:] + pert_v[0,:,:,:]), 0, 255) \
                      - np.clip(ml_image_util.undo_image_avg(image_original[0,:,:,:]), 0, 255)

            image_perturbed = image_original + clipped_v[None, :, :, :]
            #print "image_original=", image_original

            # generate image
            #im = Image.fromarray(ml_image_util.undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'))
            #im.save("./univ_perturbated_orig.png")        
            
            # predict perturbed image here
            pert_out=f_run(image_perturbed)
            #print "pert_out t=", pert_out.shape, pert_out
            ind = sorted(range(len(pert_out[0])), key=lambda x: pert_out[0][x])[-5:]
            #print "ind=",ind
            #print len(labels)
            for i in reversed(ind):
                print "--",labels[i-1].strip(), pert_out[0][i]

            #label_perturbed = np.argmax(pert_out, axis=1).flatten()
            #str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0].strip()
            #print "RESULT: label_perturbed=",labels[np.int(label_perturbed)-1].split(',')[0]
            str_label_perturbed=labels[ind[-1]-1].split(',')[0].strip()
            print "RESULT: str_label_perturbed=",str_label_perturbed
            
            pert_fname=os.path.join(local_out_dirname,cid_str+"_pert.png")
            print "INFO: pert_fname=",pert_fname
            ml_image_util.save_image(image_perturbed[0, :, :, :], str_label_perturbed, pert_fname)
            
    return status,str_label_original,predict_val
    
'''
python ml/predict_image.py -r 315489 -i 315490 -ifn "/home/django/test_data/Code/image/orig.png" -fw 0 -odn /home/django/myml/media/result/315489
''' 
if __name__ == '__main__':
    __description__ = "utilties for ml"
    main()
