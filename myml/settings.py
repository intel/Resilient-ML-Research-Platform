'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
#import mongoengine

# upload control for prediction
RECAPTCHA_PREDICT="Y"
RECAPTCHA_URL='https://www.google.com/recaptcha/api/siteverify'
PROXY='?'
LIMIT_UPLOAD_PREDICT="Y"
LIMIT_UPLOAD_DATASET="Y"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/media/" 
# used in urls.py
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://media.lawrence.com/media/", "http://example.com/media/"
MEDIA_URL = '/media/'
APP_NAME='atdml'
UPLOAD_DIR='upload'
RESULT_DIR='result'

RESULT_DIR_FULL=MEDIA_ROOT+'/'+RESULT_DIR
UPLOAD_FULL_DIR=MEDIA_ROOT+'/'+UPLOAD_DIR
FEATURE_SRC_DIR=MEDIA_ROOT+'/'+UPLOAD_DIR
TRAIN_DES_DIR=MEDIA_ROOT+'/'+RESULT_DIR
LOG_FOLDER=MEDIA_ROOT+'/log'

EXEC_LOG_FOLDER='/home/django/nfs/archive'
EXEC_RESULT_FOLDER='/home/django/nfs/result'
EXEC_PRED_FOLDER='/home/django/nfs/prediction'

EXEC_LOG_FNAME='*.only.log.xposed'
TMP_DATA_DIR=MEDIA_ROOT+'/tmpdata'
PREPROC_DEC_DIR=TMP_DATA_DIR
# for HDFS
FEATURE_DES_DIR='/user/hadoop/upload'
HDFS_RETR_DIR='/user/hadoop/upload/data_retrieved'

TASK_EXE='/bin/bash'
HDFS_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/hdfs_util.sh'
UPLOAD_HDFS_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/upload_hdfs.sh'
RETRIEVE_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/query_mongo.sh'
FEATURE_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/feature.sh'
PCA_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/pca.sh'
PCA_LOCAL_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/pca_local.sh'
TRAIN_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/train.sh'
TRAIN_SUBMIT_DNN_SCRIP=BASE_DIR+'/'+APP_NAME+'/tasks/submit_dnn.sh'


MRUN_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/multi_run.sh'
PREDICT_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/predict.sh'
PREDICT_MASSIVE_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/predict_massive.sh'
FEATURE_IMPO_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/feature_importance.sh'
SETUP_APK_SCRIPT=BASE_DIR+'/'+APP_NAME+'/tasks/setup_apk.sh'
FEATURE_IMPO_FILTER_COUNT=6
FEATURE_IMPO_LIST_COUNT=50

# Message string
MSG_UPLOAD_SUCCESS="File uploaded successfully" #101
MSG_UPLOAD_FAILED="File upload failed" #90101
MSG_UPLOAD_WARNING="File upload warning"
MSG_UPLOAD_OVER_MAX="File upload count over the maxmium limit" #90902

MSG_ADD_DATASET_SUCCESS="DataSet created successfully"#102
MSG_ADD_OPTION_SUCCESS="New option created successfully"#103

MSG_RETRIEVE_SUCCESS="Data retrieval complete successfully." #223
MSG_RETRIEVE_FAILED="Data retrieval failed!" #90223
MSG_FEATURE_SUCCESS="Feature extraction complete successfully" #221
MSG_FEATURE_FAILED="Feature extraction failed!" #90221
MSG_PCA_SUCCESS="PCA complete successfully" #225
MSG_PCA_FAILED="PCA failed!" #90225

MSG_TRAIN_SUCCESS="Train & predict complete successfully. Please click [<a href='#_if__xidx_'>Result tab</a>] below."  #222
MSG_TRAIN_FAILED="Train & predict failed!"  #90222
MSG_TRAIN_SUBMIT_SUCCESS="Training Job Submitted! Please Check [DNN Status] at Result tab below."  #226
MSG_TRAIN_SUBMIT_FAILED="Training Job Submit failed!"  #90226

MSG_FEATURE_IMPO_SUCCESS="Feature importance calculation completed" #231
MSG_FEATURE_IMPO_FAILED="Feature importance calculation failed" #90231
MSG_FEATURE_SET_SUCCESS="Feature importance update completed" #232
MSG_FEATURE_DROP_SUCCESS="Feature importance drop completed" #233

MSG_MRUN_SUCCESS="Multiple runs complete successfully." #211
MSG_MRUN_DUPLICATED="[Multiple Run Times] was repeated or 'None'!" #90212
MSG_MRUN_FAILED="Multiple runs failed." #90211

MSG_PREDICT_SUCCESS="Prediction completed." #201
MSG_PREDICT_APK_UPLOAD_SUCCESS="APK uploaded and waiting for processing." #205
MSG_PREDICT_FAILED="Prediction failed." #90201
MSG_PREDICT_DUPLICATED="Sample had been predicted already!" #
MSG_RECAPTCHA_FAILED="reCAPTCHA failed! Please complete reCAPTCHA!" #90901

# status code to control action flow
STS_000_NEW=0
STS_100_RETRIEVE=100
STS_200_PREPROCESS=200
STS_300_INIT=300
STS_400_FEATURE=400
STS_500_SRUN=500
#STS_700_=700
STS_800_MRUN=800
STS_900_PREDICT=900
STS_1000_FEATURE_IMPO=1000

# for links in menu bar <<******************
HDFS_STATUS_URL="http://?hdfs dns:50070/explorer.html#/path2/dir"
SPARK_STATUS_URL="http://?spark dns:8080/"
# for input dataset; for web only
MONGO_DNS="?mongo in dns"
MONGO_PORT=27017
MONGO_DB="?tbdsource"
MONGO_TBL="?table"
MONGO_IN_DY_PJ='{"?":1}'
MONGO_IN_DY_FL='{"?":{"$exists":1}}'
MONGO_IN_ST_PJ='{"?":1,"_id":1}'
MONGO_IN_ST_FL='{"?":{"$exists":1}}'
FEATURE_N_GRAM=2

# for output data; for web only
MONGO_OUT_DNS="?mongo outdns"
MONGO_OUT_PORT=27017
MONGO_OUT_DB="myml"
MONGO_OUT_TBL="dataset_info"
MONGO_OUT_USR=""
MONGO_OUT_PWD=""


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/dev/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret! <<******************
SECRET_KEY = '?? key string'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
#DEBUG = False

#TEMPLATE_DEBUG = True

#ALLOWED_HOSTS = []


# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    APP_NAME, 'rest_framework',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    #'django.middleware.security.SecurityMiddleware',
    #'whitenoise.middleware.WhiteNoiseMiddleware',

]
ROOT_URLCONF = 'myml.urls'

WSGI_APPLICATION = 'myml.wsgi.application'


TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates','atdml/templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

# Database
# https://docs.djangoproject.com/en/dev/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Internationalization
# https://docs.djangoproject.com/en/dev/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/dev/howto/static-files/

#STATIC_ROOT=
STATIC_URL = '/static/'
STATICFILES_DIRS= (os.path.join(BASE_DIR, "static"),)
#TEMPLATE_DIRS=(os.path.join(BASE_DIR, "templates"),)

LOGIN_URL = 'myml_login'
LOGOUT_URL = 'myml_logout'
LOGIN_REDIRECT_URL = 'default'

#ALLOWED_HOSTS = ['dns','dns2', 'localhost', '127.0.0.1']
ALLOWED_HOSTS = ['*']

# GOOGLE_RECAPTCHA <<******************
GOOGLE_RECAPTCHA_SECRET_KEY = '?se key'
GOOGLE_RECAPTCHA_SITE_KEY = '?si key'
GOOGLE_RECAPTCHA_JS = 'https://www.google.com/recaptcha/api.js'
