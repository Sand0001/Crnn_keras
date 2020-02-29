import os
import cv2
import time
import json
import logging
import skimage.io
import numpy as np
from io import BytesIO
from flask_cors import CORS
from flask import Flask, request, make_response
import random
import tensorflow as tf
import keras.backend as K
#from ocr_server import charRec
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
K.set_session(session)

OCR = False
if OCR :
    from ocr import model
else:
    from pse_detect import model

app = Flask(__name__)
CORS(app, resources=r'/*')
tmp_json_path = 'tmp_json_chn/'
fmt='%(asctime)s | [%(process)d:%(threadName)s:%(thread)d] | [%(filename)s:%(funcName)s: %(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(filename='info.log', level=logging.INFO, format=fmt)
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def get_json_path(lan):
    if lan == 'JPE':
        json_label_path = 'label_json_jap/'
    elif lan.upper() == 'CHN':
        json_label_path = 'label_json_chn/'
    elif lan.upper() == 'ENG':
        json_label_path = 'label_json_eng/'
    else:
        json_label_path = 'label_json_chn/'
    return json_label_path

@app.route('/ocr_recognition_test', methods=['POST'])
def ocr_img_pse():
    print('request.files.keys()',list(request.files.keys()))
    lan = request.args.get('language')
    print('test_lan',lan)
    #lan = 'jpe'
    #lan = 'eng'
    #lan = 'chn'
    #print('test_lan',lan)

    logging.info('语言类型%s' % lan)
    if 'json_response' in request.files.to_dict().keys():
        json_response = request.files['json_response']
        print('yes')
        if json_response:
            print('aaaaa')
    elif 'file' in request.files.to_dict().keys():
        try:
            lan = request.form['language']
            print('lan',request.form['language'])
        except:
            lan = request.args.get('language')
            print('test_lan',lan)
        file = request.files['file']
        #llann = request.files['language']
        #print('language',llann)
        if not file:
            logging.info('图片为空')
            return response(json.dumps({'code': -1, 'msg': '文件为空', 'result': ''}))
        logging.info('图片格式：%s' % file.filename)
        try:
            img_buffer = np.asarray(bytearray(file.read()), dtype='uint8')
            bytesio = BytesIO(img_buffer)
            img = skimage.io.imread(bytesio)
        except Exception as e:
            logging.info(str(e), exc_info=True)
            return response(json.dumps({'code': -2, 'msg': '文件格式错误', 'result': ''}))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            pass
        a = time.time()
        try:
            check_label = request.form['checkLabel']
            print('checkLabel', check_label)
        except:
            check_label = 'False'
        #check_label = False
        file_url = 'http://39.104.88.168/image_rec/'
        if check_label == 'False':
            if OCR:
                results,img_shape, after_detect_img_name = model(img, lan,file.filename)
                return response(json.dumps({'code':0, 'msg':'','file_url':file_url,'img_info':results,'after_detect_img_info':after_detect_img_name},cls=MyEncoder))
            else:
                recs, img_shape, after_detect_img_name,rec_name = model(img, lan, file.filename)
                return response(json.dumps({'code': 0, 'msg': '', 'file_url': file_url, 'img_info': recs,
                                            'after_detect_img_info': after_detect_img_name, 'rec_name': rec_name},
                                           cls=MyEncoder))

        else:
            after_detect_img_name = file.filename
            results = []
            txt_file = request.form['txtlabel']
            print(txt_file)
            if txt_file == 'True' :
                check_label_text = open('check_label/label.txt','r').readlines()
                #random.shuffle(check_label_text)
                check_label_text = check_label_text
                for check_label_line in check_label_text:
                    label_img_info = {}
                    label_img_info['img_name'] = check_label_line.split(' ')[0]+'.jpg'
                    label_and_rec_text = {}
                    #label_and_rec_text['label']= check_label_line['label_text']
                    label_and_rec_text['label']= ' '.join(check_label_line.split(' ')[1:]).strip()
                    label_and_rec_text['rec_text'] = '1'
                    label_img_info['text'] = label_and_rec_text
                    results.append(label_img_info)
            else:
                check_json_file = request.form['filename']
                check_label_text = open(check_json_file +'.json','r').readlines()
                # random.shuffle(check_label_text)
                for check_label_line in check_label_text:
                    check_label_line = json.loads(check_label_line)
                    label_img_info = {}
                    label_img_info['img_name'] = check_label_line['img_name']
                    label_and_rec_text = {}
                    label_and_rec_text['label']= check_label_line['text']['label']
                    label_and_rec_text['rec_text']= check_label_line['text']['rec_text']
                    label_img_info['text'] = label_and_rec_text
                    results.append(label_img_info)

            return response(json.dumps({'code':0, 'msg':'','file_url':file_url,'img_info':results,'after_detect_img_info':after_detect_img_name}))
    else:
        label_data = request.data
        print('label_data',label_data)
        if label_data != '':
            label_data = json.loads(label_data)
        aaaa =label_data['text']
        json_label_path = get_json_path(lan)
        print('.'.join(label_data['imgurl'].split('/')[-1].split('.')[:-1]))
        jsonname = json_label_path + '.'.join(label_data['imgurl'].split('/')[-1].split('.')[:-1]) + '.json'
        with open(jsonname,'w') as f:
            json.dump(label_data,f)
        print('保存成功')
        return response(json.dumps({'code':-1,'msg':'保存成功'}))

def response(res):
    rst = make_response(res)
    rst.headers['Access-Control-Allow-Origin'] = '*'
    rst.headers['Access-Control-Allow-Methods'] = 'POST'
    rst.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return rst

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888, debug=True, threaded=True)
