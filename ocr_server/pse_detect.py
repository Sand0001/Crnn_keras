# -*- coding:utf-8 -*-
import time
import cv2

from psenet.predict_val import predict as pse

def draw_box(img,img_name, boxes):
    for box in boxes:
       # box = box['location']
        box = [int(ele) for ele in box]
        cv2.line(img, (box[0], box[1]-2), (box[2], box[3]), (255, 0, 0), 2)
        cv2.line(img, (box[2], box[3]), (box[6], box[7]), (255, 0, 0), 2)
        cv2.line(img, (box[0], box[1]-2), (box[4], box[5]), (255, 0, 0), 2)
        cv2.line(img, (box[4], box[5]), (box[6], box[7]), (255, 0, 0), 2)
    after_detect_img_name = 'after_detect_'+img_name
    cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/'+ after_detect_img_name, img)
    cv2.imwrite('/data/fengjing/ocr_recognition_test/det_pic_box/'+ after_detect_img_name, img)
    #cv2.imwrite('/data/fengjing/ocr_recognition_test/chn_tmp/'+ after_detect_img_name, img)
    return after_detect_img_name

def model(img, lan, img_name, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    results = []
    h, w, _ = img.shape
    a = time.time()
    #    print('原图',img.shape)
    img_draw = img.copy()
    img_rec = img.copy()
    recname = 'after_detect_'+str(time.time()) + img_name
    text_recs = pse(img, recname, True, False)
    after_detect_img_name = draw_box(img_draw, img_name, text_recs)


    return text_recs, (w, h), after_detect_img_name,recname
