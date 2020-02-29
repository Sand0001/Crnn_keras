# -*- coding:utf-8 -*-
import cv2
import json
import time
from math import *
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = '6, 7'
from char_rec.predict import predict

from psenet.predict import predict as pse
from char_rec.get_part_img import get_part_img
from char_rec.utils import dict_add,is_valid

predict = predict(chn_charset_path ='./char_rec/corpus/chn.txt',
                        eng_charset_path='./char_rec/corpus/eng_new.txt',
                        jap_charset_path='./char_rec/corpus/japeng_1.txt',
                          eng_model_path='./char_rec/models/weights_eng_script_1_129_shufflenet-05-one.h5',
                          chn_model_path='./char_rec/models/weights_chn_script_0212_shufflenet-03-one.h5',
                          # jap_model_path = './char_rec/models/weights_jap_1101_shufflenet_change_lr01-avg1+2+3.h5',
                          jap_model_path='./char_rec/models/weights_jap_sp_0221_big_shufflenet-02-one.h5',
                          chn_res_model_path='./char_rec/models/weights_chn_0925_resnet-05-one.h5')


def sort_box(box):
    """ 
    对box进行排序
    """
    box = sorted(box, key=lambda x: x['image'].shape[1],reverse= True)
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    height = pt3[1] -pt1[1]
    if height > 50 :
        y_offset = 4
    else:
        y_offset = 2
    imgOut = imgRotation[max(1, int(pt1[1]) - y_offset ) : min(ydim , int(pt3[1]+y_offset)), max(1, int(pt1[0]) -4 ) : min(xdim, int(pt3[0])+2)]
    
    return imgOut

def get_image_info(partImg,img_name,r):

    if 'jpg' in img_name:
        picname = img_name.split('.jpg')[0] + '_' + str(time.time()) + '.jpg'
    else:
        picname = img_name.split('.png')[0] + '_' + str(time.time()) + '.jpg'

    cv2.imwrite('/data/fengjing/ocr_recognition_test/html/image_rec/' + picname, partImg[:, :, (2, 1, 0)])
    # cv2.imwrite('/data/fengjing/ocr_recognition_test/chn_tmp/'+picname,partImg[:,:,(2,1,0)])
    image = cv2.cvtColor(partImg, cv2.COLOR_BGR2GRAY)
    pic_info = {}
    pic_info['picname'] = str(picname)
    width, height = image.shape[1], image.shape[0]
    scale = height * 1.0 / 32
    width = int(width / scale)
    image = cv2.resize(image, (width, 32))
    pic_info['location'] = [str(ele) for ele in r]
    pic_info['image'] = np.array(image)
    return pic_info


def get_partImg_add_rotate(img,text_recs,img_name,adjust=False,):
    xDim, yDim = img.shape[1], img.shape[0]
    # scale = 1.0 / scale
    image_info = []
    for index, rec in enumerate(text_recs):
        # print(rec)
        # rec = [r * scale for r in rec]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])
        # imgOut = imgRotation[max(1, int(pt1[1]) - 2 ) : min(ydim , int(pt3[1])), max(1, int(pt1[0]) -2) : min(xdim, int(pt3[0])-2)]
        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度
        # partImg = img[max(1, int(pt1[1]) - 2 ) : min(yDim , int(pt3[1])), max(1, int(pt1[0]) -2) : min(xDim, int(pt3[0])-2)]
        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
    return image_info

def crop_img(box_tmp,img,img_name):
    image_info = []
    h,w = img.shape[:2]
    if len(box_tmp) == 1:
        extend_height = 3
        extend_width = 0
        r = [int(a) for a in box_tmp[0]]
        partImg = img[max(1, r[1] - extend_height):min(h, r[5] + extend_height), max(1, r[0] - extend_width):min(w, r[2] + extend_width)]
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > 3 * partImg.shape[1]:  # 过滤异常图片

            image_info.append(get_image_info(partImg, img_name,r))
    else:
        box_after_sorted_by_y = get_part_img.sort_box_by_position_y(box_tmp)
        for index_y, box_y in enumerate(box_after_sorted_by_y):
            extend_height = get_part_img.extend_pixel(index_y, box_after_sorted_by_y)
            r = [int(a) for a in box_after_sorted_by_y[index_y]]
            partImg = img[max(1, r[1] - extend_height):min(h, r[5] + extend_height), max(1, r[0] - 2):min(w, r[2] + 2)]
            image_info.append(get_image_info(partImg, img_name,r))
    return image_info

def get_image_info_with_pre_post(rec_trans,img,img_name):
    image_info = []
    box_after_sorted = get_part_img.sort_box_by_position_x(rec_trans)
    box_tmp = []
    box_standerd = box_after_sorted[0]
    index_tmp = []
    for index, box in enumerate(box_after_sorted):
        if box_standerd[2] >= box[0] and box_standerd[0] <= box[2]:  # 先大致分列  判断是不是一行
            box_tmp.append(box)
            index_tmp.append(index)
        else:
            image_info += crop_img(box_tmp,img,img_name)
            print('index_tmp', index_tmp)
            box_tmp = []
            index_tmp = []
            box_standerd = box
            box_tmp.append(box)
            index_tmp.append(index)
    if len(box_tmp)>0 :
        image_info += crop_img(box_tmp, img, img_name)
    return image_info



def charRec(lan, img ,img_name,text_recs, adjust=False, scale=None):
    """
    加载OCR模型，进行字符识别
    """

    img_info = []
    erro_record = {'wrong': 0, 'all': 0}
    #image_info = get_image_info(img,text_recs,img_name,adjust=False)
    t1 = time.time()
    #image_info = get_image_info_with_pre_post(text_recs,img,img_name)
    image_info = get_part_img.get_image_info_with_pre_post(text_recs, text_recs, img, picname=img_name)
    image_info = sort_box(image_info)
    print('预处理时间：',str(time.time()-t1))
    print('检测框数量',len(image_info))
    batch_image = []
    batch_image_name = []
    real_num = 0
    if len(image_info) > 0:
        width = image_info[0]['image'].shape[1]
        for index,image1 in enumerate(image_info):
            #print(image)
            pic_name_and_location = {}
            image = image1['image']
            pic_name_and_location['picname'] = image1['picname']
            pic_name_and_location['location'] = image1['location']
            min_width = width - 20
            if image.shape[1] >min_width:
                channel_one = np.pad(image, ((0, 0), (0, width - image.shape[1])), 'constant', constant_values=(255, 255))
                img = np.array(channel_one, 'f') / 255.0 - 0.5

                img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                batch_image.append(img)
                batch_image_name.append(pic_name_and_location)
                if index == len(image_info)-1:
                    batch_results,batch_erro_record = predict.predict_batch_test(np.array(batch_image),batch_image_name,lan)
                    erro_record = dict_add(batch_erro_record, erro_record)
                    img_info +=batch_results
                    real_num += len(batch_image)
            else:
                #batch_fill = True
                batch_results,batch_erro_record = predict.predict_batch_test(np.array(batch_image),batch_image_name,lan)
                erro_record = dict_add(batch_erro_record, erro_record)
                img_info +=batch_results
                real_num += len(batch_image)

                width = image_info[index]['image'].shape[1]
                channel_one = np.pad(image, ((0, 0), (0, width - image.shape[1])), 'constant', constant_values=(0, 0))
                img = np.array(channel_one, 'f') / 255.0 - 0.5
                img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                batch_image = [img]
                batch_image_name = [pic_name_and_location]
                if index == len(image_info)-1:
                    batch_results,batch_erro_record = predict.predict_batch_test(np.array(batch_image),batch_image_name,lan)
                    erro_record = dict_add(batch_erro_record, erro_record)
                    img_info +=batch_results
                    real_num += len(batch_image)

    print('识别的框的数量',real_num)
    print('返回结果的框的数量',len(img_info))

    return img_info if is_valid(erro_record) else []

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
    #cv2.imwrite('/data/fengjing/ocr_recognition_test/chn_tmp/'+ after_detect_img_name, img)
    return after_detect_img_name


def gen_rec_json(img,img_name,boxes):
    json_file_name = '.'.join(img_name.split('.')[:-1])+'.json'
    json_file_path = '/data2/fengjing/pse_test_data/label/'
    label_data = {"path":json_file_path+json_file_name,"width": img.shape[1], "height": img.shape[0], "bndbox":[]}
    bndbox = []
    for box in boxes:
        box_info = {'label':'text'}
        box_info['x1'] = box[1]
        box_info['y1'] = box[0]
        box_info['x2'] = box[3]
        box_info['y2'] = box[2]
        box_info['x3'] = box[5]
        box_info['y3'] = box[4]
        box_info['x4'] = box[7]
        box_info['y4'] = box[6]
        bndbox.append(box_info)
    label_data['bndbox'] = bndbox
    with open(json_file_path+json_file_name,'w') as f:
        json.dump(label_data, f)


def split(data):
    start_i = -1
    end_i = -1
    rowPairs = []
    distance_list = []
    height,width = data.shape[:2]
    num_char = round(height/width) #先大致算有几个数
    min_val = 5  #最小字的高度
    start_cor = False
    for i in range(height):
        if (not data[i].all() )and (start_i < 0):   #判断是否有黑点
            start_i = i
            if start_cor:
                distance_list.append(start_i -start_cor)
                start_cor = False
        elif (not data[i].all()):
            end_i = i
        elif (data[i].all() and start_i >= 0 and end_i >=0 and end_i-start_i>5):
            if (end_i - start_i >= min_val):
                rowPairs.append([start_i, end_i])
                start_cor = end_i
                start_i, end_i = -1, -1
    if end_i-start_i >= min_val and [start_i, end_i] not in rowPairs:   #防止漏掉
        rowPairs.append([start_i, end_i])
    distance_list = sorted(distance_list,reverse=True)  #
    if len(distance_list)+1> num_char:
        distance_list = distance_list[:num_char-1]
    if distance_list:
        min_distance = min(distance_list)
    else:
        min_distance = 0
    new_row_pairs = []
    if len(rowPairs)>num_char:
        new_start = rowPairs[0][0]
        new_end = rowPairs[0][1]
        for i in range(0, len(rowPairs)):
            if rowPairs[i][0] - new_end < min_distance and (
                    rowPairs[i][1] - new_start) / width < 1.3:  # TODO 这个1.3也是需要调整 不安全
                new_end = rowPairs[i][1]
            else:
                new_row_pairs.append([new_start, new_end])
                new_start, new_end = rowPairs[i]
        if rowPairs[-1][1] != new_row_pairs[-1][1]:
            new_row_pairs.append([new_start, new_end])
    else:
        new_row_pairs = rowPairs
    return new_row_pairs

def box_pre(box_list,img):
    new_box_list = []
    for box in box_list:
        width_list = [int(box[i]) for i in range(len(box[:-1])) if i %2 == 0]
        height_list = [int(box[i]) for i in range(len(box[:-1])) if i %2 == 1]
        width = max(width_list)-min(width_list)
        height = max(height_list)-min(height_list)
        if width*1.7 <height:
            img_crop = img[min(height_list):max(height_list),min(width_list):max(width_list)]

            img_crop_binary = cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY)
            # threth = find_binary_threth(img_crop_binary)
            threth = 128
            if threth :
                img_crop_binary[img_crop_binary<threth] = 0
                img_crop_binary[img_crop_binary>threth] = 255
                split_points  = split(img_crop_binary)
                if split_points ==[]:
                    new_box_list.append(box)
                    continue
                for index,point in enumerate(split_points):
                    start = point[0]
                    end = point[1]
                    start_ori = min(height_list)+start
                    end_ori = min(height_list)+end
                    new_box = [box[0],start_ori,box[2],start_ori,box[4],end_ori,box[6],end_ori,box[8]]
                    new_box_list.append(new_box)
        else:
            new_box_list.append(box)
    return new_box_list




def model(img, lan,img_name, adjust=False):
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
    img_pre_rec = img.copy()
    recname = 'after_detect_'+img_name
    text_recs = pse(img,recname, True,True)
    np.save('2.npy',np.array(text_recs))
    text_recs = box_pre(text_recs,img_pre_rec)

    # gen_rec_json(img, img_name, text_recs)
    b = time.time()
    print('pse的耗时：', str(b - a))
    #print(text_recs)
#    print('后面',img_rec.shape)
    #np.save('text_recs',np.array(text_recs))
    after_detect_img_name = draw_box(img_draw,img_name, text_recs)
    b = time.time()
    results = charRec(lan, img_rec,img_name, text_recs, adjust)
    c = time.time()
    print('识别的耗时：', str(c-b))

    return results, (w, h),after_detect_img_name
