import requests
import json
import sys
import shutil
import os
import random
import numpy as np
from cal_recall.script import cal_recall_precison_f1

def del_file(filePath):
    fileList = os.listdir(filePath)
    for file in fileList:
        os.remove(os.path.join(filePath,file))


def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")


picpath = ['/data2/fengjing/pse_data/text_zixuan3',
'/data2/fengjing/pse_data/text_zixuan4',
'/data2/fengjing/pse_data/text_zixuan',
'/data2/fengjing/pse_data/text_zixuan2',
'/data2/fengjing/pse_data/MTWI_2018/',
           ]
all_pic_path = 'det_pic'

det_path = '/data/fengjing/ocr_recognition_test/det_txt/'
gt_path = '/data/fengjing/ocr_recognition_test/gt_txt/'
del_file(det_path)
del_file(gt_path)

for path in picpath:
    imgPath = os.path.join(path,'image')
    img_list = os.listdir(imgPath)
    picdict = {pic[:-4]: pic for pic in img_list}

    labelPath = os.path.join(path,'label')
    if not os.path.exists(labelPath):
        labelPath = os.path.join(path,'test_label')
    label_list = os.listdir(labelPath)
    num = 0
    index = 1
    for label_file in label_list:
        if label_file.endswith('json'):
            label = json.loads(open(os.path.join(labelPath, label_file), 'r').readlines()[0])
            recs = [[d[key] for key in d if key != 'label'] for d in label['bndbox']]
            recs = np.array(recs).reshape(-1, 8)
            try:
            # if True:
                for rec in recs:

                    validate_clockwise_points(rec)
                name = picdict[label_file[:-5]]
                shutil.copy(os.path.join(imgPath, name), all_pic_path)
                np.savetxt(gt_path + name + '.txt', recs, delimiter=',', fmt='%d')
            except:
                num+=1
                continue

            #try:
            if True:
                img = os.path.join(imgPath,name)
                r = requests.post(url='http://0.0.0.0:8888/ocr_recognition_test?language=chn', files = {'file':open(img,'rb')})
                #r1 = requests.post(url='http://0.0.0.0:/ocr_pse_test?language=eng', files = {'file':open(img,'rb')})
                #print(r.content)
                recs = json.loads(r.content)['img_info']
                recs = np.array([rec[:-1] for rec in recs]).reshape(-1, 8)
                recs = recs[:, (0, 1, 2, 3, 6, 7, 4, 5)]
                np.savetxt(det_path+name+'.txt', recs, delimiter=',', fmt='%d')
                print('完成图片{}个,总共{}个'.format(index,len(img_list)))
                index+=1
            # except  Exception as e :
            #     print(e)
            #     continue


result = cal_recall_precison_f1(gt_path=gt_path, result_path=det_path)
print(result)
print('erro num',num)

