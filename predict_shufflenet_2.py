# -*- coding:utf-8 -*-
import os
import re
import sys
import json
import time
import numpy as np
from PIL import Image
from keras import backend as K
from keras.utils import multi_gpu_model
import shutil
import shufflenet_res_crnn as densenet
from char_rec.decode import decode_ctc

decode_ctc = decode_ctc(eng_dict_path_file='./char_rec/corpus/eng_dict.pkl',
                      #lfreq_chn_word_path='./char_rec/corpus/char_and_word_bigram_chneng.json',
                      #lfreq_jap_word_path='./char_rec/corpus/char_and_word_bigram_jap.json')
                      lfreq_chn_word_path='./char_rec/corpus/count_word_chn0.json',
                      lfreq_jap_word_path='./char_rec/corpus/count_word_chn0.json')


os.environ["CUDA_VISIBLE_DEVICES"] = "7,6"
Check_label =True
Decode_debug = False
GPU_NUM = 2
encode_dct = {}
char_set_txt = 'chn.txt'
if char_set_txt == 'chn.txt':
    lan = 'chn'
elif char_set_txt == 'eng.txt':
    lan = 'eng'
else:
    lan = 'jap'
char_set = open(char_set_txt, 'r', encoding='utf-8').readlines()

char_set = [c.strip('\n') for c in char_set]
char_set.append('卍')
nclass = len(char_set)
aa = time.time()
mult_model, basemodel = densenet.get_model(False, 32, nclass)

modelPath = sys.argv[2]
if os.path.exists(modelPath):
    try:
        multi_model = multi_gpu_model(basemodel, gpus=GPU_NUM)
        multi_model.load_weights(modelPath)
    except:
        basemodel.load_weights(modelPath)
model_time = time.time() - aa

def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    #    print (width, height)
    # width = 280
    img = img.resize([width, 32], Image.ANTIALIAS)
    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    X = img.reshape([1, 32, width, 1])
    X = X.swapaxes(1, 2)
    # X = np.vstack((X,X))
    # print("X", X.shape)
    a = time.time()
    # y_pred_1 = basemodel.predict_on_batch(X)
    y_pred_1 = basemodel.predict(X)
    b = time.time() - a
    y_pred = y_pred_1[:, 2:, :]
    y_pred1 = y_pred.copy()
    y_pred_2 = y_pred.copy()
    if Decode_debug:
        out, score = decode_ctc.decode_chn_eng(y_pred[0], lan, char_set)
        out_ori, score_ori = decode_ctc.decode_ori(y_pred1[0], char_set, lan)
        if out_ori != out:
            np.save('npy/'+str(time.time()) + '.jpg')
    else:
        #out,score = decode_ctc.decode_chn_eng(y_pred[0],lan,char_set)
        # b= time.time()
        out,score = decode_ctc.decode_ori(y_pred1[0],char_set,lan)
    out = out.replace(' ','▿')
    out = out.replace('　','▵')
    return out, score, b


def del_blank(word):
    word = list(filter(None, word.strip().split(' ')))
    if len(word) == 0:
        return ''

    c = word[0]
    for i in range(len(word) - 1):
        if len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i][-1])) != 0:
            if len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i + 1][0])) != 0:
                c = c + ' ' + word[i + 1]
            else:
                c = c + word[i + 1]
        else:
            c = c + word[i + 1]
    return c

def is_upper(word):
    for c in word:
        if c <= 'z' and c >= 'a':
            return False
    for c in word:
        if c <= 'Z' and c >= 'A':
            return True
    return False


if __name__ == '__main__':
    
    input_image_path = sys.argv[1]
    if "jpg" in input_image_path or 'png' in input_image_path:
        # img = Image.open(input_image_path).convert('L')
        img = Image.open(input_image_path).convert('L')
        print(predict(img))

    else:
        test_img_list = []
        correct = 0
        num = 0
        upper_correct = 0
        predict_time = 0
        upper_num = 0
        test_label_lines = []
        if Check_label:
            check_label_json = open('check_label.json','w')
        if sys.argv[3] == 'acc':
            test_label_lines = open(sys.argv[4], 'r').readlines()
            #script_label = open('../eng_test_subscript/label_pred.txt', 'a')
            # test_img_list = []
            for line in test_label_lines:
                picName = line.split(' ')[0].strip()
                test_img_list.append(picName)
        pre_time_all = 0
        for jj in range(1):
            for i in os.listdir(input_image_path):
                if "jpg" in i or 'png' in i:
                    img = Image.open(os.path.join(input_image_path, i)).convert('L')
                    if i in test_img_list:
                        text, pred_arr, pred_time = predict(img)
                        predict_time += pred_time
                        label_text = ''.join(test_label_lines[test_img_list.index(i)].split('.jpg ')[1:]).strip()

                        if len(test_label_lines) != 0:
                            del_blank_label_text = del_blank(label_text)
                            del_blank_text = del_blank(text)
                            del_blank_label_text = del_blank_label_text.replace('–', '-')
                            del_rec_text = del_blank_text.replace('–', '-')
                            del_blank_label_text, score = decode_ctc.strQ2B(del_blank_label_text,[1]*len(del_blank_label_text))
                            
                            # if '^' in del_blank_label_text or '~' in del_blank_label_text:
                                #    script_label.writelines(i + '  ' + label_text + '\n')
                                #    shutil.copy(os.path.join(input_image_path, i),'../eng_test_subscript/test/')
                                # continue
                            num += 1
                            if is_upper(label_text):
                                upper_num += 1
                                # print(i)
                                if del_blank_label_text == del_rec_text:
                                    upper_correct += 1
                                else:
                                    print('wrong')
                                    print(del_blank_label_text)
                                    print(del_blank_text)
                            if del_blank_label_text == del_rec_text:
                                correct += 1
                            else:
                                if Check_label:
                                    imagename = {}
                                    label_and_rec_text = {}
                                    label_and_rec_text['label'] = label_text
                                    label_and_rec_text['rec_text'] = text
                                    imagename['text'] = label_and_rec_text
                                    imagename['img_name'] = i
                                    info = json.dumps(imagename)
                                    check_label_json.write(info+'\n')
                                print(i)
                                print(del_blank_label_text)
                                print(del_blank_text)
            if num != 0:
                print('acc:', correct * 1.0 / num, num, upper_num, upper_correct, upper_correct * 1.0 / upper_num)
                print('model 加载时间', model_time)
                print('predict time', predict_time)
                pre_time_all += predict_time
            else:
                print('YOU AREE WRONG!!!')
        print('avgtime', pre_time_all / 1.0)

