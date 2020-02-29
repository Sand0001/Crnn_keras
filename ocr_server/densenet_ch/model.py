#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps,ImageDraw,ImageFont
import cv2
from keras.layers import Input
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model
import re
from . import keys
import itertools
#from . import densenet
#from . import crnn as densenet
from densenet_ch.eng_dict import eng_dict
import pickle
import time
import json
import logging
from zhon.hanzi import punctuation
#reload(densenet)

LAN = 'chn'
# LAN = 'jap'
MODEL = 'resnet'
#MODEL = 'crnn'
import tensorflow as tf
graph = tf.get_default_graph()
from . import shufflenet_res_crnn as densenet
rec_json_file = open('/data2/fengjing/rec_json_big_chn_line.json','a')
encode_dct =  {}
pkl_file = open('./densenet_eng/eng_dict.pkl', 'rb')
word_dict = pickle.load(pkl_file)
lfreq = json.loads(open('./densenet_ch/count_big_chn','r').readlines()[0])
lfreq_word = json.loads(open('./densenet_ch/count_word_chn0.json','r').readlines()[0])
rec_pic_path = '/data/fengjing/ocr_recognition_test/html/image_rec/'
font = ImageFont.truetype('/data/fengjing/ocr_recognition_test/fonts/chn/华文宋体.ttf',36)
char_set = open('./densenet_ch/chn7213.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
eng_dict = eng_dict('./densenet_ch/corpus/engset.txt')
char_set = [c.strip('\n') for c in char_set]
char_set.append('卍')
#r_char_set = ''
#for c in char_set:
#    r_char_set += c
#    #char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])
#r_char_set += '卍'
#char_set = r_char_set
#print (char_set[encode_dct[" "]])
#print (char_set[encode_dct["语"]])

Easily_confused = ['入','人','血', '皿', '真', '直', '淋', '沛']
Easily_confused_word = {'径':{'真径':'直径'}}
nclass = len(char_set)
# print(nclass)
mult_model, basemodel = densenet.get_model(False, 32, nclass)
modelPath = os.path.join(os.getcwd(), './models/weights_chn_1103_seal_bg_fg_shufflenet_change_lr01-02-one.h5')#1+2+3.h5weights_chn_add_lishu_test_2_avg_3_4.h5
if os.path.exists(modelPath):
    #multi_model = multi_gpu_model(basemodel, 4, cpu_relocation=True)
    #multi_model.load_weights(modelPath)
    #basemodel = multi_model
    basemodel.load_weights(modelPath)
else:
    print ("No Model Loaded, Default Model will be applied")
    import sys
    sys.exit(-1)


def isalpha(c):
    if c <= 'z' and c >= 'a' or c >= 'A' and c <= 'Z':
        return True
    # if c <= '9' and c >= '0':
    #     return True
    # if c == '.' or c == ',':
    #     return True

    return False
def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
def filter_blank(word,score_list):
    #print('del blank 前',word,score_list)
    if len(word)-word.count('▵')- word.count('▿') != len(score_list):
        print('一开始就不相等')
        return word,score_list
    else:
        word_list = list(word)
        for index ,w in enumerate(word):
            if w ==' ':
                if index == 0 or index == len(word) -1:# 判断句首和句尾
                    word_list[index] = ''
                    score_list[index] = ''
                else:
                    if is_chinese(word[index - 1]) or is_chinese(word[index + 1]):  # 如果前后出现汉字 空格去掉
                        word_list[index] = ''
                        score_list[index] = ''
                    elif word[index + 1] == ' ':  # 如果后面是空格，则 空格去掉
                        word_list[index + 1] = ''
                        score_list[index + 1] = ''
        #print(''.join(filter(None,word_list)))
        text = ''.join(list(filter(None,word_list)))
        #print('del blank 后',text,list(filter(None,score_list)))
        if len(text) ==1 and text[0] == ' ':
            return '',[]
        score_list = list(filter(None,score_list))
        if len(text)-text.count('▵')- text.count('▿')!= len(score_list):
            print('del 后不一样了',len(text),len(score_list),len(word_list))
        return text,score_list
def del_blank(word):
    word = list(filter(None, word.strip().split(' ')))
    if len(word)>0:
        c = word[0]
    else:
        return ''
    for i in range(len(word) - 1):
        if not ('\u4e00' <= word[i][-1] <= '\u9fff'):
            if ord(word[i][-1]) > 47 and ord(word[i][-1]) <= 57:
                if len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i + 1][0])) != 0:
                    c = c + ' ' + word[i + 1]
                else:
                    c = c + word[i + 1]
            elif len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i][-1])) == 0:
                if len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i + 1][0])) != 0:
                    c = c + ' ' + word[i + 1]
                else:
                    c = c + word[i + 1]
            elif len(re.compile(r'\b[a-zA-Z]+\b', re.I).findall(word[i][-1])) != 0:
                if '\u4e00' <= word[i+1][0] <= '\u9fff':
                    c = c + word[i + 1]
                else:
                    c = c + ' ' + word[i + 1]
            else:
                if '\u4e00' <= word[i+1][0] <= '\u9fff':
                    c = c + word[i + 1]
                elif len(re.compile(r'\b[a-zA-Z]+\b',re.I).findall(word[i+1][0]))!= 0:
                        c = c + ' ' + word[i + 1]
                else:
                    c = c + word[i + 1]
        else:
            c = c  + word[i + 1]
    return c
def strQ2B(a,max_score_list):
    if len(a)-a.count('▵')- a.count('▿') != len(max_score_list):
        print('在strQb之前就不相等',len(a),len(max_score_list))
    t4 = time.time()
    is_chinese_or = is_chinese(a)
    #print('判断是否是中文时间：',time.time() - t4)
    if is_chinese_or:
        a = a.replace('(','（')
        a = a.replace(')','）')
        a = a.replace(',','，')
        list_a = list(a)
        for i in re.finditer('\.|\(|（|）|\)', a):
            if list_a[i.start()] == '.':
                if max_score_list[i.start()] <0.9:

                    if re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-1]):  #判定中文

                        list_a[i.start()] ='。'
                    elif a[i.start()-1] ==')' and ((ord(a[i.start()-2]) > 47 and ord(a[i.start()-2]) < 59) or re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-2])):
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
        a = ''.join(list_a)
        t3 = time.time()
        try:
            a,max_score_list = filter_blank(a,max_score_list)
        except:
            print('wrong')
            #continue
        #a = del_blank(a)
        #print("del blank时间为：",time.time() - t3)
        return a,max_score_list
    else:
        a = a.replace('（', '(')
        a = a.replace('）', ')')
        #a = a.replace('。','.')
        t3 = time.time()
        try:

            a,max_score_list = filter_blank(a,max_score_list)
        except:
            print('wrong')
            #continue
        #a = del_blank(a)
        #print("del blank时间为：",time.time() - t3)
        return a,max_score_list

def decode_ori_ori(pred):
    char_list = []
    score_list = []
    pred_text = pred.argmax(axis=1)
    max_score_index_list = []
    t2 = time.time()
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = max(pred[i])
            if max_score < 0.1:  # 去掉概率小于0.1
             
                continue
            
            #try:
            #    if char_list[-1] == 'g' and char_list[-2] == 'm'and char_set[pred_text[i]] == 'l':
            #        char_set[pred_text[i]] = '/'
            #except:
            #    print('ssss')
            char_list.append(char_set[pred_text[i]])
            max_score_index_list.append(pred_text[i])
            score_list.append(max_score)
    text = u''.join(char_list)
    #print('原始decode时间:',time.time()-t2)
    #text = text.replace('mgl','mg/')
    #return re.compile(u'[\u4E00-\u9FA5]').sub('',text), score_list
    #print(max_score_index_list)
    t1 = time.time()
    strQ2B_text,score_list = strQ2B(text,score_list)
    t_strQ2B = time.time() - t1
    #print('t_strQ2B 后处理的时间为：',t_strQ2B)
    #if text != strQ2B_text:

        #print('转换前',text)
        #print('转换后',strQ2B(text,score_list))
    
    return strQ2B_text,score_list
def get_word_bigram_score(word_list):#没考虑开头和结尾
    score = 1
    for i, word in enumerate(word_list):
        if word != ' ':

            if i == 0:
                word_combine = word
                score = 1 * 1
            else:
                for j in range(1,i+1):
                    if word_list[i - j] != ' ':
                        word_combine = word_list[i - j]+' ' + word
                        break
            #print(word_combine)
            if word_combine in lfreq_word and i!=0:
                score = score * lfreq_word[word_combine]*1.0/lfreq_word[word]
                #print(word_combine,lfreq[word_combine])
            else:
                score = score* k
    return score
k = 0.00000001
def eng_error_correction(text_tmp_list,score_list_tmp,wrong_charindex_list,text_tmp):
    tmp_word_list = []
    tmp_dict = {}
    if len(wrong_charindex_list) ==0:
        score = get_list_score(score_list_tmp)
        return [{text_tmp:{'score':score,'score_list':score_list_tmp}}]
    for i in itertools.product(*text_tmp_list):
        word = ''.join(list(i))
        #if '卍' in word:  # 如果占位符在还得减去一个score 麻烦
        score = 1
        if word.replace('卍','').lower() in word_dict:   #将占位符过滤掉
            score_list = []
            for index, w in enumerate(word):
                if w == '卍':
                    continue
                else:
                    score_list.append(text_tmp_list[index][w])
                    score *= text_tmp_list[index][w]
            tmp_dict[word.replace('卍','')] ={'score':score,'score_list':score_list}
            logging.info('word 转换前%s'% text_tmp)
            logging.info('word 转换后%s'% word.replace('卍',''))
    if tmp_dict == {}:
        tmp_dict[text_tmp] = {'score':get_list_score(score_list_tmp),'score_list':score_list_tmp}

    tmp_word_list.append(tmp_dict)
    if len(tmp_word_list)>0:
        # print('转换前',text_tmp)
        # print('转换后',tmp_word_list)
        return tmp_word_list
    else:
        score = 1
        for s in score_list_tmp:
            score*= s
        return [{text_tmp:{'score':score,'score_list':score_list_tmp}}]
def get_list_score(score_list):
    score = 1
    for i in score_list:
        score*= i
    return score

def decode_chn_eng(pred):
    pred_text = pred.argmax(axis=1)
    text_tmp = '' #存放临时单词
    text_tmp_list = []  #存放临时的字符 + score
    score_list_tmp = []  #存放tmp Max score
    wrong_charindex_list = []  #存放tmp 嫌疑字符列表
    wrong_charindex = 0
    word_list = [] #存放word
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = pred[i][pred_text[i]]
            pred[i][pred_text[i]] = 0
            char = char_set[pred_text[i]]
            if isalpha(char) and max_score >0.1:
                if max_score < 0.9:
                    wrong_charindex_list.append(wrong_charindex)  #嫌疑字index列表
                    second_char_index = pred[i].argmax(axis = 0)
                    text_tmp_list.append({char:max_score,char_set[second_char_index]:pred[i][second_char_index]})
                else:
                    text_tmp_list.append({char:max_score})
                text_tmp+=char
                score_list_tmp.append(max_score)
                wrong_charindex += 1
            else:
                if text_tmp != '':
                    if text_tmp.lower() in word_dict:    #如果是单词 继续
                        score_list_tmp =score_list_tmp
                        text_tmp_1 = [{text_tmp:{'score':get_list_score(score_list_tmp),'score_list':score_list_tmp}}]

                        need_bigram = False
                    else: #如果不是单词 纠错
                        # if text_tmp == 'Crossiand':
                        #     print('哈哈哈哈哈哈哈', text_tmp)
                        #print('text_tmp',text_tmp)
                        text_tmp_1 = eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list,text_tmp)  #得到修正后的单词列表
                    word_list +=(text_tmp_1)
                    text_tmp = ''
                    wrong_charindex = 0
                    wrong_charindex_list = []
                    score_list_tmp = []
                    text_tmp_list = []
                #if is_chinese(char): #对汉字的处理
                if char in Easily_confused_word: #如果在易混淆词库
                    s1 = list(word_list[-1].keys())[0]
                    if len(word_list) > 0 and char_in_Easily_confused_word(s1, char):
                        tmp_s = char_in_Easily_confused_word(s1, char)

                        word_list[-1][tmp_s[0]] = word_list[-1][s1]
                        word_list[-1].pop(s1)  #将原字符删除掉
                        #text_tmp = text_tmp[:-1] + char_in_Easily_confused_word(text_tmp[-1], char)
                    word_list.append({char: {'score': max_score}})
                    #print('if',{char: {'scores': max_score}})

                elif (char in Easily_confused and max_score < 0.95) or max_score < 0.6:  #如果字符是易混淆字符且概率小于0。95 或者最大值小于0。6
                    second_char_index = pred[i].argmax(axis=0)
                    if second_char_index != nclass - 1:
                        second_char = char_set[second_char_index]
                        char  = {char:{'score':max_score},second_char:{'score':pred[i][second_char_index]}}
                        word_list.append(char)
                        #print('elif ',char)
                    else:
                        word_list.append({char:{'score':max_score}})  #加一个else啊
                else:
                    #print('else',{char:{'score':max_score}})
                    word_list.append({char:{'score':max_score}})
                '''
                if max_score<0.9:  #对符号的操作
                    second_char_index = pred[i].argmax(axis = 0)
                    if second_char_index != nclass - 1:
                        second_char = char_set[second_char_index]
                        #if
                        if ('°' != char) and ('▵'  not in char) and ('▿' not in char):
                            if '▵' in second_char or '▿' in second_char :
                                #print('获取捞出来了？', char, second_char)
                                char = {second_char:{'score':pred[i][second_char_index]}}
                                max_score = pred[i][second_char_index]
                        elif max_score <0.8 and ('▵'  not in char) and ('▿' not in char):  #如果分数<0。8 则记录需要做bigram的符号标记    将角标去掉
                            char = [{char:{'score':max_score},second_char:{'score',pred[i][second_char_index]}}]   '''
                # if type([2]) == type(char):    #一步一个坑
                #     word_list+=char
                # elif type({}) == type(char):
                #     word_list.append(char)
                # else:
                #     word_list.append({char:{'score':max_score}})

    if len(text_tmp)>0:
        text_tmp_1 = eng_error_correction(text_tmp_list, score_list_tmp, wrong_charindex_list,text_tmp)
        word_list+= text_tmp_1
    word_list = list(filter(None, word_list))  #必须要过滤 否则path为空
    paths = list(itertools.product(*word_list))
    word_bigram_score_list = []
    score_list_final = []
    if len(paths)> 1:
        #print('rrrrr')
        gamma = 1  #发射概率的阈值
        alpha = 0.5  #LM 的阈值

        for path in paths:
            word_bigram_score_path = []
            word_bigram_score = get_word_bigram_score(path)**alpha
            path= list(path)
            p_pred = 1
            for j in range(len(path)):
                #try:
                score_path = word_list[j][path[j]]['score']  #获得每个词的分数
                #except:
                #    print('score_path',word_list)
                if 'score_list' in  word_list[j][path[j]]:
                    word_bigram_score_path+=word_list[j][path[j]]['score_list']
                else:
                    word_bigram_score_path += [word_list[j][path[j]]['score']]   #获得每个path中每个字符分数列表
                p_pred *= score_path

            word_bigram_score_list.append(word_bigram_score*p_pred**gamma)   #
            score_list_final.append(word_bigram_score_path)
        max_score_index = np.argmax(np.array(word_bigram_score_list),axis=0)
        #print(''.join(paths[max_score_index]))

        return ''.join(paths[max_score_index]),score_list_final[max_score_index]
    elif len(paths)==1 :
        #score_list_final = []
        path = list(paths[0])
        for j in range(len(path)):
            if 'score_list' in word_list[j][path[j]]:
                #try:
                score_list_final += word_list[j][path[j]]['score_list']
            else:
                #try:
                score_list_final.append(word_list[j][path[j]]['score'])

        return ''.join(list(paths[0])),score_list_final   #,score_list     ###score  等下再拿出来

def predict_batch(img,picname_list):
    global graph
    with graph.as_default():
        y_pred = basemodel.predict_on_batch(img)[:,2:,:]
    #y_pred = basemodel.predict_on_batch(img)[:,2:,:]
    a = time.time()
    np.save('npy_chn/'+str(a),y_pred)
    img_info = []
    assert len(img)== len(y_pred)
    for i in range(len(y_pred)):
        y_pred_ori = y_pred[i].copy()
        try:
            text,scores = decode_chn_eng(y_pred[i])
            text,scores = strQ2B(text,scores)
        except:
            continue
        if len(scores) != len(text)-text.count('▵')-text.count('▿'):
            print('又不一样了',text,scores)
        text_ori,scores_ori = decode_ori_ori(y_pred_ori)
        #print('text_ori',text_ori)
        #if text != text_ori:
        #scores = [float(ele) for ele in scores]
        #rec = rec.tolist()
        #rec.append(degree)
        #if len(text) > 0:
        color_normal = (0,0,0)
        color_red = (255,0,0)
        width = font.getsize(text)[0]
        im = Image.new("RGB",(width+20,46),(255,255,255))
        draw = ImageDraw.Draw(im)
        imagename = {}
        imagename['img_name'] = picname_list[i]['picname']
        #imagename['text'] =
        label_and_rec_text = {}
        label_and_rec_text['label'] = ''
        label_and_rec_text['rec_text'] = text
        imagename['text'] = label_and_rec_text
        imagename['rec_img']= 'rec_'+picname_list[i]['picname']
        #imagename['location'] = picname_list[i]['location']
        
        img_info.append(imagename)
        if del_blank(text)!= del_blank(text_ori):
            imagename['text']['label'] = text_ori
            rec_json_file.write(json.dumps(imagename)+'\n')
            #json.dump(json.dumps(imagename),rec_json_file)       
        if len(scores)>0:
            if len(np.where(np.array(scores)<0.9)[0]) == 0:
               draw.text((5,5), text, fill=color_normal, font=font)
            else:
                    try:
                        start_x = 5
                        for index,t in enumerate(text):
                            if scores[index]<0.6:

                                draw.text((start_x,5), t, fill=color_red, font=font)
                            else:
                                draw.text((start_x,5), t, fill=color_normal, font=font)
                            start_x += font.getsize(t)[0]
                    except:
                        continue
        im.save(os.path.join(rec_pic_path,'rec_'+picname_list[i]['picname']))
        
    return img_info
def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    img = np.array(img)
    img = cv2.resize(img,(width,32),interpolation=cv2.INTER_CUBIC)
    #img = img.resize([width, 32], Image.ANTIALIAS)
   
    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    #img = np.array(img).astype(np.float32) / 255.0 - 0.5
    img = img.astype(np.float32) / 255.0 - 0.5
    
    X = img.reshape([1, 32, width, 1])
    X = X.swapaxes(1,2)
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, 2:, :]
   # outTxt = open('out.txt','w')
   # for i in range(len(y_pred[0]))
        
    #out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],greedy=False, beam_width=100, top_paths=2)[0][0])[:, :]
    #out = u''.join([characters[x] for x in out[0]])
    #print(y_pred.shape)
    out = decode(y_pred)
    
    return out
