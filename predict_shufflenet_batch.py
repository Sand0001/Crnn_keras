#-*- coding:utf-8 -*-
import os
import re
import sys
import cv2
import json
import time
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras import backend as K
from keras.layers import Input
from keras.models import Model
# import keras.backend as K
from keras.utils import multi_gpu_model
import shutil
#import dl_resnet_crnn as densenet
#import dl_resnet_4_crnn_cudnnlstm as densenet
import shufflenet_res_crnn as densenet
#net_4_crnn_cudnnlstmeload(densenet)

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
GPU_NUM = 2
encode_dct =  {}
'''
char_set = open('chn.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
#这里有bug，因为' '空格会被忽略，导致类别数不对
char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])
'''
char_set = open('eng.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i
char_set = [c.strip('\n') for c in char_set]
#char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])
char_set.append('卍')
#char_set = ''.join([chr(i) for i in range(32, 127)] + ['卍'])
nclass = len(char_set)
lfreq = json.loads(open('./count_big.json','r').readlines()[0])
Easily_confused = ['人','入']
#Easily_confused_word = {'径':{'真径':'直径'},'入':{'传入':'传入'}}
Easily_confused_word = {'径':{'真径':'直径'}}
punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。;,. :?!'

mult_model, basemodel = densenet.get_model(False, 32, nclass)

modelPath = sys.argv[2]
if os.path.exists(modelPath):
	try:
		multi_model = multi_gpu_model(basemodel, gpus=GPU_NUM)
		multi_model.load_weights(modelPath)
	except:
		basemodel.load_weights(modelPath)



def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
def ctc_decode(pred):
	c = K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=False, beam_width=10)[0][0]
	#print (c)
	text = ''
	for i in K.get_value(c)[0]:
		text+=char_set[i]
	return text

def decode_ori(pred):
	char_list = []
	pred_text = pred.argmax(axis=1)
	prob_list = []
	debug_list = []
	#print (pred[0])
	#print (pred[1])
	#print (pred[2])
	for i in range(len(pred_text)):
		if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
		#if pred_text[i] != nclass - 1:
			char_list.append(char_set[pred_text[i]])
			#prob_list.append(max(pred[0][i]))
			prob_list.append(pred[i][pred_text[i]])
			debug_list.append(char_set[pred_text[i]] + ":" + str(pred[i][pred_text[i]]))
	#print (u''.join(char_list))
	#print (u''.join([str(c) for c in prob_list]))
	#print (u' '.join([str(c) for c in debug_list]))
	text = u''.join(char_list)
	return text
k = 0.00000001
def get_bigram_score(s):
    score = 1
    spilt_list = list(s)
    for i,word  in enumerate(spilt_list):
        if i == 0:
            word_combine = word
            score = 1*1
        else:
            word_combine = spilt_list[i-1]+word

        if word_combine in lfreq and i!=0:
            score = score * lfreq[word_combine]*1.0/lfreq[word]
            #print(word_combine,lfreq[word_combine])
        else:
            score = score* k
            #print(word_combine,0)
    return score
def char_in_Easily_confused_word(s1,s2):
    if s1 + s2 in Easily_confused_word[s2]:
        print('转换前', s1 + s2)
        print('转换后', Easily_confused_word[s2][s1 + s2])
        return Easily_confused_word[s2][s1 + s2]
def Vierbi_simple(text_tmp,score_list_tmp,second_score_list_index_tmp,thresh_tmp):
    thresh_big = 0.95
    thresh = 0.6
    wrong_w_index_list = []
    start_index = 0
    if len(thresh_tmp) >0:
        for j in range(len(thresh_tmp)):
            wrong_w_index_list += np.where(np.array(score_list_tmp[start_index:thresh_tmp[j]-1]) < thresh)[0].tolist()
            if score_list_tmp[thresh_tmp[j]-1] < thresh_big:
                wrong_w_index_list += [thresh_tmp[j]-1]
            if j!= len(thresh_tmp)-1:
                start_index = j+1
            else:
                break
    else:
        wrong_w_index_list = np.where(np.array(score_list_tmp) < thresh)[0]  # 查看临时 score_list中有没有低于0.9的
    # try:
    #    wrong_w_index_list = list(set(wrong_w_index_list))
    # except:
    #    wrong_w_index_list = list(set(wrong_w_index_list.tolist()))
    if len(wrong_w_index_list) >0 :
        print('wrong_w_index_list ',wrong_w_index_list)
        print('wrong_w_index_list 长度',len(wrong_w_index_list))
        print('找到嫌疑句子: {}  嫌疑字为：{}'.format(text_tmp, [(text_tmp[i],score_list_tmp[i]) for i in wrong_w_index_list]))
    #if len(wrong_w_index_list) == 1 and len(score_list_tmp) > 1:
    if len(score_list_tmp) > 1:    #现在将限定一个字数解除，可更改多个错字

        for j in wrong_w_index_list:
            tmp_char_list = list(text_tmp)

            second_score_index = second_score_list_index_tmp[j].argmax(axis=0)  # second index
            second_score = second_score_list_index_tmp[j][second_score_index]
            tmp_char_list[j] = char_set[second_score_index]
            if j ==0:
                gama = 0.1  #惩罚因子
            else:
                gama = 1
            # s_score = get_bigram_score(text_tmp)
            # sb_score = get_bigram_score(''.join(tmp_char_list))   # 获取 校正后bigram分数
            s_score = get_bigram_score(text_tmp) ** gama*score_list_tmp[j]
            sb_score = get_bigram_score(''.join(tmp_char_list)) ** gama *second_score # 获取 校正后bigram分数
            if s_score > sb_score:
                text_tmp = text_tmp
            else:
                print('转换前', text_tmp)
                print('转换后', ''.join(tmp_char_list))
                text_tmp = ''.join(tmp_char_list)

    #else:
    text_tmp_final = text_tmp  # 将临时text 合并到最终返回的text里
    return text_tmp_final
def decode_Viterbi(pred):
    pred_text = pred.argmax(axis=1)
    text_final = ''
    text_tmp = ''
    score_list_tmp = []
    second_score_list_index_tmp = []
    score_list = []
    thresh_tmp = []
    num_symble = 0
    for i in range(len(pred_text)):
        # pred if pred_text[i] != nclass - 1: #and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))):
            max_score = pred[i][pred_text[i]]
            # score_list.append(max_score)
            # print(max_score)
            pred[i][pred_text[i]] = 0
            second_score_list_index_tmp.append(pred[i])

            # print(pred_text[i],second_score_index)
            # second_score_list.append(second_score_index)
            # second_score_list_float.append(pred[0][i][second_score_index]) #获取second分数
            char = char_set[pred_text[i]]
            if char not in punctuation:
                score_list_tmp.append(max_score)
                # second_score_list_index_tmp.append(second_score_index)
                num_symble += 1
                if char in Easily_confused_word:  # 向前寻找词语，向前查找一个字 强制替换易混词  现在先查找一个字 ，之后根据统计添加查找多个字
                    # thresh_tmp.append(False)    # 判断阈值的tmp_list
                    if len(text_tmp) > 0:
                        char_word = char_in_Easily_confused_word(text_tmp[-1], char)
                        if char_word:
                            text_tmp = text_tmp[:-1] + char_word
                        else:
                            text_tmp += char
                    else:
                        text_tmp += char
                elif char in Easily_confused:
                    # thresh_tmp.append(True)  # 判断阈值的tmp_list
                    thresh_tmp.append(num_symble)
                    text_tmp += char
                else:
                    # thresh_tmp.append(False)
                    text_tmp += char
            else:
                text_final += Vierbi_simple(text_tmp, score_list_tmp, second_score_list_index_tmp, thresh_tmp)
                # text_final += text_tmp_final
                text_final += char  # 将else的标点也加上
                score_list += score_list_tmp  #
                score_list.append(max_score)  # 目前分数列表里不管有没有矫正存的都是分数最大值

                second_score_list_index_tmp = []  # 将临时 second score index list 初始化
                score_list_tmp = []  # 将临时最大分数list初始化
                thresh_tmp = []
                num_symble = 0
                text_tmp = ''  # 将临时text 初始化
                continue
    if len(text_tmp) > 1:  # 为防止遗漏 ，还是加一下text_tmp
        text_tmp_final = Vierbi_simple(text_tmp, score_list_tmp, second_score_list_index_tmp, thresh_tmp)
        # text_tmp_final = text_tmp
    else:
        text_tmp_final = text_tmp

    text_final += text_tmp_final
    score_list += score_list_tmp
    text = text_final
    return text

def predict(img):
	width, height = img.size[0], img.size[1]
	scale = height * 1.0 / 32
	width = int(width / scale)
#	print (width, height) 
	#width = 280
	img = img.resize([width, 32], Image.ANTIALIAS)
	#print (img)
	'''
	img_array = np.array(img.convert('1'))
	boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
	if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
		img = ImageOps.invert(img)
	'''

	img = np.array(img).astype(np.float32) / 255.0 - 0.5
#	print (img.shape)
	X = img.reshape([1, 32, width, 1])
	X = X.swapaxes(1,2)
	#print("X", X.shape)
	y_pred_1 = basemodel.predict(X)
	#print (y_pred.shape)
	#print (y_pred[0])
	y_pred = y_pred_1[:, 2:, :]
	#print (y_pred.argmax(axis=2)[0])
	# out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
	# out = u''.join([characters[x] for x in out[0
	y_pred1 = y_pred.copy()
	a = time.time()
	#out = decode_Viterbi(y_pred[0])
	#b= time.time()
	out = decode_ori(y_pred1)
	#c = time.time()
	#print('时间差为：',c-2*b+a)
	#out = decode_Viterbi(y_pred[0])
	#out = ctc_decode(y_pred1)
	#print('out',out)
	return out,y_pred_1.argmax(axis=2)[0]
def del_blank(word):
    word = list(filter(None,word.strip().split(' ')))
    if len(word) == 0:
        return ''
    
    c = word[0]
    for i in range(len(word)-1):
        if len(re.compile(r'\b[a-zA-Z]+\b',re.I).findall(word[i][-1]))!= 0:
            if len(re.compile(r'\b[a-zA-Z]+\b',re.I).findall(word[i+1][0]))!= 0:
                c = c +' '+word[i+1]
            else:
                c = c + word[i + 1]
        else:
            c = c+word[i+1]
    return c
def del_blank_old(word):
    w1 = ''
    for index,w in enumerate(word.strip()):
        if ord(w) !=32:
            w1+=w
        else:
            if ord(word[index+1])!=32 and ord(word[index-1])!=32:
                if (ord(word[index+1]) >64 and ord(word[index+1]) <91) or (ord(word[index+1]) >96 and ord(word[index+1]) <123):
                    w1+=w
            else:
                continue
    return w1.replace('  ',' ')
def strQ2B(a):
    if is_chinese(a):
        a = a.replace('(','（')
        a = a.replace(')','）')
        a = a.replace(',','，')
        a = a.replace(':', '：')
        a = a.replace('?', '？')

        list_a = list(a)
        for i in re.finditer('\.|\(|（|）|\)', a):
            if list_a[i.start()] == '.':
                #if max_score_list[i.start()] <0.9:
                    if re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-1]):  #判定中文
                        list_a[i.start()] ='。'
                    elif a[i.start()-1] ==')' and ((ord(a[i.start()-2]) > 47 and ord(a[i.start()-2]) < 59) or re.compile(u'[\u4E00-\u9FA5]').findall(a[i.start()-2])):
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
                    elif a[i.start()-1] =='）':
                        list_a[i.start()] = '。'
        a = ''.join(list_a)
        a = del_blank(a)
        return a
    else:
        a = a.replace('（', '(')
        a = a.replace('）', ')')
        a = a.replace('。','.')
        a = a.replace('：', ':')
        a = a.replace('？', '?')
        a = del_blank(a)
        return a
def is_upper(word):
        for w in word:
                if w>= 'a' and w<='z':
                        return False
        for w in word:
                if w >= 'A' and w <= 'Z':
                        return True
        return False

def img_resize(partImg):
    image = cv2.cvtColor(partImg, cv2.COLOR_BGR2GRAY)
    width, height = image.shape[1], image.shape[0]
    scale = height * 1.0 / 32
    width = int(width / scale)
    image = cv2.resize(image, (width, 32))
    return image

if __name__ == '__main__':
        import sys
        #label_txt = open('','w')
        input_image_path = sys.argv[1]
        if "jpg" in input_image_path or 'png' in input_image_path:
            #img = Image.open(input_image_path).convert('L')
            img = Image.open(input_image_path).convert('L')
            print (predict(img))

        else:
            test_img_list = []
            correct = 0
            num = 0
            upper_num = 0
            upper_correct = 0
            test_label_lines = []
            if sys.argv[3] == 'acc':
                test_label_lines = open(sys.argv[4],'r').readlines()
                upper_test = open('upper.json','w')
                script_label = open('../eng_test_subscript/label_pred.txt','a')
                #test_img_list = []
                for line in test_label_lines:
                    picName = line.split(' ')[0].strip()+ '.jpg'
                    #print('picName',picName)
                    
                    test_img_list.append(picName)
               
            #label_txt = open('label_tmp.txt','w')
            batch_img = []
            label_list = []
            aa = time.time()
            for i in os.listdir(input_image_path):
            #for i in range(20):
                #img = Image.open(os.path.join(input_image_path,i).convert('L'))
                if "jpg" in i or 'png' in i:
                    #print(i)
                    #label_text = ''.join(test_label_lines[test_img_list.index(i)].split('.jpg ')[1:]).strip()
                    # img = Image.open(os.path.join(input_image_path, i)).convert('L')
                    # img = np.array(img, 'f') / 255.0 - 0.5
                    #
                    # img = np.expand_dims(img, axis=2).swapaxes(0, 1)
                    img = cv2.imread(os.path.join(input_image_path, i))
                    try:
                        label_list.append(test_label_lines[test_img_list.index(i)])
                        batch_img.append(img)
                    except:
                        continue
                    bb = time.time()
                    if len(batch_img) == 112*2:
                        a = time.time()
                        y_pred = basemodel.predict_on_batch(np.array(batch_img))[:,2:,:]
                        print('batch time',time.time()-a)
                        for j in range(len(y_pred)):
                            a = time.time()
                            label_text = ' '.join(label_list[j].split(' ')[1:]).strip()
                            text = decode_ori(y_pred[j])
                            #print('decode time',time.time()-a)
                            #print(text)
                            if len(test_label_lines) != 0:
                                del_blank_label_text = del_blank(label_text)
                                del_blank_text = del_blank(text)
                                del_blank_label_text = del_blank_label_text.replace('–','-')
                                del_rec_text = del_blank_text.replace('–','-')
                                del_blank_label_text = strQ2B(del_blank_label_text)
                                del_rec_text= strQ2B(del_rec_text)
                            #if '^' in del_blank_label_text or '~' in del_blank_label_text:
                            #    script_label.writelines(i + '  ' + label_text + '\n')
                            #    shutil.copy(os.path.join(input_image_path, i),'../eng_test_subscript/test/')
                            #    continue
                                num+=1
                            
                                if del_blank_label_text == del_rec_text:
                                    if is_upper(del_blank_label_text):
                                        upper_correct +=1
                                        upper_num +=1
                                    correct += 1
                                else:
				#	
                                    if is_upper(del_blank_label_text):
                                        upper_num +=1
                                    #print('text',text)
                                    #print('label_text',label_text)
                                    imagename = {}
                                    imagename['img_name'] = label_list[j].split(' ')[0] + '.jpg'
                                    #imagename['text'] =
                                    label_and_rec_text = {}
                                    label_and_rec_text['label'] = label_text
                                    label_and_rec_text['rec_text'] = text
                                    imagename['text'] = label_and_rec_text
                                    imagename['rec_img']= ''
                                    upper_test.write(json.dumps(imagename) + '\n')
                        print('一个batch 需要时间',time.time()-bb)
                        print('已完成{}个'.format(num))
                        label_list = []
                        batch_img = []
                        
                   # else:
                    #    print(i)
                     #   print(text)
                    if len(sys.argv) > 2 and sys.argv[3]!='acc':
                        #label_txt.writelines(i+ '  '  +text + '\n')
                        try:
                            pic_name = text.replace('/','_') + '.jpg'
                        except:
                            pic_name = text + '.jpg'			
                        #shutil.copy(os.path.join(input_image_path, i),os.path.join(sys.argv[3],pic_name))
                        #label_txt.writelines(pic_name + '  '  +text + '\n')
            if num != 0:
                print('acc:',correct*1.0/num,num, upper_num,upper_correct,'upper acc:',upper_correct *1.0/upper_num)
            else:
                print('YOU AREE WRONG!!!')
