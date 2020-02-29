import requests
import json
import sys
import os
import random
from xml.dom.minidom import Document
def image_rec2xml(dom,imgnameText,recs):
    dom.appendChild(tagset)
    image=dom.createElement('image')
    tagset.appendChild(image)
    imageName = dom.createElement('imageName')
    image.appendChild(imageName)
    name = dom.createTextNode(imgnameText)
    imageName.appendChild(name)
    taggedRectangles = dom.createElement('taggedRectangles')
    image.appendChild(taggedRectangles)
    for rec in recs:
        taggedRectangle = dom.createElement('taggedRectangle')
        taggedRectangles.appendChild(taggedRectangle)
        taggedRectangle.setAttribute('y',str(int(rec[1])))

        taggedRectangle.setAttribute('x',str(int(rec[0])))
        taggedRectangle.setAttribute('width', str(int(rec[2]-rec[0])))
        taggedRectangle.setAttribute('height', str(int(rec[7]-rec[1])))
        taggedRectangle.setAttribute('offset', '0')

dom = Document()
tagset=dom.createElement('tagset')

imgpath = '/data2/fengjing/pse_test_data/image'
img_list = os.listdir(imgpath) 
random.shuffle(img_list)
num = 0
index = 1
for img_name in img_list:
    try:
    #if True:
        img = os.path.join(imgpath,img_name)
        r = requests.post(url='http://0.0.0.0:8888/ocr_recognition_test?language=chn', files = {'file':open(img,'rb')})
        #r1 = requests.post(url='http://0.0.0.0:/ocr_pse_test?language=eng', files = {'file':open(img,'rb')})
        #print(r.content)
        recs = json.loads(r.content)['img_info']
        image_rec2xml(dom, img_name, recs)
        #num += len(rec_info['img_info'])
        print('完成图片{}个,总共{}个'.format(index,len(img_list)))
        #print('已完成小图片{}个'.format(num))
        index+=1
        if index >20000:
            break
    except  Exception as e :
        print(e)
        continue
with open('a.xml','w')as f:
    dom.writexml(f,indent='',addindent='\t',newl='\n',encoding='UTF-8')
#rec_info = json.loads(r.content)
#img_info = rec_info['img_info']
#for info in img_info:
#    print(info['text']['rec_text'].decode('unicode-escape'))
