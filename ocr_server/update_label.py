import os
import json
import sys

label_json_path = sys.argv[1]
label_json_list = os.listdir(label_json_path)
label_txt_lines = open(sys.argv[2],'r').readlines()
new_label_txt = open(sys.argv[3],'w')
num = 0
badnum =0
for line in label_txt_lines:
    imgname = line.split('.jpg ')[0] + '.jpg'
    pic_json = line.split('.jpg ')[0]+'.json'
    if pic_json in label_json_list:
        new_label_dict = json.loads(open(os.path.join(label_json_path,pic_json),'r').readlines()[0])
        label_text = str(new_label_dict['text'])
        badcase_or_not = new_label_dict['is_save']
        if badcase_or_not == 'false':
            badnum +=1
            continue
        else:
            num+=1     
            new_label_txt.writelines(imgname + '  ' + label_text+ '\n' )
    else:
        new_label_txt.writelines(line)
print('ignore {}  ,changed {}'.format(badnum,num))




