import json
import os
import sys
label_json_chn_list = os.listdir(sys.argv[1])

check_label_text = open('check_label.json','r').readlines()
filter_chek_json = open('filter_check_label.json','w')
#check_label_info = {}
# random.shuffle(check_label_text)
results = []
for check_label_line in check_label_text:
    check_label_line = json.loads(check_label_line)
    label_img_info = {}
    if check_label_line['img_name'][:-4] +'.json' in label_json_chn_list:
        continue
    else:
        filter_chek_json.write(json.dumps(check_label_line) + '\n')
        # label_img_info['img_name'] = check_label_line['img_name']
        #
        # label_and_rec_text = {}
        # #label_and_rec_text['label']= check_label_line['label_text']
        # label_and_rec_text['label']= check_label_line['text']['label']
        # label_and_rec_text['rec_text']= check_label_line['text']['rec_text']
        # label_img_info['text'] = label_and_rec_text

