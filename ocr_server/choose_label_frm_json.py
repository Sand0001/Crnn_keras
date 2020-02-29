import json
import sys
import os
import shutil
oripicPath = sys.argv[2]
newpicPath = sys.argv[3]
badcasePath = sys.argv[4]
newlabeltxt = open(sys.argv[5],'w')
json_list = os.listdir(sys.argv[1])
wrong_num = 0
num = 0
def is_valid(imgname):
    #if 'TNM' in imgname or 'attachment' in imgname:
    #    return True
    #else:
    #    return False
    return True
for json_file in json_list:
    
    if json_file.split('.')[-1] == 'json':
        print(json_file)
        label_dict = json.loads(open(os.path.join(sys.argv[1],json_file),'r').readlines()[0])
        #imgname = label_dict['imgurl'].encode("raw_unicode_escape").decode().split('/')[-1]
        label_text = str(label_dict['text'])
        badcase_or_not = label_dict['is_save']
        try:
#            if badcase_or_not == 'false':
#                shutil.copyfile(os.path.join(oripicPath,imgname),
#                            os.path.join(badcasePath,imgname))
            imgname =label_dict['imgurl'].split('/')[-1].split('.jpg')[0]+'.jpg'
            print(imgname)
            if is_valid(imgname):

                print(os.path.join(oripicPath,imgname))
                if os.path.exists((os.path.join(oripicPath,imgname))):
                    if badcase_or_not == 'false':
                        print('aaaaaaaa')
                        shutil.copyfile(os.path.join(oripicPath,imgname),
                                os.path.join(badcasePath,imgname))
            #label_text = label_dict['text'].encode("raw_unicode_escape").decode()
                    else:
                        newlabeltxt.writelines(imgname + '  ' + label_text+ '\n' )
                        shutil.copyfile(os.path.join(oripicPath,imgname),
                            os.path.join(newpicPath,imgname))
                        num +=1
                else:
                    print('No')
        except Exception as e:
            print(e)
            
            print(json_file)
            wrong_num += 1
            continue
print('move images:',num)
print('wrong :',wrong_num)
print('all:',len(json_list))

