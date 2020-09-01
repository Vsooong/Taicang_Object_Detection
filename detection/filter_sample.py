from xml.dom.minidom import parse
import xml.dom.minidom
import os
import numpy as np
from PIL import Image, ImageShow

# root = 'H:/DATA/'
img_root_dir='/home/dl/data/taicang/dataset/img'
ans_root_dir='/home/dl/data/taicang/dataset/label'
imgs = [i for i in os.listdir(img_root_dir) if i.endswith('jpg')]
labels = [i for i in os.listdir(ans_root_dir) if i.endswith('xml')]

remove_bad_samples = []
good_test_samples = []

def remove_no_label():
    for i in imgs:
        name = i.split('.')[0]
        label_name = '.'.join([name,'xml'])
        img_path=os.path.join(img_root_dir,i)
        if not label_name in labels:
            remove_bad_samples.append(img_path)

label_min=999
label_max=-999

for idx in range(len(imgs)):
    img_path = os.path.join(img_root_dir, imgs[idx])
    label_path = os.path.join(ans_root_dir, imgs[idx].split('.')[0] + '.xml')
    if not os.path.exists(label_path):
        remove_bad_samples.append(img_path)

for idx in range(len(labels)):
    label_path = os.path.join(ans_root_dir, labels[idx])
    img_path = os.path.join(img_root_dir, labels[idx].split('.')[0] + '.jpg')
    dom = xml.dom.minidom.parse(label_path)
    root = dom.documentElement
    height = int(root.getElementsByTagName('height')[0].firstChild.data)  # Image height
    width = int(root.getElementsByTagName('width')[0].firstChild.data)  # Image width
    d = root.getElementsByTagName('object')
    cls = []
    for index in range(len(root.getElementsByTagName('object'))):
        cl = int(root.getElementsByTagName('name')[index].firstChild.data)
        cls.append(cl)
    if len(cls) == 0:
        print(label_path)
        remove_bad_samples.append(label_path)
        remove_bad_samples.append(img_path)
        continue
    if not os.path.exists(img_path):
        remove_bad_samples.append(label_path)

    max = np.max(cls)
    min = np.min(cls)
    print(max,min)
    if min<=label_min:
        label_min=min
    if max>=label_max:
        label_max=max

    # if max > 6 or min < 1:
    #     # print(min, max, label_path)
    # #     remove_bad_samples.append(label_path)
    # #     remove_bad_samples.append(img_path)
#
#     if max!= min and len(cls)>=5:
#         good_test_samples.append(img_path)


def copyfile(files, dest_root='H:/object_detection/test_image'):
    import shutil
    for i in files:
        destination = os.path.join(dest_root, os.path.basename(i))
        if not os.path.exists(i):
            shutil.copyfile(i, destination)


def deletefile(files):
    import os
    for i in files:
        if os.path.exists(i):
            os.remove(i)
            print('removed: ', i)

# deletefile(remove_bad_samples)
print(label_max,label_min)
# copyfile(good_test_samples)


# test_img_path = "H:/object_detection/test_image_2"
# for name in os.listdir(test_img_path):
#     if name.endswith('det.jpg') or name.endswith('det2.jpg'):
#         rpath=os.path.join(test_img_path, name)
#         os.remove(rpath)
#         print("Removed: ", rpath)
