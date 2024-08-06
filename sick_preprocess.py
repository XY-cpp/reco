import glob
import os
from PIL import Image

from module_list import *
from json2labelImg import json2labelImg

root = 'dataset'
im_root = 'dataset/sick/images'
label_root = 'dataset/sick/labels'

os.makedirs(im_root + '/train')
os.makedirs(im_root + '/val')
os.makedirs(label_root + '/train')
os.makedirs(label_root + '/val')

im_list = glob.glob(root + '/sick_origin/*.JPG')
im_list.sort()
label_list = glob.glob(root + '/sick_origin/*.json')
label_list.sort()
carve_num = int(len(im_list) * 0.8)

counter = 0
for i in im_list[:carve_num]:
    im = Image.open(i)
    im = im.resize((1024,512))
    im.save(im_root+'/train/{}.jpg'.format(counter))
    counter += 1
print('Training RGB images processing has completed.')

counter = 0
for i in im_list[carve_num:]:
    im = Image.open(i)
    im = im.resize((1024, 512))
    im.save(im_root + '/val/{}.jpg'.format(counter))
    counter += 1
print('Validation RGB images processing has completed.')

counter = 0
for i in label_list[:carve_num]:
    im = json2labelImg(i)
    im = im.resize((1024, 512))
    im.save(label_root + '/train/{}.png'.format(counter))
    counter += 1
print('Training Label images processing has completed.')

counter = 0
for i in label_list[carve_num:]:
    im = json2labelImg(i)
    im = im.resize((1024, 512))
    im.save(label_root + '/val/{}.png'.format(counter))
    counter += 1
print('Validation Label images processing has completed.')

print('All Done.')
