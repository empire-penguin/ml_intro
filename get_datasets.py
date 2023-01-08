import csv
from shutil import copyfile
from tqdm import tqdm


# get targets of training images
with open('train_ids.csv', 'r') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = [int(i) for i in targets[0]]

with open('val_ids.csv', 'r') as f:
    reader = csv.reader(f)
    valIds = list(reader)

valIds = [int(i) for i in valIds[0]]


with open('test_ids.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)

testIds = [int(i) for i in testIds[0]]

for img_id in testIds:
    path = cocoTest.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/val2014/'+path, './data/images/test/'+path)


print("done")
