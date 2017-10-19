import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from augmentators import randomHueSaturationValue, randomHorizontalFlip, randomShiftScaleRotate
from u_net import get_unet_128
import glob

orig_width = 320
orig_height = 240

epochs = 50
batch_size = 1
input_size, model = get_unet_128()
model.load_weights(filepath='weights/best_weights.hdf5')

print(input_size)


test_filenames = glob.glob("input/test/*.jpg")
test_filenames = [filename.replace('\\','/').replace('.jpg', '') for filename in test_filenames]
test_filenames = [filename.split('/')[-1] for filename in test_filenames]


print('Predicting on {} samples with batch_size = {}...'.format(len(test_filenames), batch_size))
for start in tqdm(range(0, len(test_filenames), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(test_filenames))
    ids_test_batch = test_filenames[start:end]
    for id in ids_test_batch:
        img = cv2.imread('input/test/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for index, pred in enumerate(preds):
        print(preds)
        prob = cv2.resize(pred, (orig_width, orig_height)) * 255
        current_filename = ids_test_batch[index]
        cv2.imwrite('output/{}.png'.format(id), prob)

print("Done!")