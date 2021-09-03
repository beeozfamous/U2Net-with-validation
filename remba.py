import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image as Img
import cv2

THRESHOLD = 0.9
RESCALE = 255
LAYER = 2
COLOR = (0, 0, 0)
THICKNESS = 4
SAL_SHIFT = 100

src='test_data/test_images/'
mask='test_data/u2net_results/'
remba='test_data/remba/'


for filename in os.listdir(src):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):

        filename_only=os.path.splitext(filename)[0]
        lil_path=os.path.join(src, filename)
        lil_path_mask=os.path.join(mask, filename_only+".png")

        print(lil_path)
        print(lil_path_mask)

        output = load_img(lil_path_mask)
        out_img = img_to_array(output)
        out_img /= RESCALE

        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0

        shape = out_img.shape
        a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
        mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
        a_layer = mul_layer * a_layer_init
        rgba_out = np.append(out_img, a_layer, axis=2)

        input = load_img(lil_path)
        inp_img = img_to_array(input)
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        inp_img /= RESCALE

        a_layer = np.ones(shape=(shape[0], shape[1], 1))
        rgba_inp = np.append(inp_img, a_layer, axis=2)

        rem_back = (rgba_inp * rgba_out)
        rem_back_scaled = rem_back * RESCALE

        rem_last=(cv2.resize(rem_back,(int(shape[1]/3),int(shape[0]/3)))*255).astype(np.uint8)
        cv2.imwrite(remba+filename_only+".png",rem_last)

    else:
        continue


