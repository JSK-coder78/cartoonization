
from __future__ import division, print_function
# coding=utf-8

import requests

import sys
import os
import glob
import re
import uuid
import numpy as np
import tensorflow as tf
import network
import guided_filter
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()

import cv2
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)


def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model


def make_transparent_foreground(pic, mask):
  # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
  # add an alpha channel with and fill all with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
  # merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
  # create a transparent background
    bg = np.zeros(alpha_im.shape)
  # setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
  # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground

def remove_background(model, input_file):
    input_image = Image.open(input_file)
    preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

  # create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image ,bin_mask)

    return foreground, bin_mask



def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image

def cartoonize(img_name, model_path):
    
    input_photo = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.compat.v1.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.compat.v1.train.Saver(var_list=gene_vars)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    try:
        image = resize_crop(img_name)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    except:
        print('cartoonize failed')
    
    return None
              


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("loaded...........", file=sys.stderr)
    img_src = ""
    if request.method == 'POST':
        # Get the file from post request
        
        basepath = os.path.dirname(__file__)
        
        #for file in os.listdir(os.path.join('uploads')):
            #if file.endswith(".jpg"):
                #os.remove(os.path.join('uploads', file))
            
        f = request.files['file']
        
        
        # Save the file to ./uploads

        
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_src = file_path
        print(f.filename, file=sys.stderr)
        
        fname = "bg___" + f.filename
        bg_file_path = os.path.join('static/uploads', secure_filename(fname))
        
        # Make prediction
        
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
             files={'image_file': open(img_src, 'rb')},
             data={'size': 'auto'},
             headers={'X-Api-Key': 'DFChhJ15ho4cj3rKKipYZxRD'},
            )    
        print("completed...........................................", file=sys.stderr)
        
        if response.status_code == requests.codes.ok:
            with open(bg_file_path, 'wb') as out:
                out.write(response.content)
        else:
            print("Error:", response.status_code, response.text)
        
        print(bg_file_path, file=sys.stderr)
        
      
        img = plt.imread(bg_file_path)
        print(img.shape, file=sys.stderr)
        img = np.array(img)
        tp = np.zeros((img.shape[0], img.shape[1], 3))
        
        
        mask = img[:, :, 3]
        tp = img[:, :, :3]
        print(mask.shape, file=sys.stderr)
        print(tp.shape, file=sys.stderr)
        
        
        tp = tp * 255
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 0:
                    tp[i][j][0] = 255
                    tp[i][j][1] = 255
                    tp[i][j][2] = 255
        
        print(tp, file=sys.stderr)
        
        print("..........................................................", file=sys.stderr)
        print(tp.shape,  file=sys.stderr)
        output = cartoonize(tp, "model gan/")
        
        output = output.astype(np.uint8)
        print(output, file=sys.stderr)
        print(output.shape, file=sys.stderr)
        
        
        
        fname_cart = "cart___" + f.filename
        cart_file_path = os.path.join('static/uploads', secure_filename(fname_cart))
        bg = Image.fromarray(output)
        bg.save(cart_file_path) 
        print(cart_file_path, file=sys.stderr)
        
        
        if output.shape[1] < tp.shape[1]:
            temp = np.ones((output.shape[0], tp.shape[1], 3))
            temp = temp * 255

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    temp[i, j, 0] = output[i, j, 0]
                    temp[i, j, 1] = output[i, j, 1]
                    temp[i, j, 2] = output[i, j, 2]

            temp = temp.astype(np.uint8)
            tp = tp.astype(np.uint8)
            mixed = cv2.vconcat([tp, temp])
            
        else:
            tp = tp.astype(np.uint8)
            mixed = cv2.vconcat([tp, output])
            
            

        #mixed = cv2.vconcat([img, output])
        print(mixed.shape, file=sys.stderr)
        
        fname_mixed = "mixed___" + f.filename
        mixed_file_path = os.path.join('static/uploads', secure_filename(fname_mixed))
        bg = Image.fromarray(mixed)
        bg.save(mixed_file_path) 
        
        print(os.path.join('static/uploads', fname_mixed), file=sys.stderr)        

        return os.path.join('static/uploads', fname_mixed) #, "/static/uploads/" + fname_cart
        

    return ('', 204)


if __name__ == '__main__':
    app.run(debug=True)