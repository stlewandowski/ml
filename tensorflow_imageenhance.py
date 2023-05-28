"""
Original file is located at
    https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_enhancing.ipynb

##### Copyright 2019 The TensorFlow Hub Authors.
Licensed under the Apache License, Version 2.0 (the "License");

Created by @[Adrish Dey](https://github.com/captain-pool) for [Google Summer of Code](https://summerofcode.withgoogle.com/) 2019
"""

# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Declaring Constants
#IMAGE_PATH = "/content/original.png"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
#SAVED_MODEL_PATH = <local path>



def preprocess_image(image_path):

  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):

  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
def plot_image(image, title=""):

  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)


def main(indir, outdir):
    os.chdir(indir)
    for image in os.listdir(indir):
        IMAGE_PATH = image
        hr_image = preprocess_image(IMAGE_PATH)

        # Plotting Original Resolution image
        #plot_image(tf.squeeze(hr_image), title="Original Image")
        save_image(tf.squeeze(hr_image), filename="Original Image")

        model = hub.load(SAVED_MODEL_PATH)
        #model = tf.saved_model.load(SAVED_MODEL_PATH)

        start = time.time()
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)
        print("Time Taken: %f" % (time.time() - start))

        # Plotting Super Resolution Image
        #plot_image(tf.squeeze(fake_image), title="Super Resolution")
        outfile = os.path.join(outdir, image.replace(".png",""))
        save_image(tf.squeeze(fake_image), filename=outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-directory', action='store', dest='indir', help='directory to pull files from.',
                        required=True, type=str)
    parser.add_argument('-o', '--output-directory', action='store', dest='outdir', help='directory to write files to.',
                        required=True, type=str)
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    main(indir, outdir)