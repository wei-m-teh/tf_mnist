#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Tutorial using tensorflow with Keras

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import argparse


# Helper libraries
import numpy as np
import os, sys
from os.path import join
from tensorflow.examples.tutorials.mnist import input_data

NUM_CLASSES = 10

def train():

    def nn_layers():
        # model = keras.Sequential([
        #     keras.layers.Flatten(input_shape=(28, 28, 1)),
        #     keras.layers.Dense(128, activation='relu', name= "features"),
        #     keras.layers.Dense(NUM_CLASSES, activation='softmax')
        # ])
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu', name='features'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
        return model

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      fake_data=FLAGS.fake_data,
                                      reshape=False, one_hot=True)
    print(mnist.train.images.shape)
    print(mnist.test.images.shape)
    print(mnist.validation.images.shape)

    model = nn_layers()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    # save class labels to disk to color data points in TensorBoard accordingly
    with open(join(FLAGS.log_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, mnist.test.labels)

    tensorboard = keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir,
                                              embeddings_freq=1,
                                              embeddings_data=mnist.test.images,
                                              embeddings_layer_names=['features'],
                                              histogram_freq=1,
                                              write_graph=True,
                                              write_grads=True,
                                              write_images=True)
    tensorboard.sess = K.get_session()
    test_imgs = mnist.test.images.astype('float32')
    test_labels = mnist.test.labels.astype('float32')
    model.fit(mnist.train.images, mnist.train.labels, validation_data=(test_imgs, test_labels), epochs=5, batch_size=FLAGS.batch_size, callbacks=[ tensorboard ])
    val_loss, val_acc = model.evaluate(mnist.validation.images, mnist.validation.labels)
    print("validation loss: {}, validation accuracy: {}".format(val_loss, val_acc))
    return model, mnist


def predict(model, test_images):
    predictions = model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]))

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    with tf.Graph().as_default():
        model, images = train()
        predict(model, images.test.images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini Batch size to optimize the model with.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/mnist_with_summaries'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    # argument argv here is only passing the 'unparsed arguments to the function. This step is actually redundant because
    # the in main function, argv is effectively ignored.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


