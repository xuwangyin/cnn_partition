import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import keras
import inception_v3
from keras.applications import inception_v3 as keras_inception_v3
import numpy as np
import time


def save_split_models():
    model = keras_inception_v3.InceptionV3(weights='imagenet', include_top=True)
    x = np.random.rand(1, 299, 299, 3).astype(np.float32)
    for split in ['mix0', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'mix7', 'mix8']:
        split0, split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split=split)

        # if not os.path.isfile('weights/inceptionv3_{}_split0.h5'.format(split)):
        for l1, l2 in zip(split0.layers, model.layers):
            l1.set_weights(l2.get_weights())
        # split0.save_weights('weights/inceptionv3_{}_split0.h5'.format(split))
        split0.save('weights/inceptionv3_{}_split0.hdf5'.format(split))
        print("saved split0")

        # if not os.path.isfile('weights/inceptionv3_{}_split1.h5'.format(split)):
        for l1, l2 in zip(split1.layers[::-1], model.layers[::-1]):
            l1.set_weights(l2.get_weights())
        # split1.save_weights('weights/inceptionv3_{}_split1.h5'.format(split))
        split1.save_weights('weights/inceptionv3_{}_split1.hdf5'.format(split))

        print("saved " + split)


if __name__ == "__main__":
    save_split_models()
