import json

import flask
import os
from flask import Flask
from flask import request

import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import inception_v3
from keras_applications import inception_v3 as keras_inception_v3
import numpy as np
import time


app = Flask(__name__)


@app.route('/partial_inference', methods=['POST'])
def partial_inference():
    data = {"success": False}
    global mix0_split1, mix1_split1, mix2_split1, mix3_split1, mix4_split1
    global mix5_split1, mix6_split1, mix7_split1, mix8_split1
    split_models = {'mix0': mix0_split1, 'mix1': mix1_split1, 'mix2': mix2_split1,
                    'mix3': mix3_split1, 'mix4': mix4_split1, 'mix5': mix5_split1,
                    'mix6': mix6_split1, 'mix7': mix7_split1, 'mix8': mix8_split1
                    }
    if flask.request.method == "POST":
        try:
            request_data = json.loads(request.data)
            layer_output = np.array(request_data['data']).reshape(request_data['shape']).astype(np.float32)
            split1 = split_models[request_data['split']]
            start_time = time.time()
            split_pred = split1.predict(layer_output)
            data['inference_time'] = (time.time() - start_time)
            data['prediction'] = split_pred[0].tolist()
            data["success"] = True
        except:
            pass
    return flask.jsonify(data)


def load_model(split='mix5'):
    global split0, split1
    split0, split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split=split)
    x = np.random.rand(1, 299, 299, 3).astype(np.float32)
    split0.load_weights('weights/inceptionv3_{}_split0.h5'.format(split))
    split1.load_weights('weights/inceptionv3_{}_split1.h5'.format(split))
    print(split1.predict(split0.predict(x))[0, :5])


def load_split_models():
    # mix0_split1 = mix1_split1 = mix2_split1 = mix3_split1 = None
    # mix4_split1 = mix5_split1 = mix6_split1 = mix7_split1 = mix8_split1 = None
    global mix0_split1, mix1_split1, mix2_split1, mix3_split1, mix4_split1
    global mix5_split1, mix6_split1, mix7_split1, mix8_split1
    # split_models = {'mix0': mix0_split1, 'mix1': mix1_split1, 'mix2': mix2_split1,
    #                 'mix3': mix3_split1, 'mix4': mix4_split1, 'mix5': mix5_split1,
    #                 'mix6': mix6_split1, 'mix7': mix7_split1, 'mix8': mix8_split1
    #                 }
    x = np.random.rand(1, 299, 299, 3).astype(np.float32)
    # for splitname, split1 in split_models.items():
    #     split0, split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split=splitname)
    #     split1.load_weights('weights/inceptionv3_{}_split1.h5'.format(splitname))
    #     print(split1.predict(split0.predict(x))[0, :5])

    mix0_split0, mix0_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix0')
    mix0_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix0'))
    mix0_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix0'))
    print(mix0_split1.predict(mix0_split0.predict(x))[0, :5])

    mix1_split0, mix1_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix1')
    mix1_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix1'))
    mix1_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix1'))
    print(mix1_split1.predict(mix1_split0.predict(x))[0, :5])

    mix2_split0, mix2_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix2')
    mix2_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix2'))
    mix2_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix2'))
    print(mix2_split1.predict(mix2_split0.predict(x))[0, :5])

    mix3_split0, mix3_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix3')
    mix3_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix3'))
    mix3_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix3'))
    print(mix3_split1.predict(mix3_split0.predict(x))[0, :5])

    mix4_split0, mix4_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix4')
    mix4_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix4'))
    mix4_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix4'))
    print(mix4_split1.predict(mix4_split0.predict(x))[0, :5])

    mix5_split0, mix5_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix5')
    mix5_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix5'))
    mix5_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix5'))
    print(mix5_split1.predict(mix5_split0.predict(x))[0, :5])

    mix6_split0, mix6_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix6')
    mix6_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix6'))
    mix6_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix6'))
    print(mix6_split1.predict(mix6_split0.predict(x))[0, :5])

    mix7_split0, mix7_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix7')
    mix7_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix7'))
    mix7_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix7'))
    print(mix7_split1.predict(mix7_split0.predict(x))[0, :5])

    mix8_split0, mix8_split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split='mix8')
    mix8_split0.load_weights('weights/inceptionv3_{}_split0.h5'.format('mix8'))
    mix8_split1.load_weights('weights/inceptionv3_{}_split1.h5'.format('mix8'))
    print(mix8_split1.predict(mix8_split0.predict(x))[0, :5])
    

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_split_models()
    print("loaded split models")
    app.run(host='0.0.0.0')

