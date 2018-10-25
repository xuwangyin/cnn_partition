import json

import os
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import inception_v3
from keras_applications import inception_v3 as keras_inception_v3
import numpy as np


if __name__ == '__main__':
    model = keras_inception_v3.InceptionV3(weights='imagenet', include_top=True)
    x = np.random.rand(1, 299, 299, 3).astype(np.float32)
    print(model.predict(x)[0, :5])
    for split in ['mix0', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'mix7', 'mix8']:
        split0, split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split=split)
        split0.load_weights('weights/inceptionv3_{}_split0.h5'.format(split))
        layer_output = split0.predict(x)
        payload = {'data': layer_output.flatten().tolist(), 'shape': layer_output.shape, 'split': split}
        headers = {'content-type': 'application/json'}
        response = requests.post('http://128.143.233.8:5000/partial_inference', json=payload, headers=headers)
        print(response.elapsed.total_seconds())
        result = json.loads(response.text)
        print(np.array(result['prediction'])[:5])

