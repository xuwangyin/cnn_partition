import json
import io
import gzip
import zlib

import os
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import inception_v3
from keras_applications import inception_v3 as keras_inception_v3
import numpy as np

compress = True


if __name__ == '__main__':
    model = keras_inception_v3.InceptionV3(weights='imagenet', include_top=True)
    x = np.random.rand(1, 299, 299, 3).astype(np.float32)
    print(model.predict(x)[0, :5])
    for split in ['mix0', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'mix7', 'mix8']:
        split0, split1 = inception_v3.InceptionV3(weights='imagenet', include_top=True, split=split)
        split0.load_weights('weights/inceptionv3_{}_split0.h5'.format(split))
        layer_output = split0.predict(x).astype(np.float16)
        payload = {'data': layer_output.flatten().tolist(), 'shape': layer_output.shape, 'split': split}
        if compress:
            request_body = gzip.compress(json.dumps(payload).encode('utf-8'))
            response = requests.post('http://128.143.235.125:5000/partial_inference', data=request_body, headers={'content-encoding': 'gzip'})
        else:
            headers = {'content-encoding': 'json'}
            response = requests.post('http://128.143.235.125:5000/partial_inference', json=payload, headers=headers)
        result = json.loads(response.text)
        print("split: {}, response time: {}, inference time: {}".format(split, response.elapsed.total_seconds(),
                                                                        result['inference_time']))
        print(np.array(result['prediction'])[:5])

