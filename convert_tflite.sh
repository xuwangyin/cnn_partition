find weights -name "*split0.hdf5" -exec python keras_to_tensorflow/keras_to_tensorflow.py --input_model={} --output_model={} \;
