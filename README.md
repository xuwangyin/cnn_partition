# cnn_partition

Run the server

    export FLASK_APP=cnn_server.py
    python -m flask run --host=0.0.0.0
    
Send requests

    python cnn_client.py


    inceptionv3_mix2_split0.tflite
    input: [<tf.Tensor 'input_6:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed1/concat:0' shape=(?, 35, 35, 288) dtype=float32>]
    -----------------------------
    inceptionv3_mix4_split0.tflite
    input: [<tf.Tensor 'input_10:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed3/concat:0' shape=(?, 17, 17, 768) dtype=float32>]
    -----------------------------
    inceptionv3_mix3_split0.tflite
    input: [<tf.Tensor 'input_8:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed2/concat:0' shape=(?, 35, 35, 288) dtype=float32>]
    -----------------------------
    inceptionv3_mix0_split0.tflite
    input: [<tf.Tensor 'input_2:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'max_pooling2d_6/MaxPool:0' shape=(?, 35, 35, 192) dtype=float32>]
    -----------------------------
    inceptionv3_mix5_split0.tflite
    input: [<tf.Tensor 'input_12:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed4/concat:0' shape=(?, 17, 17, 768) dtype=float32>]
    -----------------------------
    inceptionv3_mix7_split0.tflite
    input: [<tf.Tensor 'input_16:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed6/concat:0' shape=(?, 17, 17, 768) dtype=float32>]
    -----------------------------
    inceptionv3_mix8_split0.tflite
    input: [<tf.Tensor 'input_18:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed7/concat:0' shape=(?, 17, 17, 768) dtype=float32>]
    -----------------------------
    inceptionv3_mix1_split0.tflite
    input: [<tf.Tensor 'input_4:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed0/concat:0' shape=(?, 35, 35, 256) dtype=float32>]
    -----------------------------
    inceptionv3_mix6_split0.tflite
    input: [<tf.Tensor 'input_14:0' shape=(?, 299, 299, 3) dtype=float32>]
    output: [<tf.Tensor 'mixed5/concat:0' shape=(?, 17, 17, 768) dtype=float32>]
