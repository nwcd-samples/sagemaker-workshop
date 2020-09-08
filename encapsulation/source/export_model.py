import tensorflow as tf
import os
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta


def export_model(model,
                 export_model_dir,
                 model_version
                 ):
    """
    :param export_model_dir: type string, save dir for exported model    url
    :param model_version: type int best
    :return:no return
    """
    with tf.get_default_graph().as_default():
        # prediction_signature
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
        print(model.output.shape, '**', tensor_info_output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input}, # Tensorflow.TensorInfo
                outputs={'result': tensor_info_output},
                #method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                 method_name= "tensorflow/serving/predict")
               
        )
        print('step1 => prediction_signature created successfully')
        # set-up a builder
        
        export_path_base = export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            # tags:SERVING,TRAINING,EVAL,GPU,TPU
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature,
                   tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              prediction_signature,

            },
            )
        print('step2 => Export path(%s) ready to export trained model' % export_path, '\n starting to export model...')
        #builder.save(as_text=True)
        builder.save()
        print('Done exporting!')
