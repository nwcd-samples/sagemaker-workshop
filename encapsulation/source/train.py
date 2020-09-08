import os
import numpy as np
import argparse
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
import tensorflow as tf
from processing import processing_data
from export_model import export_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150


logger.info('tensorflow version:{}'.format(tf.__version__))
logger.info('keras version:{}'.format(keras.__version__))
logger.info("gpu_device_name:{}".format(tf.test.gpu_device_name()))
logger.info("tf.test.is_gpu_available():{}".format(str(tf.test.is_gpu_available())))



def train(train_dir, validation_dir, test_dir, log_dir, model_dir, tf_server_dir, checkpoint_dir, class_list ,training_count, args):
    class_count = len(class_list)
    print("class_count:"+str(class_count))
    
    # 创建模型
    conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))    
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(class_count, activation='softmax'))
    
    # 查看模型结构
    logger.info('This is the number of trainable weights '
      'before freezing the conv base:'+ str(len(model.trainable_weights)))
    conv_base.trainable = False
    logger.info('This is the number of trainable weights '
          'after freezing the conv base:'+ str(len(model.trainable_weights)))
    #conv_base.summary()
    
    # 准备训练参数
    RUN = RUN + 1 if 'RUN' in locals() else 1
    EPOCHS = args.epoch_count
    batch_size = args.batch_size
    lr = args.lr
    steps_per_epoch = training_count // batch_size
    logger.info("RUN:"+str(RUN))
    logger.info("steps_per_epoch:"+str(steps_per_epoch))
    logger.info("learning rate:"+str(lr))

    # 载入图片数据
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=20,
          width_shift_range=0.30,
          height_shift_range=0.30,
          shear_range=0.20,
          zoom_range=0.40,
          horizontal_flip=True,
          fill_mode='nearest')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            classes=class_list,
            # All images will be resized to 150x150
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            classes=class_list,
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=batch_size,
            class_mode='categorical')
            
    # 第一次训练
    LOG_DIR_1 = os.path.join(log_dir, 'run{}-1'.format(RUN)) 
    LOG_FILE_PATH_1 = os.path.join(checkpoint_dir, 'checkpoint-1-{epoch:02d}-{val_acc:.4f}.hdf5')

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=2e-5) ,
                  metrics=['acc'])
    tensorboard = TensorBoard(log_dir=LOG_DIR_1, write_images=True)
    checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH_1, monitor='val_acc', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

    history = model.fit_generator(
          train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=EPOCHS,
          validation_data=validation_generator,
          validation_steps=50,
          verbose=args.verbose,
          callbacks=[tensorboard, checkpoint, early_stopping])

    # 微调模型
    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        layer.trainable = set_trainable
    #conv_base.summary()
    
    # 第二次训练
    LOG_DIR_2 = os.path.join(log_dir, 'run{}-2'.format(RUN)) 
    LOG_FILE_PATH_2 = os.path.join(checkpoint_dir, 'checkpoint-2-{epoch:02d}-{val_acc:.4f}.hdf5')
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr) ,
                  metrics=['acc'])
    tensorboard = TensorBoard(log_dir=LOG_DIR_2, write_images=True)
    checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH_2, monitor='val_acc', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

    history = model.fit_generator(
          train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=EPOCHS,
          validation_data=validation_generator,
          validation_steps=50,
          verbose=args.verbose,
          callbacks=[tensorboard, checkpoint, early_stopping])
          
    # 测试
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        class_mode='categorical')

    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
    logger.info('test acc:'+str(test_acc))
    
    
    
    logger.info("保存模型")
    model.save(os.path.join(model_dir, 'model.h5'))
    
    #model = keras.models.load_model(os.path.join(model_dir, 'model.h5'))
    export_model(
        model,
        tf_server_dir,
        1
    )
    
    

    
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    
    input_dir = args.input_dir
    training_dir = os.path.join(input_dir, "data","training")
    validation_dir = os.path.join(input_dir, "data","validation")
    testing_dir = os.path.join(input_dir, "data","testing")
    log_dir =  os.path.join(args.output_dir, "log")
    model_dir =  args.model_dir
    tf_server_dir =  os.path.join(model_dir, "tf_server")
    checkpoint_dir =  os.path.join(args.output_dir, "log/checkpoint")
    
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(tf_server_dir):
        os.makedirs(tf_server_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
        
    logger.info("epoch_count:"+str(args.epoch_count))
    logger.info("batch_size:"+str(args.batch_size))
    #logger.info("--------------processing data ----------------- ")
    class_list,training_count = processing_data(training_dir, validation_dir, testing_dir)
    #logger.info("--------------start train --------------------- ")    
    train(training_dir, validation_dir, testing_dir, log_dir, model_dir, tf_server_dir, checkpoint_dir, class_list ,training_count, args)
    logger.info("  训练完成   ")
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument(
        "-e",
        "--epoch_count",
        type=int,
        nargs="?",
        help="Epoch count",
        default=30,
    )
    parser.add_argument(
        "-r",
        "--lr",
        type=float,
        nargs="?",
        help="learning rate (default: 1e-5)",
        default=1e-5,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="Batch size (default: 32)",
        default=32,
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        help="Model保存路径.",
        default="/opt/ml/model/",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="input dir",
        default="/opt/ml/input/",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="outpudif",
        default="/opt/ml/output/",
    )
    parser.add_argument(
        "-g",
        "--gpu_list",
        type=str,
        help="gpu list",
        default="0",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help="log level",
        default=2,
    )
    args = parser.parse_args()
    main(args)
    
