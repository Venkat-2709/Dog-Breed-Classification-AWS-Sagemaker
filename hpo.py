#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import tensorflow as tf
import os
import json
import argparse

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
def net(train_data, valid_data, loss_criterion, optimizer, args):
    
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(args.units, activation='relu')(x)
    x = layers.Dropout(0.2)(x)                  
    x = layers.Dense(133, activation='softmax')(x)           

    model = Model(base_model.input, x) 
    
    model.compile(optimizer=optimizer, loss=loss_criterion, metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    checkpoint = ModelCheckpoint('/opt/ml/model/model-{epoch:03d}.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    
    model.fit(train_data, epochs=args.epochs, validation_data=valid_data, callbacks=[early_stop, checkpoint])

    return model

def data_loaders(base_dir):
    
    train_dir = '/opt/ml/input/data/train'
    validation_dir = '/opt/ml/input/data/validation'
    
    train_generator = ImageDataGenerator(rescale=1./255)
    
    valid_generator = ImageDataGenerator(rescale=1./255)
    
    
    train_data = train_generator.flow_from_directory(directory=train_dir, batch_size=32, target_size=(224, 224), class_mode='categorical')
    valid_data = valid_generator.flow_from_directory(directory=validation_dir, batch_size=32, target_size=(224, 224), class_mode='categorical')
    
    return train_data, valid_data
    

def main(args):
    
    train_data, valid_data = data_loaders(args.train)
    
    loss_criterion = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)
    
    image_classifier = net(train_data, valid_data, loss_criterion, optimizer, args)
    
    if args.current_host == args.hosts[0]:
        image_classifier.save(os.path.join(args.sm_model_dir, "000000001"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        "--units",
        type=int,
        default=64,
        metavar="N",
        help="input units for training (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float, 
        default=0.01, 
        metavar="LR", 
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
            
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    args=parser.parse_args()
    
    main(args)
