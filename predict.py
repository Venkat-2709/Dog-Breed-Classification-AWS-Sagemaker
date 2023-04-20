import tensorflow as tf
import argparse

def net():

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(args.units, activation='relu')(x)
    x = layers.Dropout(0.2)(x)                  
    x = layers.Dense(133, activation='softmax')(x)           

    model = Model(base_model.input, x) 

def model_fn(model_dir):
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()
    
    model = tf.saved_model.load(os.path.join(args.sm_model_dir, "000000001"))
    
    return model

def input_fn(input_data, content_type):
    if content_type == 'application/x-npy':
        input_data = np.load(input_data)
    else:
        raise ValueError('Unsupported content type: {}'.format(content_type))
        
    input_data = tf.keras.applications.inception_v3.preprocess_input(input_data)

    return input_data

def predict_fn(input_data, model):
    output_data = model.predict(input_data)
    
    return output_data