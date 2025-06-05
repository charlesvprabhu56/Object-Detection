import cv2 as cv
import numpy as np
from PIL import Image
from keras.applications import MobileNet
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import activations


def Model_MobileNet(train_data, train_target, test_data,test_target, epoch):
    model = MobileNet(weights='imagenet')
    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    for i in range(Data.shape[0]):
        data = Image.fromarray(np.uint8(Data[i])).convert('RGB')
        data = image.img_to_array(data)
        data = np.expand_dims(data, axis=0)
        data = np.squeeze(data)
        Data[i] = cv.resize(data, (224, 224))
        Data[i] = preprocess_input(Data[i])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=train_target,
                        steps_per_epoch=2,
                        epochs=epoch)
    preds = model.predict(test_data)
    Eval = activations.relu(test_target).numpy()
    pred = decode_predictions(preds, top=3)[0]
    return Eval, pred
