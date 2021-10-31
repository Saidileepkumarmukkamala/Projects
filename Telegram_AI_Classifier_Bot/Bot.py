from telegram.ext import * # library for telegram (pip install python-telegram-bot)
from io import BytesIO
import cv2 
import numpy as np
import tensorflow as tf


(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test = x_train/255 , x_test/255 #Normalizing

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck'] # classes that data contains

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

def start(update , context): # function for start command
    update.message.reply_text('Welcome to the Super AI Image Classifier Bot !')

def help(update , context): # function for help command
    update.message.reply_text("""
    /start = Will starts conversation
    /help = shows this message
    /train = trains neural network
    """)

def train(update , context): # function for train command
    update.message.reply_text("Model is Now being trained...... It May take some time .\n\nI will let you know once the process is finished :)")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
    model.save('classifier.model')
    update.message.reply_text("Done ! Model is now Ready , You can now send photo")

def handle_message(update,context): # function to avoid any input text messages
    update.message.reply_text('Sorry !! No text is allowed \n\nPlease train the model before sending a picture !')

def handle_photo(update,context):  # function to handle imput picture files
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()),dtype=np.uint8)


    img = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)


    prediction = model.predict(np.array([img/255]))
    update.message.reply_text(f'In this image .... I can see a {class_names[np.argmax(prediction)]}')


# Main function
updater = Updater('YOUR TOKEN HERE' , use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start",start))
dp.add_handler(CommandHandler("help",help))
dp.add_handler(CommandHandler("train",train))
dp.add_handler(MessageHandler(Filters.text,handle_message))
dp.add_handler(MessageHandler(Filters.photo,handle_photo))

updater.start_polling()
updater.idle()

