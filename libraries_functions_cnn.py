# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:40:47 2023

@author: jacks
"""

#%%
import numpy as np
import idx2numpy
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow import keras
#from keras.applications import VGG16
print ("Keras Verion:",keras.__version__)
import numpy as np
import os
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
location = ""

import matplotlib.pyplot as plt
from PIL import Image 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#%%
class model_pred:
    def importer(self, location, dataset, num_classes):
        x_train = idx2numpy.convert_from_file(location + 'EMNIST-' + dataset + 
                                              '-train-images-idx3-ubyte')
        y_train = idx2numpy.convert_from_file(location + 'EMNIST-' + dataset + 
                                              '-train-labels-idx1-ubyte')
        x_test = idx2numpy.convert_from_file(location + 'EMNIST-' + dataset + 
                                             '-test-images-idx3-ubyte')
        y_test = idx2numpy.convert_from_file(location + 'EMNIST-' + dataset + 
                                             '-test-labels-idx1-ubyte')
        #x_train, x_test = x_train / 255.0, x_test / 255.0
        break_point = (x_train.shape[0] - x_test.shape[0])
        x_val = x_train[break_point:]
        x_train = x_train[:break_point]
        y_val = y_train[break_point:]
        y_train = y_train[:break_point]
        if dataset == "letters":
            y_train = [i - 1 for i in y_train]
            y_train = np.array(y_train, dtype="uint8")
            y_train = tf.keras.utils.to_categorical(y_train,num_classes)
            y_val = [i - 1 for i in y_val]
            y_val = np.array(y_val, dtype="uint8")
            y_val = tf.keras.utils.to_categorical(y_val,num_classes)
            y_test = [i - 1 for i in y_test]
            y_test = np.array(y_test, dtype = "uint8")
            y_test = tf.keras.utils.to_categorical(y_test,num_classes)
        else:
            y_train = tf.keras.utils.to_categorical(y_train,num_classes)
            y_val = tf.keras.utils.to_categorical(y_val,num_classes)
            y_test = tf.keras.utils.to_categorical(y_test,num_classes)
            
        return(x_train, y_train, x_val, y_val, x_test, y_test)
    
    def decision_tree(self):
        global x_train
        global x_test
        global y_train
        global y_test
        nsamples, nx, ny = x_train.shape
        x_train = x_train.reshape((nsamples,nx*ny))
        
        nsamples_test, nx_test, ny_test = x_test.shape
        x_test = x_test.reshape((nsamples_test,nx_test*ny_test))
    
        clf_gini = DecisionTreeClassifier(criterion = "gini",
                                          max_depth=10, min_samples_leaf=5)
      
        clf_gini.fit(x_train, y_train)
      
        y_pred = clf_gini.predict(x_test)
    
        print("Confusion Matrix: ",
            confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
          
        print ("Accuracy : ",
        accuracy_score(y_test,y_pred)*100)

    def img_convert(self, test):
        x_train_list = []
        x_val_list = []

        if test == 1:
            x_test_list = []
            for i in range(0,len(x_val)):
                img = Image.fromarray(x_test[i])
                img = img.convert('RGB')
                img = img.resize((32,32)) 
                img = np.array(img)
                x_test_list.append(img)
            x_test_array = np.array(x_test_list)
        else:
            x_train_list = []
            x_val_list = []
            for i in range(0,len(x_train)):
                img = Image.fromarray(x_train[i])
                img = img.convert('RGB')
                img = img.resize((32,32)) 
                img = np.array(img)
                x_train_list.append(img)
            x_train_array = np.array(x_train_list)
            
            for i in range(0,len(x_val)):
                img = Image.fromarray(x_val[i])
                img = img.convert('RGB')
                img = img.resize((32,32)) 
                img = np.array(img)
                x_val_list.append(img)
            x_val_array = np.array(x_val_list)

        if test == 1:
            return(x_test_array)
        
        return(x_train_array, x_val_array)
    
    def cnn_trained(self, dataset, num_classes, nodes_hidden, act_function, 
                    trainable, aug, steps_per_epoch, epochs, validation_steps):
        
        conv_base = VGG16(weights='imagenet',include_top=False,input_shape=
                          (32, 32, 3))

        conv_base.summary()
        
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(nodes_hidden, activation=act_function))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        if trainable == 1:
            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == 'block5_conv1':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False
        elif trainable == 2:
            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                set_trainable = True
                layer.trainable = True
        else:
            conv_base.trainable = False
        
        if aug == 1:
            checkpoint_filepath = ""
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=False)
            
            train_datagen = ImageDataGenerator(rescale=1./255,
                                               rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')
            train_generator = train_datagen.flow(x_train_array, y_train,
                                                 batch_size=128)
            
            val_datagen = ImageDataGenerator(rescale=1./255)  
            validation_generator = val_datagen.flow(x_val_array, y_val, 
                                                    batch_size=20)
    
            
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.RMSprop(learning_rate=2e-5),
                          metrics=['acc'])
            
            start = time.time()
            cnn_history = model.fit_generator(train_generator,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[model_checkpoint_callback], verbose=2)
            print("Time taken to train the MLP %.1f seconds."%(time.time()-start))

            cnn_history = cnn_history.history
            epochs_plot = range(1, epochs + 1)
            
            plt.plot(epochs_plot, cnn_history['loss'], 'r', label='Training loss')
            plt.plot(epochs_plot, cnn_history['val_loss'], 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.ylim(ymin=0)
            plt.show()
            
            plt.plot(epochs_plot, cnn_history['acc'], 'r', label='Training accuracy')
            plt.plot(epochs_plot, cnn_history['val_acc'], 'b', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
            
            return(cnn_history)

        
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=2e-5),
                      metrics=['acc'])
        model.summary()

        
        checkpoint_filepath = ""
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False)
        
        
        start = time.time()
        cnn_history = model.fit(x_train_array, y_train, 
                                steps_per_epoch=steps_per_epoch,epochs=epochs,
                            validation_data=(x_val_array,y_val),
                            validation_steps=validation_steps, 
                            callbacks=[model_checkpoint_callback], verbose=2)
        print("Time taken to train the MLP %.1f seconds."%(time.time()-start))
        
        cnn_history = cnn_history.history
        epochs_plot = range(1, epochs + 1)
        
        plt.plot(epochs_plot, cnn_history['loss'], 'r', label='Training loss')
        plt.plot(epochs_plot, cnn_history['val_loss'], 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.ylim(ymin=0)
        plt.show()
        
        plt.plot(epochs_plot, cnn_history['acc'], 'r', label='Training accuracy')
        plt.plot(epochs_plot, cnn_history['val_acc'], 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        
        return(cnn_history)
    
    def cnn_manual(self, dataset, hidden_nodes, num_classes, act_function, 
                   steps_per_epoch, epochs, validation_steps):
        model = models.Sequential()
        
        model.add(layers.Conv2D(32, (3, 3), activation= act_function, 
                                input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(64, (3, 3), activation= act_function))
        model.add(layers.MaxPooling2D((2, 2)))
            
        model.add(layers.Conv2D(128, (3, 3), activation= act_function))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(hidden_nodes, activation= act_function))
        model.add(layers.Dense(num_classes, activation= 'softmax'))
        
        checkpoint_filepath = ""
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                        filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_accuracy',
                                        mode='max',
                                        save_best_only=False)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=1e-4),
                      metrics=['acc'])
        
        model.summary()
        
        start = time.time()
        cnn_history = model.fit(x_train_array, y_train, 
                                steps_per_epoch=steps_per_epoch,epochs=epochs,
                            validation_data=(x_val_array,y_val),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint_callback], verbose=2)
        print("Time taken to train the MLP %.1f seconds."%(time.time()-start))
        
        cnn_history = cnn_history.history
        epochs_plot = range(1, epochs + 1)
        
        plt.plot(epochs_plot, cnn_history['loss'], 'r', label='Training loss')
        plt.plot(epochs_plot, cnn_history['val_loss'], 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.ylim(ymin=0)
        plt.show()
        
        plt.plot(epochs_plot, cnn_history['acc'], 'r', label='Training accuracy')
        plt.plot(epochs_plot, cnn_history['val_acc'], 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        
        return(cnn_history)
    
    def epoch_load_weights(self, manual, epoch_weights, act_function, aug,
                       nodes_hidden, num_classes, trainable, dataset):    
        
        if manual == 1:
            model = models.Sequential()
            
            model.add(layers.Conv2D(32, (3, 3), activation= act_function,
                                    input_shape=(32, 32, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            
            model.add(layers.Conv2D(64, (3, 3), activation= act_function))
            model.add(layers.MaxPooling2D((2, 2)))
            
            model.add(layers.Conv2D(128, (3, 3), activation= act_function))
            model.add(layers.MaxPooling2D((2, 2)))
    
            model.add(layers.Flatten())
            model.add(layers.Dense(nodes_hidden, activation= act_function))
            model.add(layers.Dense(num_classes, activation= 'softmax'))
            
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.RMSprop(learning_rate=1e-4),
                          metrics=['acc'])
            
            model.load_weights()
            model.evaluate(x_test_array, y_test) 
            model.summary()
            return
        
        conv_base = VGG16(weights='imagenet',include_top=False,
                          input_shape=(32, 32, 3))
    
        conv_base.summary()
        
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(nodes_hidden, activation=act_function))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        if trainable == 1:
            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == 'block5_conv1':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False
        elif trainable == 2:
            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                set_trainable = True
                layer.trainable = True
        else:
            conv_base.trainable = False   
        
        if aug == 1:
            test_datagen = ImageDataGenerator(rescale=1./255)  
            test_generator = test_datagen.flow(x_test_array, y_test, 
                                                        batch_size=20)
            
            model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['acc'])
            
            model.load_weights()
            model.evaluate(test_generator) 
            model.summary()
            return
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=2e-5),
                      metrics=['acc'])
        
        model.load_weights()
        model.evaluate(x_test_array, y_test) 
        model.summary()
        