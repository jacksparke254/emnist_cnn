# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:40:24 2023

@author: jacks
"""

#%%
"""
Example of code using the Balanced EMNIST dataset
"""
"BALANCED"
#%%
balanced_cnn = model_pred()
x_train, y_train, x_val, y_val, x_test, y_test = balanced_cnn.importer(location = location, dataset = "balanced", num_classes = 47)
x_train_array, x_val_array = balanced_cnn.img_convert(test = 0)

x_test_array = letters_cnn.img_convert(test = 1)

#%%
#Pretrained vgg16
cnn_history = balanced_cnn.cnn_trained(dataset = "balanced", 
                                       nodes_hidden = 4500, num_classes = 47,
                                       act_function="relu", trainable = 0,
                                       aug=0, steps_per_epoch = 100, 
                                       epochs = 40, validation_steps = 50)

balanced_cnn.epoch_load_weights(epoch_weights="39", act_function="relu", 
                                manual=0, nodes_hidden=4500, num_classes=47, 
                                trainable=0, dataset="balanced")
#-----------------------------------------------------------------------------
# layer 5 trainable 
cnn_history = balanced_cnn.cnn_trained(dataset = "balanced", nodes_hidden = 800,
                                       num_classes = 47, act_function="relu", 
                                       trainable = 1, aug =0,
                                       steps_per_epoch = 100, epochs = 40,
                                       validation_steps = 50)

balanced_cnn.epoch_load_weights(epoch_weights="31", act_function="relu",
                                nodes_hidden=800, num_classes=47,manual=0, 
                                trainable=1, dataset="balanced")

#-----------------------------------------------------------------------------
# All trainable
cnn_history = balanced_cnn.cnn_trained(dataset = "balanced", nodes_hidden = 120,
                                       num_classes = 47, act_function="relu",
                                       trainable = 2,aug=0, 
                                       steps_per_epoch = 100, epochs = 40,
                                       validation_steps = 50)

balanced_cnn.epoch_load_weights(epoch_weights="11", act_function="relu",
                                nodes_hidden=120, num_classes=47,manual=0, 
                                trainable=2, dataset="balanced")

#-----------------------------------------------------------------------------
# With augmentation
cnn_history = balanced_cnn.cnn_trained(dataset = "balanced", nodes_hidden = 6000,
                                       num_classes = 47, act_function="relu", 
                                       trainable = 2, aug = 1, 
                                       steps_per_epoch = 100, epochs = 40,
                                       validation_steps = 50)

balanced_cnn.epoch_load_weights(epoch_weights="35", act_function="relu", aug =1,
                                nodes_hidden=6000, num_classes=47,manual=0,
                                trainable=2, dataset="balanced")

#-----------------------------------------------------------------------------
# Manual
cnn_history = balanced_cnn.cnn_manual(dataset = "balanced", hidden_nodes = 4000, 
                                      num_classes = 47, act_function="relu",
                                      steps_per_epoch = 100, epochs = 40,
                                      validation_steps = 50)

balanced_cnn.epoch_load_weights(epoch_weights="38", manual=1,
                                act_function="relu", nodes_hidden=4000,
                                num_classes=47, trainable=2, dataset="balanced", 
                                aug=0)
    