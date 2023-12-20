import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.losses import CategoricalCrossentropy
#from keras import optimizers
#from keras.optimizer_v2.adam import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2, l1
from keras.metrics import Accuracy, SparseCategoricalAccuracy, CategoricalAccuracy, AUC, \
    PrecisionAtRecall, Recall, Precision
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
import seaborn as sns
from keras_visualizer import visualizer
from keras import optimizers


class modelMaintenance :

    def __init__(self, num_layers, num_input, num_output, loss, optimizer, metrics) -> None:
        self.training_history = None
        self.MODELFILEPATH = '../Model/best_model.h5'
        self.model = self._init_model(num_layers, num_input, num_output, loss, optimizer, metrics)


    def _init_model(self, num_layers, num_input, num_output, loss, optimizer, metrics, activation='relu', dropout=0.2):
        model = Sequential()
        model.add(Input(shape=(num_input,)))

        lst_num_neurons = self._generer_multiples_de_16(num_layers)
        for n in lst_num_neurons:
            model.add(Dense(n, activation=activation))
            model.add(Dropout(dropout))

        model.add(Dense(num_output, activation='softmax'))

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)
        self.callbacks = self._create_callbacks()
        return model
    
    def summary(self):
        self.model.summary()

    def plot_model(self):
        visualizer(self.model,file_name='model', file_format='png', view=False)
        img = mpimg.imread('model.png')

        plt.imshow(img)
        plt.show()
        #keras_visualizer.plot_model(self.model, show_shapes=True)
        #keras.utils.plot_model(self.model, show_shapes=True)
    
    def train(self, X_train, y_train, X_test, y_test, batch_size, epochs, verbose=1):
        
        self.training_history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs, 
            verbose=verbose, 
            validation_data=(X_test, y_test), 
            callbacks=self.callbacks)

    def plot_history(self):
        list_metrics = []
    
        for st in self.training_history.history.keys():
            if not st.startswith('val') and not st.__contains__('lr'):
                list_metrics.append(st)

        fig, axs = plt.subplots(len(list_metrics), figsize=(8,12))
        fig.tight_layout(pad=3.0)
    
        for i, metrics in enumerate(list_metrics):
            length = len(self.training_history.history[metrics])
            axs[i].plot(np.arange(length), self.training_history.history[metrics], label='train '+metrics)
            axs[i].plot(np.arange(length), self.training_history.history['val_'+metrics], label='test '+metrics)
            axs[i].set_title(metrics)
            axs[i].legend()
        
        plt.show()

    def predict(self, X):
        return self.model.predict(X)
    
    def plot_confusion_matrix(self, y_test, y_pred, name_target):

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        # Plot non-normalized confusion matrix
        ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt=".0f")
        
        #ax.set_xlabel('Label de l\'axe X')
        #ax.set_ylabel('Label de l\'axe Y')

        ax.set_xticklabels(name_target, rotation=45)
        ax.set_yticklabels(name_target, rotation=45)

    def _create_callbacks(self):
        self.ModelCheckpoint = ModelCheckpoint(
            filepath=self.MODELFILEPATH,
            save_best_only=True,
            monitor='val_loss')
        
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)
        
        return [self.ModelCheckpoint, self.early_stopping, self.reduce_lr]
    
    def print_evaluation(self, X_test, y_test, y_pred):
        """
        Affiche les metriques du modele sur le jeu de donne de test
        """
        results = self.model.evaluate(X_test, y_test, verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print(name, ': ', value)

        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        kappa = cohen_kappa_score(y_test_flat, y_pred_flat > 0.5)
        f1 = f1_score(y_test_flat, y_pred_flat > 0.5, average='micro')
        print('kappa: ', kappa)
        print('F1_score: ', f1)
        return np.mean([np.mean(results[1:]), f1, kappa])
    
    def _generer_multiples_de_16(self,n):
        liste = [16]
        for i in range(1, n):
            liste.append(liste[i-1] * 2)
        return liste[::-1]
