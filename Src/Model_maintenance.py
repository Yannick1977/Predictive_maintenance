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
from sklearn.utils import class_weight
import seaborn as sns
from keras_visualizer import visualizer
from keras import optimizers


class modelMaintenance :

    def __init__(self, num_layers, num_input, num_output, loss, optimizer, metrics) -> None:
            """
            Initializes the ModelMaintenance object.

            Args:
                num_layers (int): The number of layers in the model.
                num_input (int): The number of input features.
                num_output (int): The number of output classes.
                loss (str): The loss function to be used during training.
                optimizer (str): The optimizer to be used during training.
                metrics (list): The evaluation metrics to be used during training.

            Returns:
                None
            """
            self.training_history = None
            self.MODELFILEPATH = '../Model/best_model.h5'
            self.model = self._init_model(num_layers, num_input, num_output, loss, optimizer, metrics)


    def _init_model(self, num_layers, num_input, num_output, loss, optimizer, metrics, activation='relu', dropout=0.2):
        """
        Initialize the model with the specified parameters.

        Args:
            num_layers (int): Number of hidden layers in the model.
            num_input (int): Number of input features.
            num_output (int): Number of output classes.
            loss (str): Loss function to be used during training.
            optimizer (str): Optimization algorithm to be used during training.
            metrics (list): List of evaluation metrics to be used during training.
            activation (str, optional): Activation function for the hidden layers. Defaults to 'relu'.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.

        Returns:
            keras.models.Sequential: The initialized model.
        """
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
        """
        Prints a summary of the model.
        """
        self.model.summary()

    def plot_model(self):
        """
        Plots the model architecture and displays it.

        Parameters:
            None

        Returns:
            None
        """
        visualizer(self.model, file_name='model', file_format='png', view=False)
        img = mpimg.imread('model.png')

        plt.imshow(img)
        plt.show()
        #keras_visualizer.plot_model(self.model, show_shapes=True)
        #keras.utils.plot_model(self.model, show_shapes=True)
    
    def train(self, X_train, y_train, X_test, y_test, batch_size, epochs, verbose=1):
            """
            Trains the model using the provided training data and evaluates it on the provided test data.

            Parameters:
                X_train (numpy.ndarray): The input features for training.
                y_train (numpy.ndarray): The target labels for training (one hot encoding).
                X_test (numpy.ndarray): The input features for testing.
                y_test (numpy.ndarray): The target labels for testing (one hot encoding).
                batch_size (int): The number of samples per gradient update.
                epochs (int): The number of times to iterate over the entire training dataset.
                verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

            Returns:
                None
            """

            y_train_labels = np.argmax(y_train, axis=1)

            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
            class_weights = dict(enumerate(class_weights))
            
            self.training_history = self.model.fit(
                x=X_train,
                y=y_train,
                class_weight=class_weights,
                batch_size=batch_size,
                epochs=epochs, 
                verbose=verbose, 
                validation_data=(X_test, y_test), 
                callbacks=self.callbacks)

    def plot_history(self):
            """
            Plots the training and validation metrics over epochs.
            """
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
        """
        Predicts the output for the given input data.

        Parameters:
            X (array-like): Input data for prediction.

        Returns:
            array-like: Predicted output.
        """
        return self.model.predict(X)
    
    def plot_confusion_matrix(self, y_test, y_pred, name_target):
        """
        Plots the confusion matrix for the predicted labels.

        Parameters:
            y_test (array-like): True labels of the test data.
            y_pred (array-like): Predicted labels of the test data.
            name_target (array-like): Names of the target labels.

        Returns:
            None
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        # Plot non-normalized confusion matrix
        ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt=".0f")
        
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        ax.set_xticklabels(name_target, rotation=45)
        ax.set_yticklabels(name_target, rotation=45)

    def _create_callbacks(self):
            """
            Create and return a list of callbacks for model training.

            Returns:
                list: A list of callbacks including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
            """
            self.ModelCheckpoint = ModelCheckpoint(
                filepath=self.MODELFILEPATH,
                save_best_only=True,
                monitor='val_loss')
            
            self.early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            
            self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)
            
            return [self.ModelCheckpoint, self.early_stopping, self.reduce_lr]
    
    def print_evaluation(self, X_test, y_test, y_pred, verbose):
        """
        Prints the evaluation metrics for the model's performance.

        Parameters:
            X_test (array-like): The input features for testing.
            y_test (array-like): The true labels for testing.
            y_pred (array-like): The predicted labels for testing.

        Returns:
            The average of the evaluation metrics.
        """
        score_details = {}
        results = self.model.evaluate(X_test, y_test, verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            if verbose:
                print(name, ': ', value)
            score_details[name] = value

        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        kappa = cohen_kappa_score(y_test_flat, y_pred_flat > 0.5)
        f1 = f1_score(y_test_flat, y_pred_flat > 0.5, average='micro')
        if verbose:
            print('kappa: ', kappa)
            print('F1_score: ', f1)
        score_details['kappa'] = kappa
        score_details['F1_score'] = f1
        return np.mean([np.mean(results[1:]), f1, kappa]), score_details
    
    def _generer_multiples_de_16(self,n):
        liste = [16]
        for i in range(1, n):
            liste.append(liste[i-1] * 2)
        return liste[::-1]
