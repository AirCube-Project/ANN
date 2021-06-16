import time
import tensorflow as tf
from tcn import TCN
from tensorflow import keras
# import scripts from other folders
import os
import sys

# sys.path.append('../')
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization, Flatten

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from nn_base.nn_base_model_har import BaseModel

# Main Application directory
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))


class HarModel(BaseModel):

    def __init__(self, config, dataset):
        """
        Constructor to initialize the ConvNet for PAT wearables' sensors dataset.
        :param config: the JSON configuration namespace.
        :param dataset: Training and testing datasets.
        :return none
        :raises none
        """

        super().__init__(config, dataset)
        return

    def define_model(self):
        """
        Construct the ConvNet model.
        :param none
        :return none
        :raises none
        """

        self.cnn_model = tf.keras.models.Sequential()
        n_timesteps, n_features = self.dataset.train_data.shape[1], self.dataset.train_data.shape[2],
        self.cnn_model.add(
            TCN(nb_stacks=3, nb_filters=128, kernel_size=3, input_shape=(n_timesteps, n_features), use_batch_norm=True,
                dropout_rate=0.2, name="TCN"))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(128, activation='relu'))
        self.cnn_model.add(Dropout(0.3))
        self.cnn_model.add(BatchNormalization())
        self.cnn_model.add(keras.layers.Dense(self.dataset.no_of_classes,
                                              activation=self.config.config_namespace.oned_dense_activation_l2, name="output")
                           )
        model_design_name = 'model_design_{}.png'.format(self.config.config_namespace.model_type)
        # Summary of the ConvNet model.
        print('Summary of the model:')
        self.cnn_model.summary()

        # save the model design
        model_design_path = os.path.join(main_app_path, self.config.config_namespace.image_dir, model_design_name)
        keras.utils.plot_model(self.cnn_model, model_design_path, show_shapes=True)
        return

    def define_functional_model(self):
        """
        Define (construct) a functional ConvNet model.
        :param none
        :return cnn_model: The ConvNet sequential model.
        :raises none
        """

        print("yet to be implemented\n")
        return

    def compile_model(self):
        """
        Configure the ConvNet model.
        :param none
        :return none
        :raises none
        """

        self.cnn_model.compile(loss=self.config.config_namespace.compile_loss,
                               optimizer=self.config.config_namespace.compile_optimizer,
                               metrics=[self.config.config_namespace.compile_metrics1]
                               )

    def fit_model(self):
        """
        Train the ConvNet model.
        :param none
        :return none
        :raises none
        """

        start_time = time.time()

        if self.config.config_namespace.validation_split == True:
            print(
                'Training phase uses "validation_plit" parameter with a ratio retrieved from the configuration file."')
            if (self.config.config_namespace.save_model == "true"):
                print("Training phase under progress, trained ConvNet model will be saved at path",
                      self.saved_model_path,
                      " ...\n")
                print("MODEL FIT")
                print(f"Epochs is {self.config.config_namespace.num_epochs}")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=20,
                                                  callbacks=self.callbacks_list_sm,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_split=self.config.config_namespace.validation_split_ratio
                                                  )
            else:
                print("Training phase under progress ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=20,
                                                  callbacks=self.callbacks_list_es,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_split=self.config.config_namespace.validation_split_ratio
                                                  )

        elif self.config.config_namespace.validation_split == False:
            print('Training phase uses "validation_data" parameter, and handles the test data."')
            if (self.config.config_namespace.save_model == "true"):
                print("Training phase under progress, trained ConvNet model will be saved at path",
                      self.saved_model_path,
                      " ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=self.config.config_namespace.num_epochs,
                                                  callbacks=self.callbacks_list,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_data=(
                                                      self.dataset.test_data, self.dataset.test_label_one_hot)
                                                  )
            else:
                print("Training phase under progress ...\n")
                self.history = self.cnn_model.fit(x=self.dataset.train_data,
                                                  y=self.dataset.train_label_one_hot,
                                                  batch_size=self.config.config_namespace.batch_size,
                                                  epochs=self.config.config_namespace.num_epochs,
                                                  verbose=self.config.config_namespace.fit_verbose,
                                                  validation_data=(
                                                      self.dataset.test_data, self.dataset.test_label_one_hot)
                                                  )
        else:
            print(
                'Not valid value is declared for the validation process. Set <true> for "validation_split" or <false> for "validation_data".')

        end_time = time.time()

        self.train_time = end_time - start_time
        print("The model took %0.3f seconds to train.\n" % self.train_time)

        return

    def evaluate_model(self):
        """
        Evaluate the ConvNet model.
        :param none
        :return none
        :raises none
        """

        self.scores = self.cnn_model.evaluate(x=self.dataset.test_data,
                                              y=self.dataset.test_label_one_hot,
                                              verbose=self.config.config_namespace.evaluate_verbose
                                              )

        print("Test loss: ", self.scores[0])
        print("Test accuracy: ", self.scores[1])

        return

    def predict(self):
        """
        Predict the class labels of testing dataset.
        :param none
        :return none
        :raises none
        """

        self.predictions = self.cnn_model.predict(x=self.dataset.test_data,
                                                  verbose=self.config.config_namespace.predict_verbose
                                                  )

        return
