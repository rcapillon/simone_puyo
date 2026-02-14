import os
import numpy as np
import keras
import pickle
from dataclasses import dataclass


@dataclass
class MLPConfig:
    """
    Dataclass for configuring the Multilayer Perceptron agent
    """
    n_common_hidden_layers: int = 1
    n_common_neurons_per_layer: int = 256

    n_value_hidden_layers: int = 1
    n_value_neurons_per_layer: int = 256

    n_policy_hidden_layers: int = 1
    n_policy_neurons_per_layer: int = 256

    learning_rate: float = 1e-3
    batch_size: int = 32

    def __post_init__(self):
        pass


class MLPAgent:
    """
    Class for Multilayer Perceptron agent
    """
    def __init__(self, name, config=MLPConfig()):
        self.name = name
        self.config = config

        self.model = None

        self.training_loss = []
        self.test_scores = []

    def build_model(self, summary=False):
        """
        builds and compiles the neural network
        """
        model_input = keras.layers.Input(shape=(14, 6, 5))

        output = keras.layers.Flatten(data_format='channels_last')(model_input)
        for i in range(self.config.n_common_hidden_layers):
            output = keras.layers.Dense(self.config.n_common_neurons_per_layer)(output)
            output = keras.activations.relu(output)

        value_output = keras.layers.Dense(self.config.n_value_neurons_per_layer)(output)
        value_output = keras.activations.relu(value_output)
        for i in range(1, self.config.n_value_hidden_layers):
            value_output = keras.layers.Dense(self.config.n_value_neurons_per_layer)(value_output)
            value_output = keras.activations.relu(value_output)
        value_output = keras.layers.Dense(1, name='value_head')(value_output)

        policy_output = keras.layers.Dense(self.config.n_policy_neurons_per_layer)(output)
        policy_output = keras.activations.relu(policy_output)
        for i in range(1, self.config.n_policy_hidden_layers):
            policy_output = keras.layers.Dense(self.config.n_policy_neurons_per_layer)(policy_output)
            policy_output = keras.activations.relu(policy_output)
        policy_output = keras.layers.Dense(22, activation=keras.activations.softmax, name='policy_head')(policy_output)

        self.model = keras.models.Model(inputs=model_input, outputs=[value_output, policy_output])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()]
        )

        if summary:
            self.model.summary()

    def load_model(self, path_to_model_dir, summary=False):
        """
        load neural network from keras file as well as training loss and test metrics
        """
        self.model = keras.models.load_model(os.path.join(path_to_model_dir, str(self.name) + '.keras'))

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()]
        )

        with open(os.path.join(path_to_model_dir, str(self.name) + '_training_loss.pkl'), 'rb') as f1:
            self.training_loss = pickle.load(f1)
        with open(os.path.join(path_to_model_dir, str(self.name) + '_test_scores.pkl'), 'rb') as f2:
            self.test_scores = pickle.load(f2)

        if summary:
            self.model.summary()

    def save_model(self, path_to_model_dir):
        """
        save neural network to keras file, as well as training loss and test metrics
        """
        self.model.save(os.path.join(path_to_model_dir, str(self.name) + '.keras'))

        with open(os.path.join(path_to_model_dir, str(self.name) + '_training_loss.pkl'), 'wb') as f1:
            pickle.dump(self.training_loss, f1)
        with open(os.path.join(path_to_model_dir, str(self.name) + '_test_scores.pkl'), 'wb') as f2:
            pickle.dump(self.test_scores, f2)

    def train(self, inputs, value_outputs, policy_outputs):
        """
        train network on sample batch, stores training loss metrics
        """
        history = self.model.fit(inputs, [value_outputs, policy_outputs], verbose=2)
        self.training_loss.append((history.history['value_head_loss'], history.history['policy_head_loss']))

    def __call__(self, inputs):
        """
        predicts value and policy from inputs
        """
        if isinstance(inputs, list):
            value, policy = self.model(np.array(inputs))
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 3:
                value, policy = self.model(inputs[np.newaxis, :, :, :])
                value = value[0, 0]
                policy = policy[0, :]
            elif inputs.ndim == 4:
                value, policy = self.model(inputs)
            else:
                raise ValueError('Input array should have 3 or 4 dimensions.')
        else:
            raise ValueError('Input should be a list or numpy array.')

        value = value.numpy()
        policy = policy.numpy()

        return value, policy
