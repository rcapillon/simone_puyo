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
        model_input = keras.layers.Input(shape=(14, 6, 4))

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

    def train(self, inputs, value_outputs, policy_outputs, epochs=1, verbose=2):
        """
        train network on sample batch, stores training loss metrics
        """
        history = self.model.fit(inputs, [value_outputs, policy_outputs], epochs=epochs, verbose=verbose)
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


@dataclass
class ResNetConfig:
    # Architecture ResNet
    num_res_blocks: int = 6
    num_filters: int = 128
    kernel_size: int = 3

    # Policy head
    policy_filters: int = 2
    policy_hidden_size: int = 256

    # Value head
    value_filters: int = 1
    value_hidden_size: int = 256

    # Regularization
    l2_regularization: float = 1e-4
    use_batch_norm: bool = True

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256

    # Loss weights
    value_loss_weight: float = 1.
    policy_loss_weight: float = 1.

    def __post_init__(self):
        pass


class ResNetAgent:
    def __init__(self, name, config=ResNetConfig()):
        self.name = name
        self.config = config

        self.model = None

        self.training_loss = []
        self.test_scores = []

    def _residual_block(self, x, filters, kernel_size, l2_reg):
        """
        Bloc résiduel standard :
        Input → Conv → BN → ReLU → Conv → BN → Add → ReLU
        """
        # Première convolution
        conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            data_format='channels_last'
        )(x)

        if self.config.use_batch_norm:
            conv1 = keras.layers.BatchNormalization()(conv1)

        conv1 = keras.layers.Activation('relu')(conv1)

        # Deuxième convolution
        conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            data_format='channels_last'
        )(conv1)

        if self.config.use_batch_norm:
            conv2 = keras.layers.BatchNormalization()(conv2)

        # Skip connection (addition)
        output = keras.layers.Add()([x, conv2])
        output = keras.layers.Activation('relu')(output)

        return output

    def _build_policy_head(self, x, l2_reg):
        """
        Policy head : Conv → BN → ReLU → Flatten → Dense → Softmax
        """
        # Convolution pour réduire les features
        policy = keras.layers.Conv2D(
            filters=self.config.policy_filters,
            kernel_size=1,  # 1x1 conv pour réduction
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            data_format='channels_last'
        )(x)

        if self.config.use_batch_norm:
            policy = keras.layers.BatchNormalization()(policy)

        policy = keras.layers.Activation('relu')(policy)

        # Flatten
        policy = keras.layers.Flatten()(policy)

        policy = keras.layers.Dense(
            self.config.policy_hidden_size,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(policy)

        # Output layer (22 actions pour Puyo)
        policy_output = keras.layers.Dense(
            22,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='policy_head'
        )(policy)

        return policy_output

    def _build_value_head(self, x, l2_reg):
        """
        Value head : Conv → BN → ReLU → Flatten → Dense → Dense → Tanh
        """
        # Convolution pour réduire les features
        value = keras.layers.Conv2D(
            filters=self.config.value_filters,
            kernel_size=1,  # 1x1 conv
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            data_format='channels_last'
        )(x)

        if self.config.use_batch_norm:
            value = keras.layers.BatchNormalization()(value)

        value = keras.layers.Activation('relu')(value)

        # Flatten
        value = keras.layers.Flatten()(value)

        value = keras.layers.Dense(
            self.config.value_hidden_size,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(value)

        # Output layer (régression, pas d'activation ou tanh)
        value_output = keras.layers.Dense(
            1,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='value_head'
        )(value)

        return value_output

    def build_model(self, summary=False):
        """
        Construit le modèle ResNet complet
        """
        # Input shape: (14, 6, 4) = (hauteur, largeur, channels)
        model_input = keras.layers.Input(shape=(14, 6, 4))

        l2_reg = self.config.l2_regularization

        # ===== INITIAL CONVOLUTION =====
        # Transforme l'input en représentation riche
        x = keras.layers.Conv2D(
            filters=self.config.num_filters,
            kernel_size=self.config.kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            data_format='channels_last'
        )(model_input)

        if self.config.use_batch_norm:
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation('relu')(x)

        # ===== RESIDUAL BLOCKS =====
        for i in range(self.config.num_res_blocks):
            x = self._residual_block(
                x,
                filters=self.config.num_filters,
                kernel_size=self.config.kernel_size,
                l2_reg=l2_reg
            )

        # ===== POLICY HEAD =====
        policy_output = self._build_policy_head(x, l2_reg)

        # ===== VALUE HEAD =====
        value_output = self._build_value_head(x, l2_reg)

        # ===== CREATE MODEL =====
        self.model = keras.models.Model(
            inputs=model_input,
            outputs=[value_output, policy_output]
        )

        # ===== COMPILE =====
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                'value_head': keras.losses.MeanSquaredError(),
                'policy_head': keras.losses.CategoricalCrossentropy()
            },
            loss_weights={'value_head': self.config.value_loss_weight, 'policy_head': self.config.policy_loss_weight}
        )

        if summary:
            self.model.summary()

    def load_model(self, path_to_model_dir, summary=False):
        """
        Charge un modèle sauvegardé
        """
        model_path = os.path.join(path_to_model_dir, str(self.name) + '.keras')
        self.model = keras.models.load_model(model_path)

        # Recompiler avec les bons paramètres
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                'value_head': keras.losses.MeanSquaredError(),
                'policy_head': keras.losses.CategoricalCrossentropy()
            },
            loss_weights={'value_head': self.config.value_loss_weight, 'policy_head': self.config.policy_loss_weight}
        )

        # Charger les historiques
        loss_path = os.path.join(path_to_model_dir, str(self.name) + '_training_loss.pkl')
        scores_path = os.path.join(path_to_model_dir, str(self.name) + '_test_scores.pkl')

        if os.path.exists(loss_path):
            with open(loss_path, 'rb') as f:
                self.training_loss = pickle.load(f)

        if os.path.exists(scores_path):
            with open(scores_path, 'rb') as f:
                self.test_scores = pickle.load(f)

        if summary:
            self.model.summary()

    def save_model(self, path_to_model_dir):
        """
        Sauvegarde le modèle et les métriques
        """
        os.makedirs(path_to_model_dir, exist_ok=True)

        model_path = os.path.join(path_to_model_dir, str(self.name) + '.keras')
        self.model.save(model_path)

        loss_path = os.path.join(path_to_model_dir, str(self.name) + '_training_loss.pkl')
        scores_path = os.path.join(path_to_model_dir, str(self.name) + '_test_scores.pkl')

        with open(loss_path, 'wb') as f:
            pickle.dump(self.training_loss, f)

        with open(scores_path, 'wb') as f:
            pickle.dump(self.test_scores, f)

    def train(self, inputs, value_outputs, policy_outputs, epochs=1, verbose=2):
        """
        Entraîne le modèle sur un batch
        """
        history = self.model.fit(
            inputs,
            {'value_head': value_outputs, 'policy_head': policy_outputs},
            batch_size=self.config.batch_size,
            epochs=epochs,
            verbose=verbose
        )

        # Sauvegarder les losses
        value_loss = history.history['value_head_loss'][-1]
        policy_loss = history.history['policy_head_loss'][-1]
        self.training_loss.append((value_loss, policy_loss))

        return history

    def __call__(self, inputs):
        """
        Inference : retourne (value, policy)
        Compatible avec votre interface actuelle
        """
        if isinstance(inputs, list):
            value, policy = self.model(np.array(inputs))
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 3:
                # Single input: (14, 6, 5) → (1, 14, 6, 5)
                value, policy = self.model(inputs[np.newaxis, :, :, :])
                value = value[0, 0]
                policy = policy[0, :]
            elif inputs.ndim == 4:
                # Batch input
                value, policy = self.model(inputs)
            else:
                raise ValueError('Input array should have 3 or 4 dimensions.')
        else:
            raise ValueError('Input should be a list or numpy array.')

        value = value.numpy()
        policy = policy.numpy()

        return value, policy