import os
import numpy as np
import tensorflow as tf
import keras
import pickle
from dataclasses import dataclass


@dataclass
class ResNetConfig:
    # Architecture ResNet
    num_res_blocks: int = 6
    num_filters: int = 64
    kernel_size: int = 3

    # Policy head
    policy_filters: int = 2
    policy_hidden_size: int = 128

    # Value head
    value_filters: int = 1
    value_hidden_size: int = 128

    # Regularization
    l2_regularization: float = 2e-4
    use_batch_norm: bool = True

    # Training
    learning_rate: float = 5e-4
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
        self._predict_fn = None

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
        # Input shape: (13, 6, 30) = (hauteur, largeur, channels)
        model_input = keras.layers.Input(shape=(13, 6, 30))

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

        self._build_predict_fn()

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

        self._build_predict_fn()

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

    def train(self, inputs, value_outputs, policy_outputs, sample_weight=None, epochs=1, verbose=2):
        """
        Entraîne le modèle sur un batch via une boucle manuelle (GradientTape).

        NOTE: on n'utilise plus model.fit() car Keras 3 résout sample_weight de
        façon incohérente sur les modèles fonctionnels multi-sorties nommées
        (outputs=[value_output, policy_output] avec loss={'value_head':..., 'policy_head':...}) :
        un sample_weight en dict provoque un KeyError interne (resolve_path positionnel),
        un sample_weight en liste est rejeté car sa structure ne matche pas celle de y (dict).
        La boucle manuelle évite complètement ce problème.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        value_targets = tf.convert_to_tensor(value_outputs, dtype=tf.float32)
        policy_targets = tf.convert_to_tensor(policy_outputs, dtype=tf.float32)

        if sample_weight is None:
            sample_weight = tf.ones_like(value_targets)
        else:
            sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float32)

        cce = keras.losses.CategoricalCrossentropy(reduction='none')

        last_value_loss = None
        last_policy_loss = None

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                value_pred, policy_pred = self.model(inputs, training=True)
                value_pred = tf.squeeze(value_pred, axis=-1)  # (batch, 1) -> (batch,)

                value_loss_per_sample = tf.square(value_targets - value_pred)
                policy_loss_per_sample = cce(policy_targets, policy_pred)

                weighted_value_loss = tf.reduce_mean(value_loss_per_sample * sample_weight)
                weighted_policy_loss = tf.reduce_mean(policy_loss_per_sample * sample_weight)

                reg_loss = tf.add_n(self.model.losses) if self.model.losses else 0.0

                total_loss = (
                    self.config.value_loss_weight * weighted_value_loss
                    + self.config.policy_loss_weight * weighted_policy_loss
                    + reg_loss
                )

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            last_value_loss = float(weighted_value_loss.numpy())
            last_policy_loss = float(weighted_policy_loss.numpy())

            if verbose:
                print(f"value_loss: {last_value_loss:.4f} - policy_loss: {last_policy_loss:.4f}")

        self.training_loss.append((last_value_loss, last_policy_loss))

        return {'value_head_loss': last_value_loss, 'policy_head_loss': last_policy_loss}

    def _build_predict_fn(self):
        """
        Compile l'appel reseau en graphe TF, trace une seule fois grace a une
        dimension de batch dynamique (None), reutilise ensuite pour n'importe
        quelle taille de batch sans retracage.
        """
        input_shape = self.model.input_shape[1:]  # (14, 6, 5), sans la dim de batch

        @tf.function(input_signature=[tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32)])
        def predict_fn(inputs):
            return self.model(inputs, training=False)

        self._predict_fn = predict_fn

    def __call__(self, inputs):
        """
        Inference : retourne (value, policy)
        """
        if isinstance(inputs, list):
            inputs = np.asarray(inputs, dtype=np.float32)
            value, policy = self._predict_fn(inputs)
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 3:
                value, policy = self._predict_fn(inputs[np.newaxis, :, :, :].astype(np.float32))
                value = value[0, 0]
                policy = policy[0, :]
            elif inputs.ndim == 4:
                value, policy = self._predict_fn(inputs.astype(np.float32))
            else:
                raise ValueError('Input array should have 3 or 4 dimensions.')
        else:
            raise ValueError('Input should be a list or numpy array.')

        value = value.numpy()
        policy = policy.numpy()

        return value, policy