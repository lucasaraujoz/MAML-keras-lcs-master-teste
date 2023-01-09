# -*- coding: utf-8 -*-
# @File : net.py
# @Author: Runist
# @Time : 2020/7/6 16:52
# @Software: PyCharm
# @Brief: 实现模型分类的网络，MAML与网络结构无关，重点在训练过程

from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np


class MAML:
    def __init__(self, input_shape, num_classes):
        """
        A classe de modelo MAML requer dois modelos, um é o peso θ usado como atualização real
         e o outro é usado para atualizar θ'
        :param input_shape: shape de entrada do modelo
        :param num_classes: numero de classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.meta_model = self.get_maml_model()

    def get_maml_model(self):
        """
        建立maml模型
        :return: maml model
        """
        model = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu",
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Flatten(),
            layers.Dense(self.num_classes, activation='softmax'),
        ])

        return model

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
        """
        PROCESSO DE TREINAMENTO DO MAML
        :param train_data: dados de treinamento, com tarefa como uma unidade
        :param inner_optimizer: O otimizador correspondente ao conjunto de suporte
        :param inner_step: várias etapas para atualização interna
        :param outer_optimizer: O otimizador correspondente ao conjunto de consulta, se o objeto não existir, o gradiente não será atualizado
        :return: batch query loss
        """
        batch_acc = []
        batch_loss = []
        task_weights = []

        # Use meta_weights para salvar o peso inicial e defini-lo como o peso do modelo de passo interno
        meta_weights = self.meta_model.get_weights()

        meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data) #????
        for support_image, support_label in zip(meta_support_image, meta_support_label):

            # Cada tarefa precisa carregar os pesos mais originais para atualização
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step): #epocas internas?
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                    acc = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label, tf.float32)
                    acc = tf.reduce_mean(acc)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # Toda vez que os pesos atualizados pelo loop interno precisam ser salvos uma vez, para garantir que o
            # loop externo atrás dos pesos treine a mesma tarefa
            task_weights.append(self.meta_model.get_weights())



        # *usa o modelo otimizado de cada tarefa* para fazer uma predição.
        # o loss médio dentre as queries de cada tarefa será utilizado para atualizar os pesos do modelo
        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):

                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights[i]) # carrega o peso da tarefa anterior treinada no suporte

                logits = self.meta_model(query_image, training=True)

                # computa o loss médio de treino
                loss = losses.sparse_categorical_crossentropy(query_label, logits) #loss da tarefa no conjunto de consulta
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss) #adiciona o loss dessa tarefa num vetor

                # extrai acurácia média de treino
                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss) # tira a media de todos os loss

        # Independente de ser atualizado ou não, é necessário carregar o peso inicial para atualização para evitar
        # que a etapa val altere o peso original
        self.meta_model.set_weights(meta_weights) # retorna ao peso inicial da rede, ja que cada tarefa qnd treinada alterava o self.meta_model e guardava os pesos das redes de cada tarefa em tasks_weights[i]
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss, mean_acc
