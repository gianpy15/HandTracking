import shutil as sh
import os
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from data_manager import path_manager


pm = path_manager.PathManager()


class TensorBoardManager:
    """
    This class is designed to manage easily different summaries in a tensorflow graph
    """

    def __init__(self, scope_name=None):
        self.__summ_list__ = []
        self.current_scope = scope_name
        self.__emb_list__ = []
        self.__proj_config = None

    def add_summary(self, obj, name, scope=None, summ_type_fun=tf.summary.scalar):
        if scope is None:
            scope = self.current_scope
        if scope is not None:
            with tf.name_scope(scope):
                self.__summ_list__ += [summ_type_fun(name, obj)]
        else:
            self.__summ_list__ += [summ_type_fun(name, obj)]

    def add_scalar(self, obj, name, scope=None):
        self.add_summary(obj, name, scope, tf.summary.scalar)

    def add_histogram(self, obj, name, scope=None):
        self.add_summary(obj, name, scope, tf.summary.histogram)

    def add_images(self, imgs, name, scope=None, max_out=3, collections=None, family=None):
        if scope is None:
            scope = self.current_scope
        if scope is not None:
            with tf.name_scope(scope):
                self.__summ_list__ += [tf.summary.image(name, imgs, max_out, collections, family)]
        else:
            self.__summ_list__ += [tf.summary.image(name, imgs, max_out, collections, family)]

    def add_embedding(self, emb_var, sprite=None, sprite_window=None):
        """
        Add an embedding for the projector visualization in tensorboard
        :param emb_var: the 2D float variable that contains the embedding
        :param name: the optional name to give to the embedding
        :param sprite: the path to the optional sprite to be attached to the embedding
        :param sprite_window: the [width, height] of a single thumbnail in the sprite
        """
        if self.__proj_config is None:
            # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
            self.__proj_config = projector.ProjectorConfig()
        # You can add multiple embeddings. Here we add only one.
        embedding = self.__proj_config.embeddings.add()
        embedding.tensor_name = emb_var.name
        if sprite is not None:
            embedding.sprite.image_path = pm.resources_path(sprite)
            embedding.sprite.single_image_dim.extend(sprite_window)

        self.__emb_list__.append((emb_var, tf.variables_initializer([emb_var])))

    def get_runnable(self, get_summaries=True, get_embeds=True):
        ret = []
        if get_summaries and len(self.__summ_list__) > 0:
            ret.append(tf.summary.merge(self.__summ_list__))

        if get_embeds:
            ret += [emb[1] for emb in self.__emb_list__]

        return ret

    def write_embeddings(self, session):
        projector.visualize_embeddings(TensorBoardManager.__writer, self.__proj_config)
        embedding_saver = tf.train.Saver(var_list=[emb[0] for emb in self.__emb_list__])
        embedding_saver.save(session,
                             os.path.join(TensorBoardManager.__save_path,
                                          TensorBoardManager.__projector_ckpt_name))

    @staticmethod
    def set_path(dir_name):
        TensorBoardManager.__save_path = os.path.join(TensorBoardManager.__def_save_path, dir_name)
        TensorBoardManager.__writer = tf.summary.FileWriter(TensorBoardManager.__save_path)

    @staticmethod
    def write_step(summ, step):
        TensorBoardManager.__writer.add_summary(summ, step)

    @staticmethod
    def write_graph(graph):
        TensorBoardManager.__writer.add_graph(graph)

    @staticmethod
    def clear_data():
        sh.rmtree(TensorBoardManager.__save_path)
        tf.summary.FileWriter(TensorBoardManager.__save_path)

    @staticmethod
    def get_path():
        return TensorBoardManager.__def_save_path

    __projector_ckpt_name = 'emb_ckpt'
    __def_save_path = pm.resources_path('tbdata')
    __save_path = __def_save_path
    __writer = tf.summary.FileWriter(__save_path)
