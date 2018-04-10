import shutil as sh
import os
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from data_manager import path_manager as pm


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
        Add an embedding for the projector visualization in tensorboard_utils
        :param emb_var: the 2D float variable (matrix of floats) that contains the embedding
        :param sprite: the path to the optional sprite to be attached to the embedding
        :param sprite_window: the [width, height] of a single thumbnail in the sprite
        """
        if self.__proj_config is None:
            # Format: tensorflow/contrib/tensorboard_utils/plugins/projector/projector_config.proto
            self.__proj_config = projector.ProjectorConfig()
        embedding = self.__proj_config.embeddings.add()
        embedding.tensor_name = emb_var.name
        if sprite is not None:
            embedding.sprite.image_path = pm.resources_path(sprite)
            embedding.sprite.single_image_dim.extend(sprite_window)

        self.__emb_list__.append((emb_var, tf.variables_initializer([emb_var])))

    def get_runnable(self, get_summaries=True, get_embeds=True):
        """
        Get all runnable elements for tensorboard_utils to be run into a session.
        :param get_summaries: decide whether to get all summaries
        :param get_embeds: decide whether to get all embeddings
        :return: a list containing all selected runnables
        """
        ret = []
        if get_summaries and len(self.__summ_list__) > 0:
            ret.append(tf.summary.merge(self.__summ_list__))

        if get_embeds:
            ret += [emb[1] for emb in self.__emb_list__]

        return ret

    def write_embeddings(self, session):
        """
        Embedding variables must be written as checkpoints, this method does the job
        :param session: the session from which the embedding values are to be taken
        """
        projector.visualize_embeddings(TensorBoardManager.__writer, self.__proj_config)
        embedding_saver = tf.train.Saver(var_list=[emb[0] for emb in self.__emb_list__])
        embedding_saver.save(session,
                             os.path.join(TensorBoardManager.__save_path,
                                          TensorBoardManager.__projector_ckpt_name))

    @staticmethod
    def set_path(dir_name):
        """
        Change the path where tensorboard_utils saves all data
        :param dir_name: the new directory destination (inside the default base one)
        """
        TensorBoardManager.__save_path = os.path.join(TensorBoardManager.__def_save_path, dir_name)
        TensorBoardManager.__writer = tf.summary.FileWriter(TensorBoardManager.__save_path)

    @staticmethod
    def write_step(summ, step):
        """
        Save all information about the summaries
        :param summ: the summaries produced by a session running runnables
        :param step: the step to label the summaries
        """
        TensorBoardManager.__writer.add_summary(summ, step)

    @staticmethod
    def write_graph(graph):
        """
        Save the given graph for tensorboard_utils to display it
        :param graph: the graph to be displayed into tensorboard_utils
        """
        TensorBoardManager.__writer.add_graph(graph)

    @staticmethod
    def clear_data():
        """
        Clear the current save path from old stuff
        """
        sh.rmtree(TensorBoardManager.__save_path)
        tf.summary.FileWriter(TensorBoardManager.__save_path)

    @staticmethod
    def get_path():
        return TensorBoardManager.__def_save_path

    __projector_ckpt_name = 'emb_ckpt'
    __def_save_path = pm.resources_path('tbdata')
    __save_path = __def_save_path
    __writer = tf.summary.FileWriter(__save_path)
