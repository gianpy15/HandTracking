# This class manages the file loading of metagraphs and checkpoints
# It can manage either both or one of them
import tensorflow as tf


class ModelLoader:
    def __init__(self, meta=None, checkpoint=None, **assocs):
        if meta is not None:
            self.__saver = tf.train.import_meta_graph(meta, clear_devices=True)
        else:
            self.__saver = tf.train.Saver()

        self.__checkpoint = checkpoint
        self.__assocs = assocs

    def get_tensor(self, tensor):
        return tf.get_default_graph().get_tensor_by_name(self.__assocs[tensor])

    def get_tensor_name(self, tensor_nickname):
        return self.__assocs[tensor_nickname]

    def load_checkpoint(self, session):
        if self.__checkpoint is not None:
            self.__saver.restore(session, self.__checkpoint)