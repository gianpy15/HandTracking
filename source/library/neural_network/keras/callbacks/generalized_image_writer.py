from keras.callbacks import Callback
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
from data import *
import numpy as np
from library.multi_threading import ThreadPoolManager
import traceback


class ImageWriter(Callback):
    """
    Manage the tensorboard asynchronous logging of images through a callback
    """
    def __init__(self, data_sequence, image_generators: dict,
                 max_items=5, name='images', freq=1):
        """
        :param data_sequence: the sequence of data in form of batch of dictionaries. Must be subscriptable.
        :param image_generators: dictionary of {name: function} to specify what images should be plotted.
                                name: the name suffix in the tensorboard view
                                function: a callable that takes one argument feed:
                                        feed: a dictionary containing all the entries of the network
                                              using naming conventions. Available keys:
                                              IN(.)         all network inputs
                                              OUT(.)        all network target outputs
                                              NET_OUT(.)    all network corresponding real outputs
                                        any extra key in the data_sequence will be added to feed as well.
        :param max_items: maximum number of images to be plotted
        :param name: name prefix to the plotted images
        :param freq: specify how often the plot should be performed. It will be done once every freq epochs.
        """
        self.tb_manager = TBManager()
        super(ImageWriter, self).__init__()
        self.basename = name
        self.freq = freq
        self.max_items = max_items

        self.image_generators = image_generators or {}

        # Build our data pool from actual data
        self.datapool = {}
        for k in data_sequence[0]:
            self.datapool[k] = data_sequence[0][k][0:max_items]

        self.placeholders = {}

    def on_train_begin(self, logs=None):
        # to initialize placeholders we need to know data type and dimension of all image outputs
        # so we need to execute generators once
        generator_feed = {}
        # Copy each key content to the feed dictionary for generators
        for k in self.datapool:
            generator_feed[k] = self.datapool[k]
        # Fake the model real output with its targets instead
        for k in OUT.reverse(OUT.filter(self.datapool.keys())):
            generator_feed[NET_OUT[k]] = self.datapool[OUT[k]]
        # Now we can execute generators and setup the correct datatypes and shapes for their placeholders
        for name in self.image_generators:
            sample_out = self.image_generators[name](generator_feed)
            self.placeholders[name] = tf.placeholder(dtype=sample_out.dtype,
                                                     shape=np.shape(sample_out))
            self.tb_manager.add_images(imgs=self.placeholders[name],
                                       name=self.basename+'_'+name,
                                       max_out=self.max_items)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            # It's time to plot. Prepare the feed for generators:
            generator_feed = {}
            # Copy each key content
            for k in self.datapool:
                generator_feed[k] = self.datapool[k]
            # Compute real outputs of the model
            net_outputs = self.model.predict({name: self.datapool[name] for name in IN.filter(self.datapool.keys())})

            output_num = len(self.model.output_layers)
            # It appears that if the model has one single output, the model.predict will not use it as extra dimension
            # But we need it, then fix:
            if output_num == 1:
                net_outputs = np.expand_dims(net_outputs, axis=0)
            # Reassign outputs to their respective name in the generator feed
            for idx in range(output_num):
                name = OUT.reverse(self.model.output_layers[idx].name)
                generator_feed[NET_OUT[name]] = net_outputs[idx]
            # Schedule computation and summary writing!
            ThreadPoolManager.get_thread_pool().submit(self.__write_step, generator_feed, epoch, tf.get_default_graph())

    def __write_step(self, generator_feeds, epoch, cur_graph):
        feed_dict = {}
        try:
            for k in self.placeholders:
                feed_dict[self.placeholders[k]] = self.image_generators[k](generator_feeds)
            with tf.Session(graph=cur_graph) as s:
                summary = s.run(self.tb_manager.get_runnable(),
                                feed_dict=feed_dict)[0]
                self.tb_manager.write_step(summary, epoch)
        except Exception as e:
            traceback.print_exc()
            raise e
