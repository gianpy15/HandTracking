from keras.utils import Sequence
from data.augmentation.data_augmenter import Augmenter
from data.datasets.reading.dataset_manager import DatasetManager
from library.multi_threading import ThreadPoolManager
from data.regularization.regularizer import Regularizer


class ValidSequence(Sequence):
    def __init__(self, data_generator: DatasetManager,
                 augmenter: Augmenter=None,
                 regularizer: Regularizer=None,
                 use_default_augmentation=False):
        """
        Create a Sequence that feed the fit Generator with new data every batch
        :param data_generator: is the Data set Manager that loads the data (in
                               async way) and returns one batch per epoch
        :param augmenter: is the Augmenter to use for augmenting data, you
                          can also use the default one setting this parameter to None
        :param use_default_augmentation: if True will use a default augmenter if the
                                         param augmenter is set to None. The default one
                                         will augment 20% of the data with 0.15 of variation
        """
        super(ValidSequence, self).__init__()
        self.epoch = 0
        self.data_generator: DatasetManager = data_generator
        self.training_samples = self.data_generator.train_samples
        self.validation_samples = self.data_generator.valid_samples
        self.batch_size = self.data_generator.batch_size
        self.data_generator = None
        self.augmenter = None
        self.augmenting_batch = None
        self.pool = ThreadPoolManager.get_thread_pool()
        self.reg = regularizer or Regularizer().normalize()
        if use_default_augmentation:
            self.augmenter = augmenter or Augmenter().shift_hue(.2).shift_sat(.2).shift_val(.2)

        self.current_batch = self.__augment_and_regularize(self.data_generator.get_validation_batch())

    def __getitem__(self, index):
        batch = self.current_batch
        del self.current_batch
        self.augmenting_batch = self.pool.submit(
            self.__augment_and_regularize(self.data_generator.get_validation_batch()))
        return batch

    def __len__(self):
        return self.data_generator.get_validation_batch_number()

    def on_epoch_end(self):
        self.epoch += 1
        self.current_batch = self.augmenting_batch.result()

    def __augment_data(self, batch):
        if self.augmenter is not None:
            return self.augmenter.apply_on_batch(batch)
        return batch

    def __regularize_data(self, batch):
        return self.reg.apply_on_batch(batch)

    def __augment_and_regularize(self, batch):
        augmented = self.augmenter.apply_on_batch(batch) if self.augmenter is not None else batch
        return self.reg.apply_on_batch(augmented)
