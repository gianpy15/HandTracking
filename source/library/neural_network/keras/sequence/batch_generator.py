from keras.utils import Sequence
from data.naming import *
from data.regularization.regularizer import Regularizer
from data.augmentation.data_augmenter import Augmenter
from library.multi_threading.thread_pool_manager import ThreadPoolManager
from threading import Condition
import traceback


class BatchGenerator(Sequence):
    def __init__(self, data_sequence,
                 augmenter: Augmenter = None,
                 regularizer: Regularizer = None):
        """
        Create a Sequence to feed the fit_generator with freshly augmented data every batch
        :param data_sequence: an object providing the raw data through subscription.
                              must implement __getitem__ and __len__.
        :param augmenter: the Augmenter to be applied to data freshly at each epoch,
                          defaults to 0.2 probability and 0.15 variance for each component
        :param regularizer: the regularizer to use to regularize data before feeding them
                            to the outside. Normalization only, by default
        """
        super(BatchGenerator, self).__init__()
        self.epoch = 0
        self.data_sequence = data_sequence
        self.reg = regularizer or Regularizer().normalize()
        self.augmenter = augmenter or Augmenter().shift_hue(.2).shift_sat(.2).shift_val(.2)

        self.batches = [None for _ in range(len(self.data_sequence))]
        self.batches_on_processing = [False for _ in range(len(self.data_sequence))]
        self.batches_ready = [False for _ in range(len(self.data_sequence))]
        self.main_lock = Condition()
        self._schedule_batch_preparation(0)

    def __getitem__(self, index):
        log("Requested index %d/%d" % (index+1, len(self)), level=COMMENTARY)
        log("Processing: %s" % self.batches_on_processing, level=DEBUG)
        log("Ready: %s" % self.batches_ready, level=DEBUG)
        # synchronize
        with self.main_lock:
            # make sure that nobody is processing it just now
            self.main_lock.wait_for(predicate=lambda: not self.batches_on_processing[index])
            if self.batches_ready[index]:
                # ancitipate the next request, because we are smart.
                next = (index + 1) % len(self)
                self._schedule_batch_preparation(next)
                # as soon as we don't change epoch, we may be asked the same batch again,
                # and it should be the same. Don't tick it as not ready.

                # lend the data!
                log("Data %d/%d ready, train: %s valid: %s" % (index+1, len(self),
                                                               np.shape(self.batches[index][0]),
                                                               np.shape(self.batches[index][1])),
                    level=COMMENTARY)
                return self.batches[index]

        # here the batch was not ready and nobody is processing it...
        # so we must do the dirty job, possibly with full resources.
        self._prepare_batch(index)
        with self.main_lock:
            # now we may schedule the next one
            next = (index + 1) % len(self)
            self._schedule_batch_preparation(next)
            # and lend the data, finally!
            log("Data %d/%d computed, train: %s valid: %s" % (index+1, len(self),
                                                              np.shape(self.batches[index][0]),
                                                              np.shape(self.batches[index][1])),
                level=COMMENTARY)
            return self.batches[index]

    def __len__(self):
        return len(self.data_sequence)

    def on_epoch_end(self):
        # the epoch is done! all work done must be redone! ...
        # ... maybe later. W lazy policies.
        print(self.batches_on_processing)
        try:
            with self.main_lock:
                # make sure that nobody is doing useless computation
                assert not any(self.batches_on_processing)
                # and sadly nullify all the work done...
                self.batches_ready = [False for _ in range(len(self))]
                # maybe it would be a good idea to anticipate batch zero
                ThreadPoolManager.get_thread_pool().submit(self._prepare_batch, 0)
        except Exception as e:
            traceback.print_exc()
            log(str(e), level=ERRORS)
            raise e

    def _prepare_batch(self, index):
        log("Preparing batch %d/%d" % (index+1, len(self)), level=DEBUG)
        try:
            self.batches_on_processing[index] = True
            # check everything is okay
            assert not self.batches_ready[index]

            # do your job
            augmented_in_batch = self.augmenter.apply_on_batch(self.data_sequence[index][IN][:])
            target_batch = self.data_sequence[index][TARGET]
            in_batch = self.reg.apply_on_batch(augmented_in_batch)

            # synchronize
            with self.main_lock:
                # say everything is ready
                self.batches[index] = (in_batch, target_batch)
                self.batches_ready[index] = True
                self.batches_on_processing[index] = False
                # wake up those lazy sleepers
                self.main_lock.notify_all()
        except Exception as e:
            traceback.print_exc()
            log(str(e), level=ERRORS)
            raise e
        log("Prepared batch %d/%d" % (index+1, len(self)), level=DEBUG)

    def _schedule_batch_preparation(self, index):
        with self.main_lock:
            if self.batches_ready[index] or self.batches_on_processing[index]:
                return
            self.batches_on_processing[index] = True
            ThreadPoolManager.get_thread_pool().submit(self._prepare_batch, index)


