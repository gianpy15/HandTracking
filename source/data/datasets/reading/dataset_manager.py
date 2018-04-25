from data.naming import *
import math
import threading
from data.datasets.reading.dataset_separator import DatasetSeparator
from data.datasets.reading.general_reading import read_formatted_batch
from library.multi_threading.thread_pool_manager import ThreadPoolManager


class DatasetManager:
    def __init__(self, train_samples, valid_samples, batch_size, dataset_dir, formatting):
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.formatting = formatting
        ThreadPoolManager.get_thread_pool().submit(fn=self.__separate_and_load)
        self.train_batch_number = None
        self.valid_batch_number = None
        self.train_availability = 0
        self.valid_availability = 0
        self.current_train_batch_index = 0
        self.current_valid_batch_index = 0
        self.traindata = None
        self.validdata = None
        self.main_lock = threading.Condition()

    def __separate_and_load(self):
        separator = DatasetSeparator(self.dataset_dir)
        trainframes, validframes = separator.select_train_validation_framelists(self.train_samples,
                                                                                self.valid_samples)
        train_avail_tot = len(trainframes)
        valid_avail_tot = len(validframes)

        self.train_batch_number = int(math.ceil(train_avail_tot / self.batch_size))
        self.valid_batch_number = int(math.ceil(valid_avail_tot / self.batch_size))
        self.traindata = [None for _ in range(self.train_batch_number)]
        self.validdata = [None for _ in range(self.valid_batch_number)]
        end = 0
        for idx in range(self.train_batch_number):
            start = end
            end = (idx + 1) * self.batch_size
            data = read_formatted_batch(frames=trainframes[start:end],
                                        formatdict=self.formatting)
            self.main_lock.acquire()
            self.traindata[idx] = data
            self.main_lock.notify_all()
            self.main_lock.release()

        end = 0
        for idx in range(self.valid_batch_number):
            start = end
            end = (idx + 1) * self.batch_size
            data = read_formatted_batch(frames=validframes[start:end],
                                        formatdict=self.formatting)
            self.main_lock.acquire()
            self.validdata[idx] = data
            self.main_lock.notify_all()
            self.main_lock.release()

    def get_training_batch(self, index=None, blocking=True):
        index = self.current_train_batch_index if index is None else index
        if not blocking:
            if self.traindata is None or self.traindata[index] is None:
                return None
            return self.traindata[index]

        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.traindata is not None)
        index = min(index, self.train_batch_number-1)
        self.main_lock.wait_for(predicate=lambda: self.traindata[index] is not None)
        self.main_lock.release()
        self.current_train_batch_index = (index + 1) % self.train_batch_number
        return self.traindata[index]

    def get_training_batch_number(self, blocking=True):
        if not blocking:
            return self.train_batch_number
        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.traindata is not None)
        self.main_lock.release()
        return self.train_batch_number

    def get_validation_batch(self, index=None, blocking=True):
        index = self.current_valid_batch_index if index is None else index
        if not blocking:
            if self.validdata is None or self.validdata[index] is None:
                return None
            return self.validdata[index]

        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.validdata is not None)
        index = min(index, self.valid_batch_number-1)
        self.main_lock.wait_for(predicate=lambda: self.validdata[index] is not None)
        self.main_lock.release()
        self.current_valid_batch_index = (index + 1) % self.valid_batch_number
        return self.validdata[index]

    def get_validation_batch_number(self, blocking=True):
        if not blocking:
            return self.valid_batch_number
        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.validdata is not None)
        self.main_lock.release()
        return self.valid_batch_number


if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show
    dm = DatasetManager(train_samples=5000,
                        valid_samples=4200,
                        batch_size=2,
                        dataset_dir=crops_path(),
                        formatting=CROPS_STD_FORMAT)
    print('starting ds reading...')
    batches = []
    for idx in range(dm.get_training_batch_number()):
        print('getting training batch %d' % idx)
        batches.append(dm.get_training_batch())
        print('got training batch %d!' % idx)

    for idx in range(dm.get_validation_batch_number()):
        print('getting valid batch %d' % idx)
        batches.append(dm.get_validation_batch())
        print('got valid batch %d!' % idx)

    for batch in batches[:3]:
        for img in batch[IN]:
            imshow(img)
            show()


