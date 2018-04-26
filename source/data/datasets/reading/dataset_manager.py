from data.naming import *
import math
import threading
from data.datasets.reading.dataset_separator import DatasetSeparator
from data.datasets.reading.general_reading import read_formatted_batch
from library.multi_threading.thread_pool_manager import ThreadPoolManager
import traceback


class DatasetManager:
    class _DataSequence:
        def __init__(self, getitem, lenf):
            self.getitemf = getitem
            self.lenf = lenf

        def __getitem__(self, item):
            return self.getitemf(item)

        def __len__(self):
            return self.lenf()

        def __iter__(self):
            for idx in range(len(self)):
                yield self[idx]

    def __init__(self, train_samples, valid_samples, batch_size, dataset_dir, formatting, exclude_videos=None):
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.formatting = formatting
        self.exclude_videos = exclude_videos or []
        ThreadPoolManager.get_thread_pool().submit(fn=self.__separate_and_load)
        self.train_batch_number = None
        self.valid_batch_number = None
        self.current_train_batch_index = 0
        self.current_valid_batch_index = 0
        self.batchdata = None
        self.main_lock = threading.Condition()
        self.loading_done = False
        self.trainframes = None
        self.validframes = None
        self.urgent_queue = []

    def __separate_and_load(self):
        separator = DatasetSeparator(self.dataset_dir)
        separator.exclude_videos(self.exclude_videos)
        try:
            self.trainframes, self.validframes = separator.select_train_validation_framelists(self.train_samples,
                                                                                    self.valid_samples)
        except Exception as e:
            traceback.print_exc()
            log(str(e), level=ERRORS)
            raise e
        train_avail_tot = len(self.trainframes)
        valid_avail_tot = len(self.validframes)
        self.train_batch_number = int(math.ceil(train_avail_tot / self.batch_size))
        self.valid_batch_number = int(math.ceil(valid_avail_tot / self.batch_size))
        self.batchdata = [None for _ in range(self.train_batch_number + self.valid_batch_number)]
        while not self.loading_done:
            idx, frames = self.__get_next_batch()
            if idx is None:
                break
            assert self.batchdata[idx] is None
            log("DATA LOADING WORKER: loading %s batch %d" % ("train" if idx < self.train_batch_number
                                                              else "valid",
                                                              idx if idx < self.train_batch_number
                                                              else idx - self.train_batch_number),
                level=DEBUG)
            try:
                data = read_formatted_batch(frames=frames,
                                            formatdict=self.formatting)
            except Exception as e:
                traceback.print_exc()
                log(str(e), level=ERRORS)
                raise e
            self.main_lock.acquire()
            self.batchdata[idx] = data
            self.main_lock.notify_all()
            self.main_lock.release()
        log("DATA LOADING WORKER: quitting")

    def __get_next_batch(self):
        assert self.batchdata is not None
        index = None
        # try to determine next index to process by popping the urgent queue first
        while index is None and len(self.urgent_queue) > 0:
            index = self.urgent_queue.pop()
            if self.batchdata[index] is not None:
                index = None
        # if no urgent batch has been requested, pick the first uncompleted one
        if index is None:
            for tidx in range(len(self.batchdata)):
                if self.batchdata[tidx] is None:
                    index = tidx
                    break
        # if no batch is uncompleted, we are done
        if index is None:
            self.loading_done = True
            return None, None
        # first we have train batches, if index is in the train range:
        if index < self.train_batch_number:
            start = self.batch_size * index
            end = start + self.batch_size
            return index, self.trainframes[start:end]
        # else we are in the validation batches section:
        else:
            start = self.batch_size * (index - self.train_batch_number)
            end = start + self.batch_size
            return index, self.validframes[start:end]

    def get_training_batch(self, index=None, blocking=True):
        if index is None:
            index = self.current_train_batch_index
            self.current_train_batch_index = (self.current_train_batch_index + 1) % self.train_batch_number
        if not blocking:
            if self.batchdata is None:
                return None
            index = min(index, self.train_batch_number-1)
            return self.batchdata[index]

        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
        index = min(index, self.train_batch_number-1)
        if self.batchdata[index] is None:
            self.urgent_queue.append(index)
        self.main_lock.wait_for(predicate=lambda: self.batchdata[index] is not None)
        self.main_lock.release()
        return self.batchdata[index]

    def get_training_batch_number(self, blocking=True):
        if not blocking:
            return self.train_batch_number
        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
        self.main_lock.release()
        return self.train_batch_number

    def get_validation_batch(self, index=None, blocking=True):
        if index is None:
            index = self.current_valid_batch_index
            self.current_valid_batch_index = (self.current_valid_batch_index + 1) % self.valid_batch_number

        if not blocking:
            if self.batchdata is None:
                return None
            index = min(index, self.valid_batch_number-1)
            return self.batchdata[self.train_batch_number + index]

        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
        index = min(index, self.valid_batch_number-1)
        realindex = self.train_batch_number + index
        if self.batchdata[realindex] is None:
            self.urgent_queue.append(realindex)
        self.main_lock.wait_for(predicate=lambda: self.batchdata[realindex] is not None)
        self.main_lock.release()
        return self.batchdata[realindex]

    def get_validation_batch_number(self, blocking=True):
        if not blocking:
            return self.valid_batch_number
        self.main_lock.acquire()
        self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
        self.main_lock.release()
        return self.valid_batch_number

    def train(self, blocking=True):
        return DatasetManager._DataSequence(getitem=lambda idx: self.get_training_batch(index=idx,
                                                                                        blocking=blocking),
                                            lenf=lambda: self.get_training_batch_number(blocking=blocking))

    def valid(self, blocking=True):
        return DatasetManager._DataSequence(getitem=lambda idx: self.get_validation_batch(index=idx,
                                                                                          blocking=blocking),
                                            lenf=lambda: self.get_validation_batch_number(blocking=blocking))


if __name__ == '__main__':
    # from matplotlib.pyplot import imshow, show
    import time
    dm = DatasetManager(train_samples=5000,
                        valid_samples=5000,
                        batch_size=100,
                        dataset_dir=crops_path(),
                        formatting=CROPS_STD_FORMAT)
    print('starting ds reading...')
    batches = []
    set_verbosity(DEBUG)
    t1 = time.time()
    valid = dm.valid()
    train = dm.train()
    idx = 0
    for valid_batch in valid:
        t = time.time()
        print('REQUEST: valid batch %d' % idx)
        batches.append(valid_batch)
        print('RESULT valid batch %d in %f ms' % (idx, 1000*(time.time()-t)))
        idx += 1

    for idx in range(len(train)):
        t = time.time()
        print('REQUEST: train batch %d' % idx)
        batches.append(train[idx])
        print('RESULT: train batch %d in %f ms' % (idx, 1000*(time.time()-t)))
    print('total time spent: %f ms' % (1000*(time.time()-t1)))
    # for batch in batches:
    #     for img in batch[IN]:
    #        imshow(img)
    #        show()


