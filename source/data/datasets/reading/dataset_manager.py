from data import *
from library import *
import math
import threading
from data.datasets.reading.dataset_separator import DatasetSeparator
from data.datasets.reading.general_reading import read_formatted_batch
from library.multi_threading.thread_pool_manager import ThreadPoolManager
import traceback

# Here is all the asynchronous reading logic.
# Please use this class to manage datasets to enable asynchronous I/O


class DatasetManager:
    """
    An asynchronous manager of datasets: it reads data and provides them formatted
    as specified. It implements the abstraction of having all data instantly available.

    Data are formatted and provided in the form of batch dictionaries.
    The formatting dictionary specified at initialization determines the content of the dictionaries.
    example:
        data formatted as:
        fmt = {
            'in': (IMGDATA, DEPTHDATA),  // builds RGBD data according to formatting specification
            'out': (HEATDATA,)          // just plain heatmap data
        }
        then:
        dm = DatasetManager(...., formatting=fmt) // produce one istance
        train_data = dm.train()  // get interface to train data, see _DataSequence down here
        train_data[i] -> the dictionary of contents relative to the i-th batch:
        train_data[i] is:
            {
                'in': <a 4-DIM numpy array containing a single RGBD data batch>
                'out': <a 4-DIM numpy array containing a single heatmap data batch>
            }
        // NOTE: if the formatting specification produces dimension D,
        //       the batched fields will have dimension D+1

    """
    class _DataSequence:
        """
        A simplified interface to the DatasetManager.
        Data are provided in the form of dictionary-like sequences for one data stream at a time.

        This interface provides full expressiveness towards one single data partition (either train or valid)

        DataSequences are provided by:
            DataManager.train() -> sequence of train batchdicts
            DataManager.valid() -> sequence of validation batchdicts

        A sequence can be accessed in many pythonic ways:

            seq[i] -> returns i-th batchdict of sequence
            len(seq) -> returns the total number of batchdicts in the sequence
            for elem in seq: -> iterates over all batchdicts in the sequence

        """
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
        """
        Initializes a DatasetManager specifying all necessary parameters for asynchronous data reading
        :param train_samples: the amount of trainig samples to use. May provide less samples than specified.
        :param valid_samples: the amount of validation samples to use. May provide less samples than specified.
        :param batch_size: the number of frames to include in each chunk of read and provided data.
                           Notice that the last batch may be smaller than specified.
                           Use a batch size greater than both training and validation samples to
                           build one single big bach of data
        :param dataset_dir: The directory of the dataset, absolute path
        :param formatting: The formatting specification to provide the data.
                           See the class doc and formatting.py for details
        :param exclude_videos: a list of regexes of video names to be excluded from the dataset
        """
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
        log("DATA LOADING WORKER: initializing...", level=COMMENTARY)
        separator = DatasetSeparator(self.dataset_dir)
        separator.exclude_videos(self.exclude_videos)
        try:
            self.trainframes, self.validframes = separator.select_train_validation_framelists(self.train_samples,
                                                                                    self.valid_samples)
        except Exception as e:
            traceback.print_exc()
            log(str(e), level=ERRORS)
            raise e
        log("DATA LOADING WORKER: determined training and validation set", level=COMMENTARY)
        train_avail_tot = len(self.trainframes)
        valid_avail_tot = len(self.validframes)
        self.train_batch_number = int(math.ceil(train_avail_tot / self.batch_size))
        self.valid_batch_number = int(math.ceil(valid_avail_tot / self.batch_size))
        self.batchdata = [None for _ in range(self.train_batch_number + self.valid_batch_number)]
        log("DATA LOADING WORKER: starting to load data...", level=COMMENTARY)
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
            with self.main_lock:
                self.batchdata[idx] = data
                self.main_lock.notify_all()
        log("DATA LOADING WORKER: work done, quitting", level=COMMENTARY)

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
        log("Requested training batch %d" % index, level=DEBUG)
        if index is None:
            index = self.current_train_batch_index
            self.current_train_batch_index = (self.current_train_batch_index + 1) % self.train_batch_number
        if not blocking:
            if self.batchdata is None:
                return None
            index = min(index, self.train_batch_number-1)
            return self.batchdata[index]

        with self.main_lock:
            self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
            index = min(index, self.train_batch_number-1)
            if self.batchdata[index] is None:
                self.urgent_queue.append(index)
            self.main_lock.wait_for(predicate=lambda: self.batchdata[index] is not None)

        return self.batchdata[index]

    def get_training_batch_number(self, blocking=True):
        log("Requested training batch number", level=DEBUG)
        if not blocking:
            return self.train_batch_number
        with self.main_lock:
            self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
        return self.train_batch_number

    def get_validation_batch(self, index=None, blocking=True):
        log("Requested validation batch %d" % index, level=DEBUG)
        if index is None:
            index = self.current_valid_batch_index
            self.current_valid_batch_index = (self.current_valid_batch_index + 1) % self.valid_batch_number

        if not blocking:
            if self.batchdata is None:
                return None
            index = min(index, self.valid_batch_number-1)
            return self.batchdata[self.train_batch_number + index]

        with self.main_lock:
            self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)
            index = min(index, self.valid_batch_number-1)
            realindex = self.train_batch_number + index
            if self.batchdata[realindex] is None:
                self.urgent_queue.append(realindex)
            self.main_lock.wait_for(predicate=lambda: self.batchdata[realindex] is not None)
        return self.batchdata[realindex]

    def get_validation_batch_number(self, blocking=True):
        log("Requested validation batch number", level=DEBUG)
        if not blocking:
            return self.valid_batch_number
        with self.main_lock:
            self.main_lock.wait_for(predicate=lambda: self.batchdata is not None)

        return self.valid_batch_number

    def train(self, blocking=True):
        """
        Get complete read-only access to the training data as a sequence-like interface.
        See _DataSequence specification on top of this class.
        :param blocking: determine the behaviour on unavailable data:
            blocking=True => set requested data as urgent, and wait for them before returning
            blocking=False => returns None if data is not available
        :return: a sequence-like interface to training data
        """
        return DatasetManager._DataSequence(getitem=lambda idx: self.get_training_batch(index=idx,
                                                                                        blocking=blocking),
                                            lenf=lambda: self.get_training_batch_number(blocking=blocking))

    def valid(self, blocking=True):
        """
        Get complete read-only access to the validation data as a sequence-like interface.
        See _DataSequence specification on top of this class.
        :param blocking: determine the behaviour on unavailable data:
            blocking=True => set requested data as urgent, and wait for them before returning
            blocking=False => returns None if data is not available
        :return: a sequence-like interface to validation data
        """
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


