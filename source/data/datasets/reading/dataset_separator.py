import os
import re
import numpy as np
import random


class DatasetSeparator:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_info = self.__get_available_dataset_stats()

    def __get_available_dataset_stats(self):
        framelist = os.listdir(self.dataset_dir)
        stats = {}
        for frame in framelist:
            name = self.__extract_vidname(frame)
            if name is None:
                continue
            if name in stats.keys():
                stats[name] += 1
            else:
                stats[name] = 1
        return stats

    def exclude_videos(self, videos):
        to_be_deleted = [vid for vid in self.dataset_info.keys() if
                         any([re.match(vidreg, vid) for vidreg in videos])]
        for vid in to_be_deleted:
            del self.dataset_info[vid]
        return self.dataset_info

    def __available_frames(self, vidlist):
        count = 0
        availablevids = self.dataset_info.keys()
        for vid in vidlist:
            if vid in availablevids:
                count += self.dataset_info[vid]
        return count

    def __extract_vidname(self, framename):
        name_match = re.match("(?P<vid_name>^.*)_[^_]*\.mat$", framename)
        if name_match is None:
            return None
        return name_match.groups("vid_name")[0]

    def __vidlist_to_full_framelist(self, vidlist):
        ret = []
        framelist = os.listdir(self.dataset_dir)
        for frame in framelist:
            name = self.__extract_vidname(frame)
            if name is None or name not in vidlist:
                continue
            fullname = os.path.join(self.dataset_dir, frame)
            ret.append(fullname)
        return ret

    def select_train_validation_framelists(self, train, valid):

        # select which videos will be used for which task
        trainvids, validvids, sharedvids = self.__choose_train_valid_shared_videos(train, valid)
        # read the complete frames list
        trainpool = self.__vidlist_to_full_framelist(trainvids)
        validpool = self.__vidlist_to_full_framelist(validvids)
        # if any shared video is needed, shared based on needs
        if len(sharedvids) > 0:
            trainavail = self.__available_frames(trainvids)
            validavail = self.__available_frames(validvids)
            sharedframes = self.__vidlist_to_full_framelist(sharedvids)
            trainmissing = max(0, train-trainavail)
            validmissing = max(0, valid-validavail)
            assert trainmissing or validmissing
            to_train_idx = int(len(sharedframes) * trainmissing / (trainmissing + validmissing))
            trainpool += sharedframes[:to_train_idx]
            validpool += sharedframes[to_train_idx:]

        # shuffle them all! if needed, of course
        if len(trainpool) > train:
            random.shuffle(trainpool)
        if len(validpool) > valid:
            random.shuffle(validpool)

        return trainpool[:train], validpool[:valid]

    def __choose_train_valid_shared_videos(self, train, valid):
        assigned_to_train = 0
        assigned_to_valid = 0
        available_videos = list(self.dataset_info.keys())
        available_videos.sort(key=self.dataset_info.__getitem__)
        trainvids = []
        validvids = []
        train_satis = 0. if train > 0 else np.inf
        valid_satis = 0. if valid > 0 else np.inf

        def count_smaller_than(value):
            c = 0
            for v in available_videos:
                if self.dataset_info[v] < value:
                    c += 1
            return c

        while len(available_videos) > 0:
            nextbatch, missing = (trainvids, train - assigned_to_train) if train_satis <= valid_satis \
                else (validvids, valid - assigned_to_valid)
            if missing <= 0 or missing >= self.dataset_info[available_videos[-1]]:
                vid = available_videos.pop()
            else:
                vid = available_videos.pop(count_smaller_than(missing))
            nextbatch.append(vid)
            if nextbatch is trainvids:
                assigned_to_train += self.dataset_info[vid]
                train_satis = assigned_to_train / train if train > 0 else np.inf
            else:
                assigned_to_valid += self.dataset_info[vid]
                valid_satis = assigned_to_valid / valid if valid > 0 else np.inf

        if train_satis >= 1 and valid_satis >= 1:
            return trainvids, validvids, []
        elif train_satis < 1 and valid_satis < 1:
            return trainvids, validvids, []
        elif train_satis == np.inf or valid_satis == np.inf:
            return trainvids, validvids, []
        else:
            poolbatch, target = (trainvids, valid - assigned_to_valid) if train_satis >= valid_satis \
                else (validvids, train - assigned_to_train)
            poolbatch.sort(key=self.dataset_info.__getitem__)
            shared = poolbatch.pop(count_smaller_than(target))
            return trainvids, validvids, [shared]


if __name__ == '__main__':
    from data.naming import *
    ds = DatasetSeparator(crops_path())
    ds.exclude_videos('hands')
    sel = ds.select_train_validation_framelists(train=5, valid=5)
    print('train: %d' % len(sel[0]))
    print('valid: %d' % len(sel[1]))
    print('intersection: %s' % set(sel[0]).intersection(set(sel[1])))
