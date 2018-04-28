from data import *


class ProcessingPlan:
    def __init__(self, augmenter: Augmenter=None,
                 regularizer: Regularizer=None,
                 keyset: set=None):
        self.ops = {}
        if keyset is not None:
            if augmenter is not None:
                for k in keyset:
                    self.augment(augmenter=augmenter,
                                 key=k)
            if regularizer is not None:
                for k in keyset:
                    self.regularize(key=k,
                                    regularizer=regularizer)

    def __setitem__(self, key, value):
        self.ops[key] = value

    def add_outer(self, key, fun):
        if key in self.ops.keys():
            cur_f = self.ops[key]
            self.ops[key] = lambda x: fun(cur_f(x))
        else:
            self[key] = fun
        return self

    def add_inner(self, key, fun):
        if key in self.ops.keys():
            cur_f = self.ops[key]
            self.ops[key] = lambda x: cur_f(fun(x))
        else:
            self[key] = fun
        return self

    def augment(self, key, augmenter: Augmenter):
        return self.add_inner(key=key,
                              fun=augmenter.apply_on_batch)

    def regularize(self, key, regularizer: Regularizer):
        return self.add_outer(key=key,
                              fun=regularizer.apply_on_batch)

    def process_batch(self, batch: dict):
        ret = {}
        for k in batch.keys():
            if k in self.ops.keys():
                ret[k] = self.ops[k](np.array(batch[k]))
            else:
                ret[k] = np.array(batch[k])
        return ret

    def process_filtered_batch(self, batch: dict, name_gen: NameGenerator):
        return self.process_batch(batch={k: batch[k] for k in name_gen.filter(batch.keys())})


if __name__ == '__main__':
    from matplotlib.pyplot import show, imshow
    data = DatasetManager(dataset_dir=crops_path(),
                          train_samples=3,
                          valid_samples=0,
                          batch_size=2,
                          formatting=CROPS_STD_FORMAT)

    tr = data.train()
    for batch in tr:
        for img in batch[IN(0)]:
            imshow(img)
            show()

    aug = Augmenter().shift_hue(prob=1, var=0.3).shift_sat(prob=1, var=0.3)
    pp = ProcessingPlan().augment(IN(0), aug)

    for batch in tr:
        for img in pp.process_filtered_batch(batch, IN)[IN(0)]:
            imshow(img)
            show()