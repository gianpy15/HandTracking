from data import *

# Define a plan for online batch processing.

# If used by a BatchGenerator (library.neural_network.keras.sequence.batch_generator.py)
# the specified operations will be performed on the specified batches of data
# each time before feeding them to the network


class ProcessingPlan:
    """
        Define a plan for online batch processing.

        Easy definition:
            aug = Augmenter().shiftHue(0.3)
            reg = Regularizer().normalize()
            pp = ProcessingPlane(augmenter=aug,
                                 regularizer=reg,
                                 keyset={IN(0), IN(1)})
            // just define a processing plan that performs augmentation
            // and regularization on entries IN(0) and IN(1) of the network data

        Complete definition:
            pp.add_inner(key=OUT(0), fun=lambda x: 2*x)
            // schedules the given function directly on all data coming from the OUT(0) entry
            pp.add_outer(...)
            // analogous meaning, but the function is applied after all the others
            pp[OUT(0)] = lambda x: 2*x
            // deletes any other schedules on OUT(0) and applies only the given function

        Use:
            The ProcessingPlan is mainly intended to be defined and then passed to a BatchGenerator
            or to the train_model function that will make BatchGenerators

            for custom use info refer to the doc of methods process_batch and process_filtered_batch
    """
    def __init__(self, augmenter: Augmenter=None,
                 regularizer: Regularizer=None,
                 keyset: set=None):
        """

        :param augmenter:
        :param regularizer:
        :param keyset:
        """
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
        """
        Apply all scheduled actions over a batch dictonary.
        Operations are not applied in place, so that the original batch data is preserved.
        Unspecified fields will be just copied in the output.

        :param batch: the batch dictionary to be processed
        :return: a dictionary of results, fields are copied if no op has been scheduled for them
        """
        ret = {}
        for k in batch.keys():
            if k in self.ops.keys():
                ret[k] = self.ops[k](np.array(batch[k]))
            else:
                ret[k] = np.array(batch[k])
        return ret

    def process_filtered_batch(self, batch: dict, name_gen: NameGenerator):
        """
        Filter the batch only on names provided by the specified NameGenerator
        and then process only those fields. Avoids useless copies.
        """
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