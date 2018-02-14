from abc import abstractmethod, ABCMeta


class AbsLayer:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.output = None
        self.ready = False

    def set_ready(self):
        self.ready = True

    @abstractmethod
    def make_layer(self, other):
        pass

    def __add__(self, other):
        if isinstance(other, AbsLayer):
            return CompoundLayer(self, other)
        return NotImplemented

    def __radd__(self, other):
        self.make_layer(other)
        return self


class CompoundLayer(AbsLayer):
    def __init__(self, layer1, layer2):
        AbsLayer.__init__(self)
        self.__layer1 = layer1
        self.__layer2 = layer2
        if self.__layer1.ready and not self.__layer2.ready:
            self.connect()
        elif self.__layer2.ready:
            self.set_ready()
            self.output = self.__layer2.output

    def make_layer(self, inputs):
        self.__layer1.make_layer(inputs)
        self.connect()

    def connect(self):
        self.set_ready()
        self.__layer2.make_layer(self.__layer1.output)
        self.output = self.__layer2.output

    def __getattr__(self, item):
        return getattr(self.__layer2, item)
