import random as rand
from library.utils.deprecation import deprecated_class


def __check_data_consistence__(data):
    length = len(data[0])
    for d in data:
        if len(d) != length:
            return False
    return True


@deprecated_class
class DataManager:
    def __init__(self, *data, **args):
        if not __check_data_consistence__(data):
            length = []
            for d in data:
                length.append(len(d))
            raise IndexError('The dimensions of the sets are different', length)

        i = 0
        self.__data__ = {}
        for key in args.keys():
            self.__data__[key] = [d[i:i + args[key]] for d in data], 0
            i += args[key]

    def get_random_set(self, key, dimension):
        """
        :param key: is the subset of the initial data from which takes the data
        :param dimension: is the output dimension (must be less than the
                          initialization value)
        :return: a subset of a given dimension of the data from a random index
        """
        if key not in self.__data__.keys():
            return None
        else:
            data = self.__data__[key][0]
            i = rand.randint(0, len(data) - dimension)

            self.__tuple_assignment__(key, i + dimension)
            return [d[i:i + dimension] for d in data]

    def get_sequential_set(self, key, dimension):
        """
        :param key: is the subset of the initial data from which takes the data
        :param dimension: is the output dimension (must be less than the
                          initialization value)
        :return: a subset of a given dimension of the data from a sequential index
        """
        if key not in self.__data__.keys():
            return None
        data = self.__data__[key]
        if data[1] + dimension > len(data[0][0]):
            self.__tuple_assignment__(key, 0)
        data = self.__data__[key]
        i = data[1]
        self.__tuple_assignment__(key, i + dimension)
        # print data
        return [d[i:i + dimension] for d in data[0]]

    def __tuple_assignment__(self, key, value):
        self.__data__[key] = list(self.__data__[key])
        self.__data__[key][1] = value
        self.__data__[key] = tuple(self.__data__[key])
