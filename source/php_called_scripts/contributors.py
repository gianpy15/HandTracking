import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from hand_data_management.naming import contributors

if __name__ == '__main__':
    contributors_file = open(contributors, 'r')
    s = [[]]
    for line in contributors_file:
        s.append(line.split())

    s = [elem for elem in s if len(elem) != 0]

    for elem in s:
        elem[1] = int(elem[1])

    s.sort(key=lambda elem: elem[1], reverse=True)

    s = s[0:10]

    for i in range(len(s)):
        print('{}.'.format(i + 1), s[i][0], s[i][1])

    n = 10 - len(s)
    for i in range(n, 10):
        print('{}.'.format(i + 1))
