from hand_data_management.naming import contributors


contributors_file = open(contributors, 'r')
s = [[]]
for line in contributors_file:
    s.append(line.split())

s = [elem for elem in s if len(elem) != 0]

for elem in s:
    elem[1] = int(elem[1])

s.sort(key=lambda elem: elem[1], reverse=True)

s = s[0:10]

for elem in s:
    print(elem[0], "->", elem[1])
