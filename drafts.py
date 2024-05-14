# l = ['a', 'b', 'c']
# b = ['d', 'e', 'c']
#
# # print(l)
# # l = l[:-1]
# print(l)
# print(b)
#
# if set(l).isdisjoint(b):
#     print('two separate sets')
# else:
#     print('there are common elements')

# n = 100_000_000
# for i in range(n):
#     print(f'\r{int(i/n * 100)} % : {i:_}', end='')

# from itertools import pairwise
#
# lst = [1,2,3,4,5]
# # lst = [1,2]
# print(lst[-2:])
# print("Original list - ", lst)
# print("Successive overlapping pairs - ", list(pairwise(lst)))

l = [('a', 1, [1, 3, 4])]
print(l)
l[-1][2].append(12)
print(l)