l = ['a', 'b', 'c']
b = ['d', 'e', 'c']

# print(l)
# l = l[:-1]
print(l)
print(b)

if set(l).isdisjoint(b):
    print('two separate sets')
else:
    print('there are common elements')