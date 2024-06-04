# import time
#
# anj_name = ['A', 'NJE', 'LI', 'KA']
#
# for i in anj_name:
#     print(i)
#     time.sleep(1)
#     print('PAPARAPARAPARAM!')
#     time.sleep(1)
# print('................')
# time.sleep(1)
# for i in anj_name:
#     print(i)
#     time.sleep(0.5)
# print('TADAAAAAAM!')

import copy
import time

original_dict = {'a': 1, 'b': 2, 'c': 3}

# Using copy() method
start = time.time()
copied_dict1 = original_dict.copy()
print("copy() method:", time.time() - start)

# Using dict() constructor
start = time.time()
copied_dict2 = dict(original_dict)
print("dict() constructor:", time.time() - start)

# Using dictionary comprehension
start = time.time()
copied_dict3 = {k: v for k, v in original_dict.items()}
print("dictionary comprehension:", time.time() - start)

# Using copy module
start = time.time()
copied_dict4 = copy.copy(original_dict)
print("copy module:", time.time() - start)
