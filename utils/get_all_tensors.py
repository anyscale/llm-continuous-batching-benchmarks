"""
Utility for identifying the source of PyTorch memory leaks.
"""

from collections import defaultdict
import torch
import gc

import random
a = []
for _ in range(100):
    a.append(torch.ones(random.randint(1, 10), random.randint(5, 7)).to('cuda'))
    a.append(torch.ones(random.randint(1, 3), random.randint(5, 7)).to('cuda'))


d = defaultdict(int)

for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            t = tuple(obj.size()) + (obj.dtype, obj.device)
            d[t] += 1
    except:
        pass

for count, obj_signature in sorted([(count, sig) for sig, count in d.items()], key=lambda x: x[0], reverse=True):
    print(count, obj_signature)
