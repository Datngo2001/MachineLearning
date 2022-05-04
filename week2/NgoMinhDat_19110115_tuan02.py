# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 08:19:29 2022

@author: ngomi
"""

# %%
input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
output = []
for x in input:
    if (x % 5) == 0: 
        output.append(x)
    pass
print(output)
# %%
import numpy as np

arr = np.array([-2, 6, 3, 10, 15, 48])
print(arr[2:5:2])
print(arr[1:6:2])
print(arr[3:6:1])
print(arr[-1:2:-1])
# %%

def sapxep(arr, sapseptang):
    isReverse = not sapseptang
    return arr.sort(reverse=isReverse)

arr = [-2, 6, 3, 10, 15, 48]
sapxep(arr, True)
print(arr)
sapxep(arr, False)
print(arr)
    

