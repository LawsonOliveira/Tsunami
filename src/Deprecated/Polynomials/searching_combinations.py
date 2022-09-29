import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))
import os
from itertools import permutations, combinations_with_replacement, combinations
from time import time


def f_sort(t):
    '''
    A criteria to sort the possible combinations. It will prefer small exponents and more dependance between different dimension. 
    The less interesting is a combination, the higher is the image.

    Variable : 
        t : a tuple
    Returns :
        res : a float to sort a list of tuples
    '''
    res = sum(t)
    n_zeros = 0
    for x in t:
        if not(x):
            n_zeros += 1
    res += n_zeros/len(t)

    return res


def combi_itertools(N, D):
    '''
    returns sorted possible combinations of D tuples of integer in [0,N-1].
    Use itertools package

    Variables :
        - N (int) = N-1 is the higher possible component.
        - D (int) = the dimension.
    Returns :
        list of sorted possible combinations of exponents
    '''
    L = list(range(N))
    # print(L)
    combi_poss = set()
    for c in combinations_with_replacement(L, D):
        # print(c)
        if not c in combi_poss:
            for c in permutations(c):
                combi_poss = combi_poss | {c}
    combi_poss = list(combi_poss)
    return sorted(combi_poss, key=f_sort)


def combi_indep(N, D):
    '''
    returns sorted possible combinations of D tuples of integer in [0,N-1].
    DO NOT use itertools package
    Recursive

    Variables :
        - N (int) = N-1 is the higher possible component.
        - D (int) = the dimension, the size of a combination
    Returns :
        list of possible combinations of exponents (not sorted)
    '''
    combi_poss = []
    # initialize
    if not(D):
        # no possibility
        return combi_poss
    elif D == 1:
        return [[i] for i in range(N)]
    # core of recursivity
    for combi_Dprec in combi_indep(N, D-1):
        for i in range(N):
            combi_poss.append([i]+combi_Dprec)
    return combi_poss


if __name__ == '__main__':
    N = 20
    D = 3
    time_start = time()
    C = sorted(combi_indep(N, D), key=f_sort)
    time_end = time()
    print("combi_indep + sorted() :", time_end-time_start)

    time_start = time()
    C = combi_itertools(N, D)
    time_end = time()
    print("combi_itertools :", time_end-time_start)

    # Remarks :
    # combi_indep + sorted is faster !
