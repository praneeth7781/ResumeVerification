import sys
from operator import itemgetter
import math
import numpy as np
from numpy.linalg import norm
import sys

def cosine_similarity(x, y):
    a = x.reshape((x.shape[1],))
    b = y.reshape((y.shape[1],))
    return np.inner(a, b) / (norm(a)*norm(b))


def find_similar(file, word):
    doc = np.load(file, allow_pickle=True)

    vec = doc.item()
    word_vec = vec.get(word)
    dist = {}
    # print type(vec)
    for element in vec.items():
        # print element[0]

        if (element[0] == word):
            continue

        arr = element[1]
        cos = cosine_similarity(word_vec, arr)
        dist[element[0]] = cos
        # print (element[0] , dist[element[0]])

    sorted_dist = sorted(dist.items(), key=itemgetter(1))
    print(sorted_dist[0:10])


def similar_check(file, worda, wordb):
    doc = np.load(file, allow_pickle=True)

    vec = doc.item()
    word_veca = vec.get(worda)
    word_vecb = vec.get(wordb)
    print (cosine_similarity(word_veca, word_vecb))


def find_analogy(file, worda, wordb, wordc):
    doc = np.load(file, allow_pickle=True)

    vec = doc.item()
    word_veca = vec.get(worda)
    word_vecb = vec.get(wordb)
    word_vecc = vec.get(wordc)
    word_vec = word_veca - word_vecb + word_vecc
    dist = {}
    # print type(vec)
    for element in vec.items():
        # print element[0]

        if (element[0] == worda or element[0] == wordb or element[0] == wordc):
            continue

        arr = element[1]
        cos = cosine_similarity(word_vec, arr)
        dist[element[0]] = cos
        # print (element[0] , dist[element[0]])

    sorted_dist = sorted(dist.items(), key=itemgetter(1), reverse=True)
    print(sorted_dist[0:10])

similar_check(sys.argv[1], sys.argv[2], sys.argv[3])
# find_similar(sys.argv[1], sys.argv[2])
# find_analogy(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
