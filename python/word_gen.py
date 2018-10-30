import json
import numpy as np
import os
import pickle
from prob import Distribution 

def map_chrs_to_int(raw):
    if len(raw) == 1:
        return ord(raw[0]) - ord('a')
    a = ord(raw[0]) - ord('a')
    b = ord(raw[1]) - ord('a')
    return (a+1) * 26 + b

def map_int_to_chrs(num):
    if num < 26:
        return chr(ord('a') + num)
    a = num // 26 - 1
    b = num % 26
    return chr(ord('a')+a) + chr(ord('a')+b)

def remove_bad_chars(raw):
    re = []
    word = ''
    for i in raw:
        if (i >= ord('a') and i <= ord('z')) or (i >= ord('A') and i <= ord('Z')):
            word += chr(i)
        else:
            if word != '':
                word = word.lower()
                re.append(word)
                word = ''
    return re

def load_data(fname):
    with open(fname, 'rb') as fin:
        for item in fin.readlines():
            yield item

def cut(arr, i, j):
    return arr[max(0,i):j]

def get_trans_matrix_v2():
    try:
        trans_matrix = pickle.load(open('transMatrix2.pkl','rb'))
    except:
        trans_matrix = np.zeros((26*27, 27), dtype='float32')
        base = ord('a')
        tmpL = os.walk('/home/liuyu/liuyu_lib/data/EngFictions/HarryPotter')
        wordsSet = set([])
        p, d, f = next(tmpL)
        for fname in f:
            for item in load_data(os.path.join(p, fname)):
                a = remove_bad_chars(item)
                for item in a:
                    wordsSet.add(item)
        wordsList = list(wordsSet)
        for word in wordsList:
            L = len(word)
            for i in range(L-1):
                inc = map_chrs_to_int(cut(word, i-1, i+1))
                outc = ord(word[i+1])-base
                trans_matrix[inc][outc] += 1
            end = map_chrs_to_int(cut(word, L-2, L))
            trans_matrix[end][26] += 1
        trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)
        pickle.dump(trans_matrix, open('transMatrix2.pkl','wb'))
    return trans_matrix

def get_trans_matrix_v1():
    try:
        trans_matrix = pickle.load(open('transMatrix.pkl','rb'))
    except:
        trans_matrix = np.zeros((26*27, 27), dtype='float32')
        base = ord('a')
        tmpL = os.walk('/home/liuyu/liuyu_lib/data/EngFictions/HarryPotter')
        p, d, f = next(tmpL)
        for fname in f:
            for item in load_data(os.path.join(p, fname)):
                a = remove_bad_chars(item)
                for word in a:
                    L = len(word)
                    for i in range(L-1):
                        inc = map_chrs_to_int(cut(word, i-1, i+1))
                        outc = ord(word[i+1])-base
                        trans_matrix[inc][outc] += 1
                    end = map_chrs_to_int(cut(word, L-2, L))
                    trans_matrix[end][26] += 1
        trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)
        pickle.dump(trans_matrix, open('transMatrix.pkl','wb'))
    return trans_matrix

class WordGen:
    def __init__(self, version=1):
        if version == 1:
            self.matrix = get_trans_matrix_v1()
        else:
            self.matrix = get_trans_matrix_v2()
        L = list(range(27))
        self.distribution = [Distribution(L, self.matrix[i]) for i in range(26*27)]

    def getWord(self, min_len=5, start = 0):
        re = ''
        start = start
        tries = 0
        while start != 26:
            re += chr(start + ord('a'))
            idx = map_chrs_to_int(cut(re, len(re)-2, len(re)))
            nexts = self.distribution[idx].sample_one()
            if len(re) < min_len:
                while nexts == 26 and tries < 20:
                    nexts = self.distribution[idx].sample_one()
                    tries += 1
            start = nexts
        return re

def main():
    gen = WordGen(version=2)
    for i in range(26):
        print(gen.getWord(min(10, np.random.randint(i//2,2*i+1)), start = i))

if __name__ == '__main__':
    main()
