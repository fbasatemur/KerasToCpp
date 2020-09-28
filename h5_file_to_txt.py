# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:54:15 2020

@author: fbasatemur
"""

import os
import h5py
import sys

def hierarchy(d):
    for item in d:
        if ' 0 member' in str(d[item]):
            print(d[item].name, ['empty group'])
        if isinstance(d[item], h5py.Group):
            hierarchy(d[item])
        else: #Dataset

            if not os.path.isdir("./" + d[item].name.split('/')[-2]):
                  os.mkdir("./" + d[item].name.split('/')[-2])

            os.chdir("./" + d[item].name.split('/')[-2]+'/')
            f = open(d[item].name.split('/')[-1].split(':')[0] + ".txt", "w")
            
            arrayDim = len(h5[d[item].name][:].shape[:])
            
            if arrayDim == 1:
                  for value in h5[d[item].name][:]:
                        f.write(str(value) + " ")
            
            elif arrayDim == 2:
                  for row in range(h5[d[item].name][:].shape[0]):
                        for col in range(h5[d[item].name][:].shape[1]):
                              f.write(str(h5[d[item].name][row][col]) + " ")
                        f.write("\n")
                  
            f.close()
            os.chdir("..")

try:
      filename = sys.argv[1]
      h5 = h5py.File(filename, 'r')

      hierarchy(h5)
      print ("Succesfull")
except:
      print ("file not found !")


#  /dense/dense/kernel:0 view
# print(h5['/dense/dense/kernel:0'][:])