# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:54:15 2020

@author: fbasatemur
"""

import os
import h5py
import sys
import numpy as np

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
            arrayDim = len(h5[d[item].name][:].shape[:])
            
            if arrayDim == 1:
                  np.savetxt(d[item].name.split('/')[-1].split(':')[0] + ".txt", h5[d[item].name][:], fmt='%.6e')
            
            elif arrayDim == 2:
                  np.savetxt(d[item].name.split('/')[-1].split(':')[0] + ".txt", h5[d[item].name][:][:], fmt='%.6e')
                  
            elif arrayDim == 3:
                  f = open(d[item].name.split('/')[-1].split(':')[0] + ".txt", "w")
                  # h5[d[item].name][row][col][depth]
                  for depth in range(h5[d[item].name][:].shape[2]):
                        for row in range(h5[d[item].name][:].shape[0]):
                              for col in range(h5[d[item].name][:].shape[1]):
                                    f.write(format(h5[d[item].name][row][col][depth], '.9f') + " ")
                              f.write("\n")
                        f.write("\n")
                  f.close()

            elif arrayDim == 4:
                  f = open(d[item].name.split('/')[-1].split(':')[0] + ".txt", "w")
                  # h5[d[item].name][row][col][depth][filter]
                  for filter in range(h5[d[item].name][:].shape[3]):
                        for depth in range(h5[d[item].name][:].shape[2]):
                              for row in range(h5[d[item].name][:].shape[0]):
                                    for col in range(h5[d[item].name][:].shape[1]):
                                          f.write(format(h5[d[item].name][row][col][depth][filter], '.9f') + " ")
                                    f.write("\n")
                              f.write("\n")
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