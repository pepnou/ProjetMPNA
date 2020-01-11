#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:02:48 2020

@author: pepnou
"""

import re
#import matplotlib.pyplot as plt
import numpy as np
import math as m

start = 6
stop = 14

data = np.zeros(shape=(40, 2, stop-start+1), dtype=float)
data[:, 0] = range(start, stop+1)

f = open("perf3", "r")
res1 = open("results1.dat", "w")
res1.write("# X   Y   Z\n")
res2 = open("results2.dat", "w")
res2.write("# X   Y   Z\n")

for x in f:
    param = [int(s) for s in re.findall(r'-?\d+', x)][0:2]
    time = [float(s) for s in re.findall(r'-?\d+\.?\d*', x)][1]
    data[param[0]-1,0,param[1]-6] = param[1]
    data[param[0]-1,1,param[1]-6] = time
    #res.write(str(param[0])+"   "+str(param[1])+"   "+str(time)+"\n\n")

f.close()

for i in range(40):
    for j in range(stop-start+1):
        if data[i, 1, j] != 0 :
            #res.write(str(i+1)+" "+str(j+start)+" "+str(m.log(data[i, 1, j], 2))+"\n")
            res1.write(str(i+1)+" "+str(j+start)+" "+str(data[i, 1, j]**(1./3.))+"\n")
        #    res2.write(str(i+1)+" "+str(j+start)+" "+str(data[i, 1, j]**(1./3.))+"\n")
        else:
            res1.write(str(i+1)+" "+str(j+start)+" NAN\n")
        #res1.write(str(i+1)+" "+str(j+start)+" "+str(data[i, 1, j]**(1./3.))+"\n")
        #res2.write(str(i+1)+" "+str(j+start)+" "+str(data[i, 1, j]**(1./3.))+"\n")
    res1.write("\n")
    res2.write("\n")

res1.close()
res2.close()

#print(data[4])


#plt.figure(figsize=(100,100))
#for i in range(40):
#    plt.plot(data[i, 0], data[i, 1], label=str(i+1)+'nodes')

#plt.legend()
#plt.show()
#plt.savefig("result")

#import csv

#fcsv = open('results.csv', 'w')

#with fcsv:
#    writer = csv.writer(fcsv)
#    writer.writerow(data[0, 0])
#    for i in range(40):
#        writer.writerow(data[i, 1])