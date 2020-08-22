#Run Inst.s

# Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.
# This file is a part of Emap2sec package with -
# Reference:  Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, and Daisuke Kihara. Protein Secondary Structure Detection in Intermediate Resolution Cryo-Electron Microscopy Maps Using Deep Learning. Nature Methods (2019).
## License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License V3 for more details.
#
# You should have received a copy of the GNU v3.0 General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.


import sys
import numpy as np

dataFile = sys.argv[1];
outputFile = sys.argv[2];
factor=4
fil1 = open(dataFile);
fil3 = open(outputFile,'w');    
lines1 = fil1.readlines();

j=0;


count=0
coords=[]
for line in lines1:
    if(line.rstrip()==''):
        continue
    elif(line.startswith("-2") or line.startswith('#C: Res= -2')):
        continue   
    elif(line.startswith('#C:')):
        equ = line.split('=')
        coords = equ[len(equ)-1].split(' ')[1:4]
        continue
 
 
    if(line.startswith("-1")):
        fil3.write(str(int(int(coords[0])/factor))+","+str(int(int(coords[1])/factor))+","+str(int(int(coords[2].rstrip())/factor))+','+line)
        continue    

    elif(line.startswith('#Base') or line.startswith('#Steps')):
        continue    

    elif(line.startswith("#Voxel")):
        fil3.write(line.split()[2]+","+line.split()[3]+","+line.split()[4])
        fil3.write('\n')
        continue
    elif(line.startswith("#dmax")):
        continue    

    li = line.split(',');
    flag=0;
    #print(len(labelArray1))
    for i in li:
        if(flag==0):
            label = "0"
            fil3.write(str(int(int(coords[0])/factor))+","+str(int(int(coords[1])/factor))+","+str(int(int(coords[2].rstrip())/factor))+',')
            fil3.write(label)
            flag=1;
        else:
            fil3.write(',');
            fil3.write(i);
    count+=1
