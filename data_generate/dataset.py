#Run Inst.s

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
