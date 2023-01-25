# Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.
# This file is a part of Emap2sec package with -
# Reference:  Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, and Daisuke Kihara. Protein Secondary Structure Detection in Intermediate Resolution Cryo-Electron Microscopy Maps Using Deep Learning. Nature Methods (2019).
## License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License V3 for more details.
#
# You should have received a copy of the GNU v3.0 General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.

import sys, os

# Auxiliar functions
def getFilePath(rawPath):
    """
    This function parses a given path.
    """
    absPath = os.path.expanduser(rawPath) if "~" in rawPath else rawPath
    return os.path.abspath(absPath)

def readFile(filePath):
    """
    This function returns the content of a file.
    Raises an exception if there were any problems reading it.
    """
    try:
        file = open(getFilePath(filePath), "r")
        content = file.readlines()
        file.close()
        return content
    except Exception as e:
        raise Exception("There was an error opening file {}.\n{}".format(filePath, e))

# Getting params
dataFilePath = sys.argv[1]
outputFilePath = sys.argv[2]

# Reading input files
dataFileLines = readFile(dataFilePath)

# Opening output file
outputFile = open(getFilePath(outputFilePath), "w")

# Writing dataset file into output path
factor = 4
coords = []
for line in dataFileLines:
    if(line.rstrip() != "" and not line.startswith("-2") and not line.startswith("#C: Res= -2")
        and not line.startswith("#Base") and not line.startswith("#Steps") and not line.startswith("#dmax")):

        if(line.startswith("#C:")):
            equ = line.split("=")
            coords = equ[len(equ) - 1].split(" ")[1:4]

        elif(line.startswith("-1")):
            outputFile.write(str(int(int(coords[0]) / factor)) + "," + str(int(int(coords[1]) / factor)) + "," + str(int(int(coords[2].rstrip()) / factor)) + "," + line)

        elif(line.startswith("#Voxel")):
            outputFile.write(line.split()[2] + "," + line.split()[3] + "," + line.split()[4] + "\n")

        else:
            li = line.split(",")
            outputFile.write(str(int(int(coords[0]) / factor)) + "," + str(int(int(coords[1]) / factor)) + "," + str(int(int(coords[2].rstrip()) / factor)) + ",0")
            for i in li[1:]:
                outputFile.write("," + i)

# Closing output file
outputFile.close()
