# coding: utf-8

# In[1]:

#!/usr/bin/env python

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


from pymol import cmd
#show secondary structures as spheres
cmd.show_as('spheres','visual')
cmd.select('chain D and visual')
cmd.hide("spheres","sele")
cmd.delete("sele")
#color secondary structures
cmd.color("magenta",'chain C and visual')
cmd.color("yellow",'chain B and visual')
cmd.color("limegreen",'chain A and visual')
#color crystal structure
cmd.color("magenta",'ss h')
cmd.color("yellow",'ss s')
cmd.color("limegreen",'ss l')
