# coding: utf-8

# In[1]:

#!/usr/bin/env python

# Copyright (C) 2018 Sai Raghavendra Maddhuri, Genki Terashi, Daisuke Kihara, and Purdue University.
# This file is a part of Emap2sec package with -
# Reference:  Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, and Daisuke Kihara. Protein Secondary Structure Detection in Intermediate Resolution Cryo-Electron Microscopy Maps Using Deep Learning. In submission (2018).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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