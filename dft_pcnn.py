# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy
from numpy.random import *
import numpy as np
from pyscf import gto, dft, scf, cc, mp
from pyscf.dft import numint
import math
import os
import sys
import time
import xc_pcNN as xc

print("input the target .xyz path: ")
target_mol = input()
print("input the number of unpaired electrons: ")
spin = int(input())
print("input the charge number: ")
charge = int(input())

hidden = 100

# loadm="xc_scan1"#original
loadm = "pcNN"

glevel = 3


def mldft(mol, eval_xc, glevel=3):
    if mol.spin == 0:
        mfl = dft.RKS(mol)
    else:
        mfl = dft.UKS(mol)
    mfl.grids.level = glevel
    mfl = mfl.define_xc_(eval_xc, 'MGGA')
    mfl.kernel()
    return mfl


model = xc.Net()
model.hidden = hidden
model.mkmat()
model.load(loadm)


mol = gto.Mole()
mol.verbose = 4
mol.basis = "6-31G"
mol.charge = charge
mol.atom = open(target_mol).read()
mol.spin = spin
mol.max_memory = 4000
mol.build()

mfl = mldft(mol, model.eval_xc, glevel)
Eori1 = mfl.e_tot
