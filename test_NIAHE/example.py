#!/usr/bin/env python3
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1' 
#os.environ['MKL_NUM_THREADS'] = '1'

## these linesline if you want to use the git version of the code, instead of the one installed by pip
num_proc=1


import os

#os.symlink("../wannierberri","wannierberri")

import wannierberri as wberri

import numpy as np


#SYM=wberri.symmetry

Efermi=np.linspace(12.,13.,11)
system=wberri.System_w90(seedname='wannier')

#generators=[SYM.Inversion,SYM.C4z,SYM.TimeReversal*SYM.C2x]
#system.set_symmetry(generators)
grid=wberri.Grid(system,length=100)

wberri.integrate(system,
            grid=grid,
            Efermi=Efermi, 
            smearEf=10,
            #quantities=["NIAHE"],
            quantities=["ahc"],
            #quantities=["ahc","dos","cumdos"],
            numproc=num_proc,
            adpt_num_iter=0,
            fftlib='fftw', #default.  alternative  option - 'numpy'
            fout_name='CuMnAs',
            restart=False,
            )
