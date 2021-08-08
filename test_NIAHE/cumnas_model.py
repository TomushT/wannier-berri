#!/usr/bin/env python                                            

# Toy graphene model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt

def plot_bands(my_model):
    # generate list of k-points following a segmented path in the BZ
    # list of nodes (high-symmetry points) that will be connected
    path=[[0.,0.],[.5,0.],[.5,.5],[.0,.5],[-.5,.5],[-.5,0.]]
    # labels of the nodes
    label=(r'$\Gamma$', r'$X$', r'$M$', r'$Y$', r"$M'$", r"$X'$")
    # total number of interpolated k-points along the path
    nk=500
    
    # call function k_path to construct the actual path
    (k_vec,k_dist,k_node)=my_model.k_path(path,nk)
    # inputs:
    #   path, nk: see above
    #   my_model: the pythtb model
    # outputs:
    #   k_vec: list of interpolated k-points
    #   k_dist: horizontal axis position of each k-point in the list
    #   k_node: horizontal axis position of each original node
    
    
    print('---------------------------------------')
    print('starting calculation')
    print('---------------------------------------')
    print('Calculating bands...')
    
    # obtain eigenvalues to be plotted
    evals=my_model.solve_all(k_vec)
    
    # figure for bandstructure
    
    fig, ax = plt.subplots()
    # specify horizontal axis details
    # set range of horizontal axis
    ax.set_xlim(k_node[0],k_node[-1])
    # put tickmarks and labels at node positions
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    # add vertical lines at node positions
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    # put title
    ax.set_title("Band structure")
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy")
    
    # plot first and second band
    for i in range(len(evals)):
        ax.plot(k_dist,evals[i])
    
    # make an PDF figure of a plot
    fig.tight_layout()
    fig.savefig("cumnas_model_bands.pdf")
    
    print('Done.\n')
    

def niahe(model):
    
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1' 
    os.environ['MKL_NUM_THREADS'] = '1'
    import matplotlib
    matplotlib.use('Agg')
    import wannierberri as wberri
    import numpy as np
    import matplotlib.pyplot as plt
    plt.close('all')
    
    # Call the interface for TBmodels to define the system class
    system=wberri.System_PythTB(model,berry=True)

    SYM=wberri.symmetry

    Efermi=np.linspace(-2,2,1001)

    generators=[SYM.Mx,SYM.Mz,SYM.C2y]
    #system.set_symmetry(generators)
    grid=wberri.Grid(system,length=100)
    
    wberri.integrate(system,
                grid=grid,
                Efermi=Efermi, 
                smearEf=10,
                quantities=["NIAHE"],
                #quantities=["ahc"],
                #quantities=["ahc","dos","cumdos"],
                numproc=2,
                adpt_num_iter=40,
                adpt_fac = 2,
                fftlib='fftw', #default.  alternative  option - 'numpy'
                fout_name='cumnas_model',
                restart=False,
                )









# initialize the CuMnAs model
lat=[[1.0,0.0],[0.0,1.0]]
orb=[[0.,0.],[0.,0.],[0.5,0.5],[0.5,0.5]]

# model parameters from PRL 117 106402 (2017)
t = 1
t1 = 0.08*t
l = 0.8*t
J = 0.6*t
nx = 1.
ny = 9.
nz = 0.

nabs = np.sqrt(nx**2 + ny**2 + nz**2)
nx = nx/nabs
ny = ny/nabs
nz = nz/nabs

my_model=tb_model(2,2,lat,orb)

# set on-site energies
my_model.set_onsite([J*nz,-J*nz,-J*nz,J*nz])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(-0.5*t, 0, 2, [ 0, 0]) 
my_model.set_hop(-0.5*t, 0, 2, [-1, 0]) 
my_model.set_hop(-0.5*t, 0, 2, [ 0,-1]) 
my_model.set_hop(-0.5*t, 0, 2, [-1,-1]) 
my_model.set_hop(-0.5*t, 1, 3, [ 0, 0]) 
my_model.set_hop(-0.5*t, 1, 3, [-1, 0]) 
my_model.set_hop(-0.5*t, 1, 3, [ 0,-1]) 
my_model.set_hop(-0.5*t, 1, 3, [-1,-1]) 
my_model.set_hop(-0.5*t1, 0, 0, [ 1, 0]) 
my_model.set_hop(-0.5*t1, 0, 0, [ 0, 1]) 
my_model.set_hop(-0.5*t1, 1, 1, [ 1, 0]) 
my_model.set_hop(-0.5*t1, 1, 1, [ 0, 1]) 
my_model.set_hop(-0.5*t1, 2, 2, [ 1, 0]) 
my_model.set_hop(-0.5*t1, 2, 2, [ 0, 1]) 
my_model.set_hop(-0.5*t1, 3, 3, [ 1, 0]) 
my_model.set_hop(-0.5*t1, 3, 3, [ 0, 1]) 
my_model.set_hop(-0.5*l, 0, 1, [ 1, 0]) 
my_model.set_hop(0.5*l, 0, 1, [-1, 0]) 
my_model.set_hop(0.5*l*1j, 0, 1, [ 0, 1]) 
my_model.set_hop(-0.5*l*1j, 0, 1, [ 0,-1]) 
my_model.set_hop(0.5*l, 2, 3, [ 1, 0]) 
my_model.set_hop(-0.5*l, 2, 3, [-1, 0]) 
my_model.set_hop(-0.5*l*1j, 2, 3, [ 0, 1]) 
my_model.set_hop(0.5*l*1j, 2, 3, [ 0,-1]) 
my_model.set_hop(J*(nx-1j*ny), 0, 1, [ 0, 0]) 
my_model.set_hop(J*(-nx+1j*ny), 2, 3, [0, 0]) 

# print tight-binding model
my_model.display()

#plot_bands(my_model)

niahe(my_model)
