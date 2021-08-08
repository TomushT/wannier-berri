import numpy as np
import matplotlib.pyplot as plt

z1 = np.loadtxt('cumnas_model-NIAHE_iter-0010.dat', skiprows=1) 
z2 = np.loadtxt('cumnas_model-NIAHE_iter-0040.dat', skiprows=1) 

# Plot
fig, ax = plt.subplots()

ax.plot(z1[:,0],z1[:,32],color='grey', label='xyy-10')
ax.plot(z2[:,0],z2[:,32],color='r', label='xyy-40')
ax.set_xlim([-2,2])
ax.legend(fontsize = 14)

#ax.set_xlabel(r"$\sigma$ (bohr)",fontsize = 16)
#ax.set_ylabel(r"$E_g / E_g(\sigma = 5.67$ bohr)",fontsize = 16)

#ax.tick_params(axis='x', which='major', labelsize=14)
#ax.tick_params(axis='y', which='major', labelsize=14)

# make an PDF figure of a plot
fig.tight_layout()
#fig.savefig("bulk3D_gaps.pdf")
plt.show()

