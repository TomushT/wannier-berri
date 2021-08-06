import numpy as np
import matplotlib.pyplot as plt

z = np.loadtxt('CuMnAs-NIAHE_iter-0010.dat', skiprows=1) 

# Plot
fig, ax = plt.subplots()

ax.plot(z[:,0],z[:,3],color='grey', label='10')
ax.set_xlim([6.6,7.6])
ax.legend(fontsize = 14)

#ax.set_xlabel(r"$\sigma$ (bohr)",fontsize = 16)
#ax.set_ylabel(r"$E_g / E_g(\sigma = 5.67$ bohr)",fontsize = 16)

#ax.tick_params(axis='x', which='major', labelsize=14)
#ax.tick_params(axis='y', which='major', labelsize=14)

# make an PDF figure of a plot
fig.tight_layout()
#fig.savefig("bulk3D_gaps.pdf")
plt.show()

