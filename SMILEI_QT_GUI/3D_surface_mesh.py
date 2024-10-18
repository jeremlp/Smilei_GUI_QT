import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import numpy as np
from numpy import sqrt, pi, exp, cos, sin,arctan2
from skimage import measure
# from skimage.measure import mesh_reduce
from scipy.interpolate import interpn

l0 = 2*pi
plt.close("all")
cluster_path = os.environ["SMILEI_CLUSTER"]

PATH = rf"{cluster_path}\plasma_OAM_XDIR_NO_IONS_DIAGS"

anim_ms = 100

S = happi.Open(PATH)
a0 = S.namelist.a0
Lx = S.namelist.Lx
Ltrans = S.namelist.Ltrans
Ly,Lz = Ltrans, Ltrans
w0 = S.namelist.w0


def map_colors(p3dc, func, cmap='RdBu'):
    """
Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s],y[s],z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T

    # compute the function in the barycentres
    # print(xb,yb,zb)
    x = np.arange(0,Bx.shape[0])
    y = np.arange(0,Bx.shape[1])
    z = np.arange(0,Bx.shape[2])

    values = interpn((x,y,z), Bx, np.array([xb, yb, zb]).T,bounds_error =False)

    values[np.isnan(values)] = 0
    max_val = np.abs(iso_value)*1.25
    values[np.abs(values)>max_val] = np.sign(values[np.abs(values)>max_val])*max_val

    # print(np.abs(values))
    # values = func(xb, yb, zb)

    # usual stuff
    norm = Normalize()
    colors = get_cmap(cmap)(norm(values))
    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)

#==================================================
# MAGNETIC FIELD BZ
#==================================================
Bx_diag = S.Probe(0,"Bx")
# Bx_diag = S.Probe("3D","Bx")


Bx_diag.toVTK()

"""

Bx = np.array(Bx_diag.getData())[-1,:88]
print(Bx.shape)

paxisX,paxisY,paxisZ = Bx_diag.getAxis("axis1")[:,0],Bx_diag.getAxis("axis2")[:,1],Bx_diag.getAxis("axis3")[:,2]


X,Y,Z = np.meshgrid(paxisX,paxisY-Ltrans/2,paxisZ-Ltrans/2,indexing="ij")


iso_value = -0.01  # Define the isosurface value to visualize
vertices, faces, _, _ = measure.marching_cubes(Bx, level=iso_value, step_size=5)
# vertices, faces = mesh_reduce(vertices, faces, ratio=0.3)



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d', azim=-160)

# Plot the isosurface using the extracted vertices and faces
p3dc = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, alpha=0.5)


# Vi = interpn((paxisX,paxisY,paxisZ), Bz, np.array([xi,yi,zi]).T)



mappable = map_colors(p3dc, Bx, 'RdYlBu')
plt.colorbar(mappable,ax=ax)"""















azeazeeazaze

# Define a scalar field


# Create 3D plot


# Plot 3D surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, zorder=10)  # Higher zorder for surface
# Project onto the xy plane
ax.contourf(X, Y, Z, zdir='z', offset=-1, cmap='viridis', zorder=1, alpha=0.5)
# Project onto the xz plane
ax.contourf(X, Z, Y, zdir='y', offset=5, cmap='plasma', zorder=1, alpha=0.5)

# Project onto the yz plane
ax.contourf(Z, Y, X, zdir='x', offset=-5, cmap='RdYlBu', zorder=1, alpha=0.5)

# Set limits for axes
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-1, 1)

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Scalar Field with Projections')

plt.show()
