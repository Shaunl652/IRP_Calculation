# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 16:33:10 2025

@author: shaun
"""
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from scipy.constants import speed_of_light, epsilon_0, hbar, pi

def points(r,t,p):
    from numpy import sin, cos
    # Converts the spherical polars to cartesian coords
    x = r*sin(t)*cos(p)
    y = r*sin(t)*sin(p)
    z = r*cos(t)
    
    return x,y,z



def Ref_Beam(x,y,z,w0,E0,WL,direction):
    """
    Produces a Gaussian beam to act as a reference
    Uses Paraxial Approx from: https://en.wikipedia.org/wiki/Gaussian_beam

    Parameters
    ----------
    x : FLOAT
        x measurement location.
    y : FLOAT
        y measurement loaction.
    z : FLOAT
        z measurement location.
    w0 : FLOAT
        Beam waist.
    E0 : FLOAT
        Electric field amplitude.
    WL : FLOAT
        Laser wavelength.
    direction : INT
        The direction of propogation. +1 for +z and -1 for -z.

    Returns
    -------
    FLOAT
        The real part of the electric field at the location (x,y,z).

    """
        
    
    r = np.sqrt(x**2+y**2) # Radial distance from beam centre
    zR = pi*w0**2/WL # Rayleigh Range
    Inv_Curve = z/(z**2 + zR**2) # Inverse of the radius of curvetrure
    
    w   = w0*np.sqrt(1+(z/zR)**2) # Beam size as a function of z position
    phi = np.arctan(z/zR)         # Gouy phase shift as a function of z posiiton
    
    term1 = E0*w0/w
    term2 = np.exp(-r**2/w**2)
    if direction < 0:
        k = -2*pi/WL # Wave number of the laser
        term3 = np.exp(-1j* (k*z + Inv_Curve*k*r**2/2 + phi) )
    else:
        k = 2*pi/WL # Wave number of the laser
        term3 = np.exp(-1j* (k*z + Inv_Curve*k*r**2/2 - phi) )
    
    return term1*term2*term3
    
    

# Keep all of these in microns for now
WL = 1.55 # Laser wavelength
k = 2*pi/WL # wave number
omega = k*speed_of_light
W0 = 12 # Laser waist
E0 = 10000 # incident E-field strength in V/micron

Z0 = 376.730313668


u = np.linspace(0,1,200)
v = np.linspace(1,0,200)
theta = np.arccos(2*v-1)
phis = 2*np.pi*u

# Mesh grid these things to get 2D polts of the angles
THETA, PHI = np.meshgrid(theta, phis)
# Find how x,y, and z vary across the unit sphere
X,Y,Z = points(1e6,THETA,PHI)

beam_back = Ref_Beam(X*1e6,Y*1e6,Z*1e6,W0,E0,WL,-1)

I = np.abs(beam_back)


Xi = X*I
Yi = Y*I
Zi = Z*I
d = np.sqrt((Xi)**2+(Yi)**2+(Zi)**2)
d_norm = d/d.max()

# Plots the information patteren


fig,ax = plt.subplots(subplot_kw={'projection':'3d'})

ax.plot_surface(Xi,Yi,Zi,facecolors=plt.cm.jet(d_norm),shade=False)
ax.set(xlabel='x',ylabel='y',zlabel='z')
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(d)
#fig.colorbar(m,ax=ax)

# Set the view angle
ax.view_init(elev=20,azim=30,vertical_axis='x')

# Set the axis limmits to be the same to see the right shape
max_range = np.array([Xi.max()-Xi.min(), Yi.max()-Yi.min(), Zi.max()-Zi.min()]).max() / 2.0

# Finds the mid points
mid_x = (Xi.max()+Xi.min()) * 0.5
mid_y = (Yi.max()+Yi.min()) * 0.5
mid_z = (Zi.max()+Zi.min()) * 0.5

# Set the limits
Lim_Mod = 0.8
xmin = (mid_x - max_range)*Lim_Mod
xmax = (mid_x + max_range)*Lim_Mod
ymin = (mid_y - max_range)*Lim_Mod
ymax = (mid_y + max_range)*Lim_Mod
zmin = (mid_z - max_range)*Lim_Mod
zmax = (mid_z + max_range)*Lim_Mod

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)