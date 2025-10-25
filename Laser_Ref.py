# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:17 2023

@author: shaun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Reads in the scattered field for the various points and the reference field
# Then performs the process as described in the Novotny paper to find the IRP plots
# However, this code uses the laser as the reference field
# =============================================================================

import numpy as np

from scipy.interpolate import griddata
from scipy.constants import speed_of_light, epsilon_0, hbar, pi
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cmath import phase
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import matplotlib.patheffects as PathEffects



# These are the varibles to edit
File_Path = r'single_beam' # Folder where the data is stored
File_Name = 'H1' # Name of the data file
Plot_title = False
Rotation  = True # Set to true if you ant to see the roational info


# These first two functions are mostly about reading in the data
def valid(line):
    # Returns False if the first character in the line is '#'
    if line.startswith("#"): return False
    return True

def extract(line):
    # Pulls the values out of the file
    values = line.strip().split()
    return [v for v in values]

def points(r,t,p):
    from numpy import sin, cos
    # Converts the spherical polars to cartesian coords
    x = r*sin(t)*cos(p)
    y = r*sin(t)*sin(p)
    z = r*cos(t)
    
    return x,y,z

def build_E_field(data,tag):
    # Takes in the feild components at each location and converts them to an
    # array of complex number
    output = []
    # Loops through each line in the data set (maybe a quicker way to do this, but I'll find it later (?))
    for line in data:
        # Only does the calculations for the right translation
        if line[4]== tag:
            Ex = float(line[6])  + 1j*float(line[7])
            Ey = float(line[8])  + 1j*float(line[9])
            Ez = float(line[10]) + 1j*float(line[11])
            output.append([Ex,Ey,Ez])
    return np.array(output)#/1e-6

def build_H_field(data,tag):
    # Takes in the feild components at each location and converts them to an
    # array of complex number
    output = []
    # Loops through each line in the data set (maybe a quicker way to do this, but I'll find it later (?))
    for line in data:
        # Only does the calculations for the right translation
        if line[4]== tag:
            Ex = float(line[12]) + 1j*float(line[13])
            Ey = float(line[14]) + 1j*float(line[15])
            Ez = float(line[16]) + 1j*float(line[17])
            output.append([Ex,Ey,Ez])
    return np.array(output)#/1e-6

def dfdx(f,x):
    return (f[1]-f[0])/(x[1]-x[0])

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
        k = 2*pi/WL # Wave number of the laser
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
E0 = 1 # incident E-field strength in V/micron

Z0 = 376.730313668


if Rotation:
    axis_list = ['z']#['x','y','z','thetax','thetay','thetaz']
    
else:
    axis_list = ['x','y','z']

# Sets up the NAs to calculate the detection efficencies
NA_L = 0.041*2
NA_R = 0.041*2
theta_L = np.arcsin(NA_L) # Angle covered by LEFT lens
theta_R = np.arcsin(NA_R) # Angle covered by RIGHT lens
    
with open(f'Data_Files/{File_Path}/{File_Name}.EvalPoints.scattered','r') as file:
    lines = (line.strip() for line in file if valid(line))
    data_sca = np.array([extract(line) for line in lines])
    
# with open(f'{File_Name}.EvalPoints.total','r') as file:
#     lines = (line.strip() for line in file if valid(line))
#     data_tot = np.array([extract(line) for line in lines])
    
    
tag_tmp = data_sca[0,4] # only takes the 
coords_str = [vals[:3] for vals in data_sca if vals[4]==tag_tmp]
x = []
y = []
z = []

for point in coords_str:
    x.append(point[0])
    y.append(point[1])
    z.append(point[2])


# Converts the measurment locations to floats
xs = [float(xi) for xi in x]
ys = [float(yi) for yi in y]
zs = [float(zi) for zi in z]

# Sets up an array containing the measurment locations
coords = np.array([xs,ys,zs]).T#/1e6

# Redraws the s_imp to be in spherical polars
# These first few lines are just picking equal spacing on a unit sphere (see: https://mathworld.wolfram.com/SpherePointPicking.html)
u = np.linspace(0,1,200)
v = np.linspace(1,0,200)
theta = np.arccos(2*v-1)
phis = 2*np.pi*u

# Mesh grid these things to get 2D polts of the angles
THETA, PHI = np.meshgrid(theta, phis)
# Find how x,y, and z vary across the unit sphere
X,Y,Z = points(1,THETA,PHI)
dTHETA = np.gradient(THETA,axis=1)

dPHI = np.gradient(PHI,axis=0)
domega = (-np.gradient(np.cos(THETA),axis=1)*dPHI).flatten() 

# Start on detection efficeny stuff
# This block of code finds the number of array elements to cover the lens NA

# The difference between each point in the theta space
info_dtheta = np.gradient(theta)

# We only look at theta as we're assuming a circular lens, so we go over theta_NA
# in the theta axis and 2pi in the phi axis

# Finds the number of elements for the integral in the left detection efficency calculation
R_count = 0
angle_R = 0

while angle_R < theta_R:
    angle_R += info_dtheta[R_count]
    R_count+=1
    
# Finds the number of elements for the integral in the right detection efficency calculation
# The LEFT lens is centered on theta = pi which is the end of the array
L_count = len(info_dtheta)-1
angle_L = 0

while angle_L < theta_L:
    angle_L += info_dtheta[L_count]
    L_count-=1

flux_figs = {} # Save the figures so we can look at them in the matplot window
eta_vals = {}  # Save the detection efficencies so we can see them

def Fisher_Info(mu,angle=False):
    
    # Get the +ve and -ve translations for each axis
    trans_str = ['-'+mu,'+'+mu]
    
    if angle:
        # Rotate by 1 degree in each direction
        trans_flt = [-1,+1]
    else:
        # Move by lambda/100 in each direction
        trans_flt = [-0.0155e-6,+0.0155e-6]
    
    # Scattered feilds at each detector
    E_Sca = {loc: np.conj(build_E_field(data_sca,loc))  for loc in trans_str}
    H_Sca = {loc:         build_H_field(data_sca,loc)  for loc in trans_str}
    # Finds the scattered field at mu=0 (no translation/rotation)
    E_0   = np.conj(build_E_field(data_sca,'0'))
    H_0   =         build_H_field(data_sca,'0')
    S_0   = np.cross(E_0,H_0)
    
    # E_Ref = build_E_field(data_tot,trans_str[0]) - build_E_field(data_sca,trans_str[0]) 
    # H_Ref = build_H_field(data_tot,trans_str[0]) - build_H_field(data_sca,trans_str[0]) 
    
    S_FI_temp = [] # stores the FI flux at each detector
    
    for det_pos,x in enumerate(xs):
        print(f'\rDetector {det_pos+1:05}/{len(xs)}',end='')
        Ex_sca = []
        Ey_sca = []
        Ez_sca = []
        
        Hx_sca = []
        Hy_sca = []
        Hz_sca = []
        
        Sx_sca = []
        Sy_sca = []
        Sz_sca = []
        
        S_Sca = []
        
        Unit_Vec = coords[det_pos]/1e6
        
        if Unit_Vec[2] < 0:
            # E_Ref = np.array([0,0,0])
            # H_Ref = np.array([0,0,0])
            x_loc,y_loc,z_loc = coords[det_pos]
            E_Ref = np.array([np.conj(Ref_Beam(x_loc,y_loc,z_loc,W0,E0,WL,-1)),
                              0,
                              0])
            
            H_Ref = np.array([0,
                              Ref_Beam(x_loc,y_loc,z_loc,W0,-E0,WL,-1)/Z0,
                              0])
        else:
            # E_Ref = np.array([0,0,0])
            # H_Ref = np.array([0,0,0])
            x_loc,y_loc,z_loc = coords[det_pos]
            E_Ref = np.array([np.conj(Ref_Beam(x_loc,y_loc,z_loc,W0,E0,WL,+1)),
                              0,
                              0])
            
            H_Ref = np.array([0,
                              Ref_Beam(x_loc,y_loc,z_loc,W0,E0,WL,+1)/Z0,
                              0])
            
        S_ref = np.cross(E_Ref,H_Ref)
        

            
        # Finds the power at the detector as a function of satterer position
        for trans in trans_str:
            # Finds the total x,y,z components for the field at the dectecor for
            # for the detector at 'det_pos' with the particle at location 'trans'
            Ex,Ey,Ez = E_Sca[trans][det_pos]
            Hx,Hy,Hz = H_Sca[trans][det_pos]
            Sx,Sy,Sz = np.cross(E_Sca[trans][det_pos],H_Sca[trans][det_pos])
            
            Ex_sca.append(Ex)
            Ey_sca.append(Ey)
            Ez_sca.append(Ez) 
            
            Hx_sca.append(Hx)
            Hy_sca.append(Hy)
            Hz_sca.append(Hz) 
            
            Sx_sca.append(Sx)
            Sy_sca.append(Sy)
            Sz_sca.append(Sz)
    

        # Usimple derivitive works for linear change in field (small changes in mu)
        dE = np.array([dfdx(Ex_sca,trans_flt), dfdx(Ey_sca,trans_flt), dfdx(Ez_sca,trans_flt)])
        dH = np.array([dfdx(Hx_sca,trans_flt), dfdx(Hy_sca,trans_flt), dfdx(Hz_sca,trans_flt)])
    
        

        dS_sca = np.array([dfdx(Sx_sca,trans_flt),dfdx(Sy_sca,trans_flt),dfdx(Sz_sca,trans_flt)])
        
        
        

        # makes the fisher info at each detector
        # S_FI_vec = 
        
        photon_flux_derivitive = np.real(dS_sca + np.cross(dE,H_Ref) + np.cross(E_Ref,dH))@Unit_Vec*domega[det_pos]
        photon_flux_value      = 2*hbar*omega*np.real(S_0[det_pos] + np.cross(E_0[det_pos],H_Ref) + np.cross(E_Ref,H_0[det_pos]) +S_ref)@Unit_Vec*domega[det_pos]
        S_FI_temp.append(photon_flux_derivitive**2/photon_flux_value)
        # S_FI_temp.append((np.real(dS_sca)@Unit_Vec)**2/(np.real(S_0[det_pos])@Unit_Vec))
    return np.array(S_FI_temp)


plot_path = f'Plots/Laser_Ref/{File_Path}/{File_Name}'
Path(plot_path).mkdir(parents=True, exist_ok=True) 

figs = {}
#fig,axes = plt.subplots(figsize=(5,15),nrows=len(axis_list),subplot_kw={'projection':'3d'})
plt.rcParams['mathtext.fontset']="cm"
titles  = {'x':"$x$",'y':"$y$",'z':"$z$",'thetax':"$\\theta_x$",'thetay':"$\\theta_y$",'thetaz':"$\\theta_z$"}
if File_Name == 'H3':
    theta_z_zoom = 0.75
else:
    theta_z_zoom = 0.45
Lim_Mod = {'x':0.7,  'y':0.8,  'z':0.7,  'thetax':0.85,          'thetay':0.75,          'thetaz':theta_z_zoom}

for i,motionaxis in enumerate(axis_list):
    print(f'Working on motion in the {motionaxis} axis')

    if motionaxis=='thetax' or motionaxis=='thetay' or motionaxis=='thetaz':
        S_FI = Fisher_Info(motionaxis,angle=True)
    else:
        S_FI = Fisher_Info(motionaxis,angle=False)
    # Remeshes the spectral density from a list of locations to a 2D grid at the locations in X,Y,Z
    S_FI_grid = griddata(coords, np.array(S_FI), (X,Y,Z), method='nearest')
    

    
    # Finds the information radiated to each angle (See the start of Sec II C)
    # Finds the information radiated to each angle (See the start of Sec II C)
    I = S_FI_grid/np.trapz(np.trapz(S_FI_grid,axis=1)) #
    
    print('\nFound Info Plot')

    # So, we want to integrate over the angle theta_L
    # int_0^2pi int_0^theta_L I sin(theta) dtheta dphi

    # Detection effiencies for the RIGHT and LEFT lens respectivly
    eta_R = np.trapz(np.trapz((I[:,:R_count+1]).T,axis=0))
    
    eta_L = np.trapz(np.trapz((I[:,L_count:]).T,axis=0))

    # Save the etas
    eta_vals[motionaxis] = {'left': eta_L, 'right': eta_R}
    

    
    # Put the plots together
    Xi = X*I
    Yi = Y*I
    Zi = Z*I
    d = np.sqrt(Xi**2+Yi**2+Zi**2)
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
    
    xmin = (mid_x - max_range)*Lim_Mod[motionaxis]
    xmax = (mid_x + max_range)*Lim_Mod[motionaxis]
    ymin = (mid_y - max_range)*Lim_Mod[motionaxis]
    ymax = (mid_y + max_range)*Lim_Mod[motionaxis]
    zmin = (mid_z - max_range)*Lim_Mod[motionaxis]
    zmax = (mid_z + max_range)*Lim_Mod[motionaxis]
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_axis_off()
    
    # Label the motion axis
    if Plot_title:
        txt = ax.text2D(0,0.8,titles[motionaxis],size=60,transform=ax.transAxes)
        #txt = ax.text(Xi.max(),Yi.max(),Zi.min(),s=titles[motionaxis],size=60)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    
    
    txt_size = 35
    if eta_L > 1/9:
        txt = ax.text2D(0.2,0.2,s=f'{eta_L:.2f}',size=txt_size,color="b",horizontalalignment='center',transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    elif eta_L >= 0.01:
        txt = ax.text2D(0.2,0.2,s=f'{eta_L:.2f}',size=txt_size,color="k",horizontalalignment='center',transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    else:
        txt = ax.text2D(0.2,0.2,s=f'{eta_L:.1e}',size=txt_size,color="k",horizontalalignment='center',transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    
    
    
    if eta_R > 1/9:
        txt = ax.text2D(0.8,0.2,s=f'{eta_R:.2f}',size=txt_size,color="b",horizontalalignment='center',transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    elif eta_R >= 0.01:
        txt = ax.text2D(0.8,0.2,s=f'{eta_R:.2f}',size=txt_size,color="k",horizontalalignment='center',transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    else:
        txt = ax.text2D(0.8,0.2,s=f'{eta_R:.1e}',size=txt_size,color="k",horizontalalignment='center',transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    
    
    
    
    #fig.tight_layout()
    fig.savefig(f'{plot_path}/{i+1}_{motionaxis}_Plot.pdf')
    fig.savefig(f'{plot_path}/{i+1}_{motionaxis}_Plot.png')
    figs[motionaxis] = fig
    #plt.close()


















