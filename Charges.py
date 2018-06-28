__author__ = 'enaghib'

import numpy as np
import re
import healpy as hp
import matplotlib.pyplot as plt



def charges_preevaluated(n = 1000, m = None, ignore_first_col = False):
    '''
    reads in the solution of the electrons on a sphere potential minimization
    :param n: number of tiles
    :param m: version of tiling (refer to the readme file in the directory of the data)
    :return: x,y,z coordinates of n points on the sphere
    '''

    if m is None:
        data =  np.loadtxt('solutions/n{}.txt'.format(n))
    else:
        data = np.loadtxt('solutions/n{}{}.txt'.format(n,m))

    data = np.transpose(data)
    if ignore_first_col:
        x = data[1]; y = data[2]; z = data[3]
    else:
        x = data[0]; y = data[1]; z = data[2]

    return x,y,z

def Muller_uniform_random(n):
    '''
    :param n: number of points
    :return:  x,y,z coordinates of n uniformly distributed points on the sphere
    '''
    a = np.random.normal(0,1,[3,n])
    norm_a = np.sqrt(np.power(a[0],2)+np.power(a[1],2)+np.power(a[2],2))
    x = np.divide(a[0],norm_a)
    y = np.divide(a[1],norm_a)
    z = np.divide(a[2],norm_a)
    return x,y,z

def healpix_points(nside = 8):
    '''
    :param nside: 2^N
    :return: x,y,z coordinates of n_pix points
    '''
    n_pix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(n_pix))

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x,y,z


def Marsaglia_uniform_random(n):
    #Marsaglia (1972)
    i = 0
    x=[]; y=[]; z=[]
    while(i<n):
        a = np.random.uniform(-1,1)
        b = np.random.uniform(-1,1)
        factor = a*a+b*b
        if  factor>= 1:
            continue
        x = np.append(x, 2*a*np.sqrt(1-factor))
        y = np.append(y, 2*b*np.sqrt(1-factor))
        z = np.append(z, 1 - 2*np.sqrt(factor))
        i +=1

    return x,y,z

def rotate_config(x,y,z, new_axis, theta = None):
    '''
    :param x: vector of x coordinates of a configuration on the sphere
    :param y: vector of y coordinates of a configuration on the sphere
    :param z: vector of z coordinates of a configuration on the sphere
    :param new_axis: axis of the rotation
    :param theta: angle of the rotation
    :return: new coordinates of the rotated configuration
    '''
    u_x = new_axis[0]; u_y = new_axis[1]; u_z = new_axis[2]
    if theta is None:
        theta = np.random.uniform(0, 2*np.pi)
    cos_theta = np.cos(theta); sin_theta = np.sin(theta)
    R = np.eye(3) #rotation matrix
    R[0,0] = cos_theta + u_x*u_x*(1-cos_theta)
    R[0,1] = u_x*u_y*(1-cos_theta) - u_z*sin_theta
    R[0,2] = u_x*u_z*(1-cos_theta) - u_y*sin_theta
    R[1,0] = u_y*u_x*(1-cos_theta) + u_z*sin_theta
    R[1,1] = cos_theta + u_y*u_y*(1-cos_theta)
    R[1,2] = u_y*u_z*(1-cos_theta) - u_x*sin_theta
    R[2,0] = u_z*u_x*(1-cos_theta) - u_y*sin_theta
    R[2,1] = u_z*u_y*(1-cos_theta) + u_x*sin_theta
    R[2,2] = cos_theta + u_z*u_z*(1-cos_theta)

    R = R.T
    new_x = R[0,0]*x + R[0,1]*y + R[0,2]*z
    new_y = R[1,0]*x + R[1,1]*y + R[1,2]*z
    new_z = R[2,0]*x + R[2,1]*y + R[2,2]*z
    return new_x, new_y, new_z


def hybrid_charge_rot(n, n_r = 4):
    '''
    A hybrid method to overcome the curse of dimensionality in the optimization of the charge potential
    :param n: number of charges, directly evaluated for minimal potential
    :param n_r: number of repetition of the primary configuration
    :return: x,y,z coordinates of a new configuration of size n*n_r
    '''
    x_prim, y_prim ,z_prim = charges_preevaluated(n)
    x_rot,y_rot,z_rot = Muller_uniform_random(n_r)

    x = []
    y = []
    z = []

    for x_r, y_r, z_r in zip(x_rot,y_rot,z_rot):
        new_x, new_y, new_z = rotate_config(x_prim,y_prim,z_prim,[x_r, y_r, z_r])
        x = np.append(x,new_x)
        y = np.append(y,new_y)
        z = np.append(z,new_z)

    return x,y,z


def tile_sphere(x,y,z, x_mesh, y_mesh, z_mesh, tile_radius = np.radians(9.62/2.)):
    '''
    receives the center of tilings, and a fine grid on the sphere, and decides which pixels are tiled with how many tiles
    :param x: vector of x coordinates of the center of the circular tiles
    :param y: vector of y coordinates of the center of the circular tiles
    :param z: vector of z coordinates of the center of the circular tiles
    :param x_mesh: vector of x coordinates of the center of the discretization mesh on the sphere
    :param y_mesh: vector of y coordinates of the center of the discretization mesh on the sphere
    :param z_mesh: vector of z coordinates of the center of the discretization mesh on the sphere
    :param tile_radius: radius of the tile in radians, if None, then radius of the LSST's field of view
    :return: 0 <= density <= n_tiles
    '''

    n_tiles = len(x)
    n_pix = len(x_mesh)
    density = np.zeros(n_pix)
    cartesian_distance_squared = np.square(np.sin(tile_radius))

    for x_pix, y_pix, z_pix, i in zip(x_mesh,y_mesh,z_mesh, range(n_pix)):
        delta_x = x - x_pix
        delta_y = y - y_pix
        delta_z = z - z_pix
        dist = np.square(delta_x) + np.square(delta_y) + np.square(delta_z)
        density[i] = np.sum(dist <= cartesian_distance_squared)
    #density = np.divide(density, n_tiles)
    return density

def uniformity_measure(density):
    '''
    :param density: receives discretized density and evaluates the uniformity of the coverage
    :return: variance, mean, median
    '''
    variance = np.var(density)
    mean = np.mean(density)
    median = np.median(density)
    return variance, mean, median

def plot_tiling(density, x_mesh, y_mesh, z_mesh, title = None):
    hp.mollview(density, title=title)
    plt.show()

def plot_histograms(density, nside):
    theta, phi = hp.pix2ang(nside, np.arange(len(density)))
    plt.hist(density)
    plt.subplot(211)



#run

# evaluate configuration
n = 1610 #number of the tiles
m =  None  #data version
ignore_first_col = False
x,y,z = charges_preevaluated(n, m, ignore_first_col)
#x,y,z = Muller_uniform_random(n)
##x,y,z = healpix_points(16)
#x,y,z = hybrid_charge_rot(n,10)
print(len(x))
# evaluate discretization default 512
x_mesh, y_mesh, z_mesh = healpix_points(512)

# evaluate density of each pixel
r_tile = np.radians(3.5/2.)
density = tile_sphere(x, y, z, x_mesh, y_mesh, z_mesh, r_tile)
np.save('solutions/n{}{}random'.format(n,m), density)

# evaluate uniformity of the tiling
variance, mean, median = uniformity_measure(density)
print(variance, mean, median)
# plot
plot_tiling(density, x_mesh, y_mesh, z_mesh, 'N = {}, r = {:1.3f} radians'.format(n,r_tile))



