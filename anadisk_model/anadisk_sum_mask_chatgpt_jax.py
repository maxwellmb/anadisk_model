##### May 6, 2018 #####
# The goal of this effort is to re-generalize things. In particular, I want only one function, that will generate x disks, based on x scattering phase functions. 


# Analytic disk model
# This version is being developed to create a version that is 3D and can be summed along the line of sight instead of intergrated. 

import matplotlib.pyplot as plt
import numpy as np
import math as mt
from datetime import datetime
# from numba import jit
# from numba import vectorize,float64
# from scipy.interpolate import interp1d
import scipy.ndimage.filters as snf
import copy
import jax
import jax.numpy as jnp
from jax import jit
# from jax.scipy.interpolate import interp1d
from jax.numpy import interp



###################################################################################
####################### Some Built-In Scattering Functions ########################
###################################################################################

@jit
def hgg_phase_function(phi, g):
    #Inputs:
    # g - the g 
    # phi - the scattering angle in radians
    g = g[0]

    cos_phi = jnp.cos(phi)
    g2p1 = g**2 + 1
    gg = 2*g
    k = 1./(4*jnp.pi)*(1-g*g)
    return k/(g2p1 - (gg*cos_phi))**1.5

@jit
def hgg_phase_function_alt(phi, g=0.4):
    #Inputs:
    # g - the g 
    # phi - the scattering angle in radians
    # g #= g[0]

    cos_phi = jnp.cos(phi)
    g2p1 = g**2 + 1
    gg = 2*g
    k = 1./(4*jnp.pi)*(1-g*g)
    return k/(g2p1 - (gg*cos_phi))**1.5

# This function will accept a vector of scattering angles, and a vector of scattering efficiencies 
# and then compute a cubic spline that fits through them all. 
#@jit

def phase_function_spline(phi, efficiency):
    #Input arguments: 
    #     angles - in radians
    #     efficiencies - from 0 to 1
    return interp(phi, efficiency)


@jit
def rayleigh(phi, args):
    #Using Rayleigh scattering (e.g. Eq 9 in Graham2007+)
    pmax = args[0]
    return pmax*jnp.sin(phi)**2/(1+jnp.cos(phi)**2)

#@jit
# @vectorize([float64(float64, float64)])
# def modified_rayleigh(phi, args):
#     #Using a Rayleigh scattering function where the peak is shifted by args[1]
#     pmax = args[0] #The maximum scattering phase function
#     return pmax*np.sin(phi-np.pi/2+args[1])**2/(1+np.cos(phi-np.pi/2+args[1])**2)
##########################################################################################
############ Gen disk and integrand for a 1 scattering function disk #####################
##########################################################################################

# @jit
def calculate_disk(xci,zpsi_dx,yy_dy2,x2,z2,x,zpci,xsi,a_r,R1,R2, beta_in, beta_out,scattering_function_list):
    '''
    # compute the brightness in each pixel
    # see analytic-disk.nb - originally by James R. Graham
    '''
    
    #The 'x' distance in the disk plane
    xx=(xci + zpsi_dx)

    #Distance in the disk plane
    d1_2 = yy_dy2 + xx**2
    d1 = jnp.sqrt(d1_2)

    #Total distance from the center 
    d2 = x2 + yy_dy2 + z2

    #The line of sight scattering angle
    cos_phi=x/d2**0.5
    phi = jnp.arccos(cos_phi)

    #The scale height exponent
    zz = (zpci - xsi)
    expo = (zz**2)/d1_2

    #The power law here has been moved from previous versions of anadisk so that we only calculate it once
    int1 = (R1/d1)**beta_in * ((d1 < R1) + (d1 >= R1)*(d1 <= R2)*(R1/d1)**(beta_out - beta_in))
    int1 = jax.lax.select(d1 > R2, int1*0., int1)
    int1 = jax.lax.select(d2 == 0., int1*0., int1)

    #Set up the vertical profile
    int2 = jnp.exp( (0.5/a_r**2)*expo) / int1
    #Include the diluation of starlight. 
    int3 = int2 * d2 

    #Vectorize the computation over scattering_function_list
    out = jnp.array([sf(phi)/int3 for sf in scattering_function_list]).T
    # out=sf(phi)/int3

    return out



#@jit
def generate_disk(scattering_function_list,
    R1=74.42, R2=82.45, beta_in=-7.5,beta_out=1.0, aspect_ratio=0.1, inc=76.49, pa=30, distance=72.8, 
    psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0., los_factor = 4, dim = 281.,pixscale=0.01414):
    '''

    Keyword Arguments:
    pixscale    -   The pixelscale to be used in "/pixel. Defaults to GPI's pixel scale (0.01414)
    dim         -   The final image will be dim/sampling x dim/sampling pixels. Defaults to GPI datacube size.

    '''

    #The number of input scattering phase functions and hence the number of disks to generate
    n_sf = len(scattering_function_list) 
    # n_sf = 1

    ###########################################
    ### Setup the initial coordinate system ###
    ###########################################
    npts=int(np.floor(dim/sampling)) #The number of pixels to use in the final image directions
    npts_los = int(los_factor*npts) #The number of points along the line of sight 

    factor = (pixscale*distance)*sampling # A multiplicative factor determined by the sampling. 

    # In all of the following we only want to do calculations in part of the non-masked part of the array
    # So we need to replicate the mask along the line of sight.  
    if mask is not None:
        mask = np.dstack([~mask]*npts_los)
    else: 
        mask = np.zeros([npts,npts],dtype=bool)
        mask = np.dstack([~mask]*npts_los)

    #Set up the coordiname arrays
    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight 
    # +ve y is going right from the center
    # +ve z is going up from the center
    z,y,x = np.indices([npts,npts,npts_los])

    #Center the line-of-sight coordinates on the disk center. 

    ## THIS WAY DOESN'T WORK. IT CREATES INCONCISTENT RESULTS. 
    # x[mask] = x[mask]/(npts_los/(2*R2)) - R2 #We only need to calculate this where things aren't masked. 

    #THIS WAY IS A BIT SLOWER, BUT IT WORKS.
    #Here we'll try just a set pixel scale equal to the y/z pixel scale divided by the los_factor
    x = x.astype('float')
    x[mask] = x[mask] - npts_los/2. #We only need to calculate this where things aren't masked. 
    x[mask] *=factor/los_factor

    #Setting up the output array
    threeD_disk = np.zeros([npts,npts,npts_los,n_sf]) + np.nan
    
    #####################################
    ### Set up the coordinate system ####
    #####################################

    #Inclination Calculations
    incl = np.radians(90-inc)
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination

    # x*cosine i and x*sin i 
    xci = x[mask] * ci
    xsi = x[mask] * si

    #Position angle calculations
    pa_rad=np.radians(90-pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)
    a_r=aspect_ratio

    # Rotate the coordinates in the image frame for the position angle
    # yy=y[mask]*(cos_pa*factor) - z[mask] * (sin_pa*factor) - ((cos_pa*npts/2*factor)-sin_pa*npts/2*factor) #Rotate the y coordinate by the PA
    # zz=y[mask]*(sin_pa*factor) + z[mask] * (cos_pa*factor) - ((cos_pa*npts/2*factor)+sin_pa*npts/2*factor) #Rotate the z coordinate by the PA

    yy=y[mask]*(cos_pa*factor) - z[mask] * (sin_pa*factor) - ((cos_pa*psfcenx*factor)-sin_pa*psfceny*factor) #Rotate the y coordinate by the PA
    zz=y[mask]*(sin_pa*factor) + z[mask] * (cos_pa*factor) - ((cos_pa*psfceny*factor)+sin_pa*psfcenx*factor) #Rotate the z coordinate by the PA


    #The distance from the center in each coordiate squared
    # y2 = np.square(yy)
    z2 = np.square(zz)
    x2 = np.square(x[mask])

    #This rotates the coordinates in and out of the sky
    zpci=zz*ci #Rotate the z coordinate by the inclination. 
    zpsi=zz*si
    
    #Subtract the stellocentric offset
    zpsi_dx = zpsi - dx
    yy_dy = yy - dy

    #The distance from the stellocentric offset squared
    yy_dy2=np.square(yy_dy)

    # ########################################################
    # ### Now calculate the actual brightness in each bin ####
    # ######################################################## 

    # threeD_disk[:,:,:][mask] = calculate_disk(xci,zpsi_dx,yy_dy2,x2,z2,x[mask],zpci,xsi,aspect_ratio,R1,R2,beta_in,beta_out,scattering_function_list)
    threeD_disk[:,:,:][mask] = jit(calculate_disk(xci,zpsi_dx,yy_dy2,x2,z2,x[mask],zpci,xsi,aspect_ratio,R1,R2,beta_in,beta_out,scattering_function_list))


    return np.sum(threeD_disk,axis=2)

########################################################################################
########################################################################################
########################################################################################
if __name__ == "__main__":

    sampling = 1

    #With two HG functions
    # sf1 = hgg_phase_function
    # sf1_args = [0.8]

    # sf2 = hgg_phase_function
    # sf2_args = [0.3]

    # im = gen_disk_dxdy_2disk(sf1, sf2,sf1_args=sf1_args, sf2_args=sf2_args, sampling=2)
    start_time = datetime.now()
    #With splines fit to HG function + rayleigh
    n_points = 20
    angles = np.linspace(0,np.pi,n_points)
    g = 0.8
    pmax = 0.3

    hg = hgg_phase_function(angles,[g])
    #interp1d doesn't exist in JAX in the same way so for testing I'm just using HGG. 
    f = hgg_phase_function_alt 
    # f = phase_function_spline(angles,hg)

    pol = hg*rayleigh(angles, [pmax])
    f_pol = hgg_phase_function_alt 
    # f_pol = phase_function_spline(angles,pol)

    y,x = np.indices([281,281])
    rads = np.sqrt((x-140)**2+(y-140)**2)
    mask = (rads > 90)

    # f = lambda x: hgg_phase_function(x,[g])
    start1 = datetime.now()
    # im = generate_disk(f,los_factor=1,mask=mask)
    im = generate_disk([f,f_pol],los_factor=1,mask=mask)
    end = datetime.now()
    print(end-start1)

    im1 = im[:,:,0]
    im2 = im[:,:,1]
    # before_gauss = datetime.now()
    # #Testing 
    sigma = 1.3/sampling
    im1 = snf.gaussian_filter(im1,sigma)
    im2 = snf.gaussian_filter(im2,sigma)
    end_time = datetime.now()
    print("Total Time: {}".format(end_time-start_time))
    # print("Time to smooth with Gaussian: {}".format(datetime.now() - before_gauss))
    # #No smoothing
    # im1 = im[0]
    # im2 = im[1]

    fig = plt.figure()

    fig.add_subplot(121)
    plt.imshow(im1)
    fig.add_subplot(122)
    plt.imshow(im2)
    # plt.imshow(im)
    plt.show()
# return x

