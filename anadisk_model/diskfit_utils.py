import numpy as np
import scipy as sp
import copy


def congrid(a, newdims, method='linear', centre=False, minusone=False):

    #Grabbed from the SCIPY cookbook
    #http://wiki.scipy.org/Cookbook/Rebinning

    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = sp.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = sp.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = sp.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None

def spatially_bin_dataset(dataset, bin_width):

    new_dataset = dataset

    shp = dataset.input.shape

    nchannels = shp[0] #This will likely be 2 for pol mode or 37 for spec mode

    new_dims = [shp[1]//bin_width, shp[2]//bin_width]

    new_data = []
    new_centers = []
    for i in range(nchannels):
        new_data.append(congrid(dataset.input[i,:,:],new_dims))
        new_centers.append([dataset.centers[i,0]*new_dims[0]/shp[1], dataset.centers[i,1]*new_dims[1]/shp[2]])


    new_dataset.input = np.array(new_data)
    new_dataset.centers = np.array(new_centers)
    new_dataset._IWA = dataset._IWA*new_dims[0]/shp[1]
    if dataset.OWA is not None:
        new_dataset.OWA = dataset.OWA*new_dims[0]/shp[1]

    return new_dataset


def dist_circle(array, xcen=0.,ycen=0.):
    #This function returns a 2d distance array from xcen and ycen
    #Like the IDL function of the same name
    ndim = np.size(array.shape)
    
    if ndim > 2: 
        print("DIST_CIRCLE: Right now dist_circle on accepts 2D arrays")
        print("Returning")
        return -1

    Y,X = np.indices(array.shape)

    X -= int(xcen)
    Y -= int(ycen)

    R=np.sqrt( X**2 + Y**2 )

    return R

########################################################
def make_snr_map_pol(fits, fits_n, xcen=140, ycen=140, width=3):
    
    mask = (fits == fits) #The mask of good pixels

    snr_map=fits*0. #The output SNR map
    size=np.shape(fits)

    #Get the distance array
    rad = dist_circle(fits, xcen=xcen, ycen=ycen)
    rad *= mask #Apply the nan mask

    #Width is the size of the annulus
    for i in np.arange(size[0]/2./width-1)*width-1:
        pixels = np.where((rad > i - width/2.) & (rad < i + width/2.)) #What pixels are in the annulus?
        snr_map[pixels]=fits[pixels]/np.std(fits_n[pixels]) #The SNR will be the fits divided by the stddev in the annulus of the fits_n file
    snr_map[fits != fits] = np.nan
    return snr_map

########################################################
def make_snr_map_total_intensity(image, rads, theta_mask):
    snr_map = np.abs(np.copy(image))
    width = int(np.shape(image)[0]//2)
    std_array = np.zeros(width)

    for rad in range(width):
            rad_cond = (rads > rad - 0.5) & (rads < rad + 0.5)
            std_array[rad] = np.nanstd(image[ (~theta_mask) & rad_cond])
            snr_map[rad_cond] /= std_array[rad]
    return snr_map

#####################################################

def get_corr(data1, data2):   
    where_to_corr = (data1 == data1) & (data2 == data2) 
    #I believe this bit was copied and pasted from pyklip at some point. 
    covar_psfs=np.cov([data2[where_to_corr], data1[where_to_corr]])
    covar_diag = np.diagflat(1./np.sqrt(np.diag(covar_psfs)))
    corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)
    return corr_psfs[0,1]

#####################################################

def collapse_spec_dataset(dataset):
    #This function collapses the wavelength channels of a spec mode cube by summing the
    #Input: 
    # dataset - a GPI dataset object. 

    output = copy.deepcopy(dataset)

    #Check to make sure it's not a pol mode cube
    if dataset.prihdrs[0]['DISPERSR'] == 'WOLLASTON':
        print("You input a pol mode dataset. You probably don't have to collapse it. Returning the original dataset")
        return dataset
    
    nfiles = np.size(np.unique(dataset.filenames))
    nchannels = 37 #Hardcoded for now. 

    # Average all spec cubes along wavelength axis.
    input_collapsed = []
    for i in range(nfiles):
        input_collapsed.append(np.nansum(dataset.input[i*nchannels:(i+1)*(nchannels)-1,:,:], axis=0))
    
    input_collapsed = np.array(input_collapsed)
    output.input = input_collapsed
    
    # Average centers of all wavelength slices and store as new centers.
    centers_collapsed = []
    sl = 0
    while sl < dataset.centers.shape[0]:
        centers_collapsed.append(np.mean(dataset.centers[sl:sl+nchannels], axis=0))
        sl += nchannels

    centers_collapsed = np.array(centers_collapsed)
    output.centers = centers_collapsed
    
    # Reduce dataset info from 37 slices to 1 slice.
    output.PAs = dataset.PAs[range(0,len(dataset.PAs),nchannels)]
    output.filenums = dataset.filenums[range(0,len(dataset.filenums),nchannels)]
    output.filenames = dataset.filenames[range(0,len(dataset.filenames),nchannels)]
    output.wcs = dataset.wcs[range(0,len(dataset.wcs),nchannels)]
    output.spot_flux = dataset.spot_flux[range(0,len(dataset.wcs),nchannels)]
    
    # Lie to pyklip about wavelengths.
    output.wvs = np.ones(input_collapsed.shape[0])

    return output


def call_gen_disk(theta, fits, sampling, mask):
    
    #logl Input parameters: 
    #   theta       -   the fit parameters 
    #   total_im    -   the post-klip total intensity image
    #   pol_im      -   the polarized image (usually Q_r, but could also be an unbiased P <- be careful about the noise)
    #   
    #   kl_mode     -   the kl mode to forward model. Just choose one. Don't be difficult. 
    #   sampling    -   the binning factor
    #   sigma       -   the sigma for a gaussian filter. 
    #   n_anchors   -   the number of anchors to use for the total intensity scattering functions. 
    #                   polarization fraction uses two less because we set the polarized scattering at 0 and 180 to be 0. 

    # Here the total intensity scattering phase function and the polarized fraction will be implemented as functions
    # based on 15 anchor points for each. 

    #Fit paramters: 
    r1 = mt.exp(theta[0])                            # The inner radius in AU. we fit for log(r1), so it's exp(log(r1)) here
    r2 = mt.exp(theta[1])                            # The outer radius in AU
    beta = theta[2]                                  # the radial powerlaw
    a_r = mt.exp(theta[3])                           # The scale height aspect ratio
    inc = np.degrees(np.arccos(theta[4]))            # The disk inclination in degrees
    pa = theta[5]                                    # The disk position angle in degrees
    dx = theta[6]                                    # The disk x offset in the disk plane in AU (see Millar-Blanchaer at al. 2016 for a good description)
    dy = theta[7]                                    # the disk y offset in the disk plane in AU
    tint_anchors = theta[8:8+n_anchors]            # the total intensity scattering function anchors. These will be in an unnormalized intensity unit
    pint_anchors0 = theta[8+n_anchors:8+2*n_anchors-2] # the polarized fraction scattering function anchors. #Here we'll use two less angles because we'll anchor 0 and 180 degrees to 0. 


    #Calculate the total intensity spline function
    angles = np.linspace(0, np.pi, n_anchors) # e.g. With 15 angles, we pin down the phase function every 12 degrees
    tint_func = interp1d(angles, tint_anchors, kind='cubic') #This one should be good. 

    #Calculate the polarized intensity spline function
    # Create a function so we can evaluate the total intensity phase function at any point.
    # tint_func = UnivariateSpline(angles, tint_anchors ,k=3, s = 0.1) #This one gives weird answers. 
    # tint_func = InterpolatedUnivariateSpline #This one might be alright, but needs testing. 
    pint_anchors = np.copy(tint_anchors)
    pint_anchors[0] = 0. #Pin down 0 and 180 degrees to 0. since we'll get 0 polarization fraction for pure forward or backward scattering
    pint_anchors[-1] = 0.
    pint_anchors[1:-1] = pint_anchors0
    # pint_func = UnivariateSpline(angles, pint_anchors ,k=3, s = 0.1) # Create a function so we can evaluate the polarized intensity phase function at any point.
    pint_func = interp1d(anlges, pint_anchors,kind='cubic')
    
    #generate the model
    ims = gen_disk(tint_func, pint_func, R1=r1,R2=r2, beta=beta, aspect_ratio=a_r, inc=inc, pa=pa, distance=distance, \
        dx=dx, dy=dy, sampling=sampling,mask=mask)

    t_im = ims[0] #The total intensity image
    p_im = ims[1] #The polarized intensity image 

    return t_im,p_im 

def subwindow_dataset(dataset,bounds,new_psfcenx=140,new_psfceny=140):
    '''
    Reduce the size of a pyklip dataset, only using the pixels within the 'bounds'

    Inputs:
    dataset - a pyklip dataset
    bounds  - a 4 element list that gives the bounds as [xlow,xhigh,ylow,yhigh]
    Keywords:
    new_psfcenx - the new psfcenter x-coordinate
    new_psfceny - the new psfcenter y-coordinate
    '''

    #Get the bounds
    xlow,xhigh,ylow,yhigh = bounds

    # The number of images
    nim = dataset.input.shape[0]

    #Cutout the subwindow
    dataset.input = dataset.input[:,ylow:yhigh,xlow:xhigh]

    #Shift the centers. 
    new_centers = copy.deepcopy(dataset.centers)
    dataset.centers[:,0] -= xlow
    dataset.centers[:,1] -= ylow


