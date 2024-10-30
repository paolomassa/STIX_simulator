import numpy as np


def ComputeVisibilitiesGaussian(xc, yc, FWHM_max, FWHM_min, flux, angle, u, v):
    """
    Function for computing the visibilities corresponding to a bi-dimensional Gaussian source
    
    INPUT
    xc: float
        x coordinate of the center of the Gaussian function
    
    yc: float
        y coordinate of the center of the Gaussian function
    
    FWHM_min: float
        min Full Width at Half Maximum (FWHM) of the Gaussian function
    
    FWHM_max: float
        max Full Width at Half Maximum (FWHM) of the Gaussian function
    
    flux: float
        intensity (i.e., total flux) of the gaussian function. It is the value of the integral of the Gaussian
    
    angle: float
        orientation angle (measured counterclockwise from the positive x-axis) of the Gaussian function
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX

    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX
    
    OUTPUT:
    Array of dimension 2*len(u) containing the real and the imaginary values of the visibilities
    """
    
    dtor = np.pi/180
    
    u_rot =  np.cos(angle * dtor) * u + np.sin(angle * dtor) * v
    v_rot = -np.sin(angle * dtor) * u + np.cos(angle * dtor) * v
    
    vis = np.zeros(2*len(u))

    vis[0:len(u)] = flux * np.exp(-(np.pi**2 / (4*np.log(2))) * \
                    ((u_rot * FWHM_max)**2 + (v_rot * FWHM_min)**2)) * \
                    np.cos(2 * np.pi * (xc * u + yc * v))
    
    vis[len(u):2*len(u)] = flux * np.exp(-(np.pi**2 / (4*np.log(2))) * \
                           ((u_rot * FWHM_max)**2 + (v_rot * FWHM_min)**2)) * \
                           np.sin(2 * np.pi * (xc * u + yc * v))
    
    return vis


def CreateGaussianSource(xc, yc, FWHM_min, FWHM_max, flux, angle, npix=129, pix=1):
    """
    Function for generating a bi-dimensional Gaussian source
    
    INPUT
    xc: float
        x coordinate of the center of the Gaussian function

    yc: float
        y coordinate of the center of the Gaussian function

    FWHM_min: float
        min Full Width at Half Maximum (FWHM) of the Gaussian function

    FWHM_max: float
        max Full Width at Half Maximum (FWHM) of the Gaussian function

    flux: float
        intensity (i.e., total flux) of the gaussian function. It is the value of the integral of the Gaussian

    angle: float
        orientation angle (measured counterclockwise from the positive x-axis) of the Gaussian function
    
    npix: int
        number of pixels in the rows and in the columns of the image. It must be odd (default, 129)
    
    pix: float
        pixel size (in arcsec). Default is 1
    
    OUTPUT:
    Array of dimension npix x npix containing the image of the bi-dimensional Gaussian source
    """

    x = np.linspace(-(npix - 1)/2, (npix - 1)/2, num=npix)*pix
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, npix, axis=0)

    y = np.linspace((npix - 1)/2, -(npix - 1)/2, num=npix)*pix
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, npix, axis=1)
    
    # Min and max standard deviation of the Gaussian from FWHM min and max
    sigma_min = FWHM_min / (2*np.sqrt(2*np.log(2)))
    sigma_max = FWHM_max / (2*np.sqrt(2*np.log(2)))
    
    # Roto-translation of the (x,y)-plane    
    dtor = np.pi/180. # conversion factor from degrees to radians
    
    x_rot =  np.cos(angle * dtor) * (x - xc) + np.sin(angle * dtor) * (y - yc)
    y_rot = -np.sin(angle * dtor) * (x - xc) + np.cos(angle * dtor) * (y - yc)
    
    gauss = flux/(2*np.pi*sigma_min*sigma_max) * np.exp(-x_rot**2/(2*sigma_max**2)-y_rot**2/(2*sigma_min**2))
    
    return gauss


def Fourier_matrix_STIX(u, v, n_pix, pix_size):
    """
    Function for creating the Fourier matrix to be used for computing the STIX visibility values from an image
    
    INPUTS:
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX
    
    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX
    
    n_pix: int
        number of pixels (rows/columns) used to discretize the considered Field-of-View (FOV)
    
    pix_size: float
        pixel size (in arcsec)
    """
    
    x = np.linspace(-(n_pix - 1)/2, (n_pix - 1)/2, num=n_pix)*pix_size
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, n_pix, axis=0)

    y = np.linspace((n_pix - 1)/2, -(n_pix - 1)/2, num=n_pix)*pix_size
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, n_pix, axis=1)
    
    dim = len(u)
    F = np.zeros((2*dim, n_pix*n_pix))
    
    for i in range(dim):
        
        phase = 2*np.pi*(x * u[i] + y * v[i])
        F[i, :]    = np.reshape(np.cos(phase), (n_pix*n_pix,))
        F[i+dim, :] = np.reshape(np.sin(phase), (n_pix*n_pix,))
        
    return F * pix_size**2


def ComputeCountsGaussian(xc, yc, FWHM_max, FWHM_min, flux, angle, u, v,
                          sf, sr, pf, pr, phase_corr, pixel_area, pixel_phase_factor):
    """
    Function for computing the counts registered by the STIX detector pixels when the emitting source is a Gaussian
    function. It is based on Equation (35) of "The STIX imaging concept" (Massa et al., Solar Physics, 2023)
    
    INPUT
    xc: float
        x coordinate of the center of the Gaussian function

    yc: float
        y coordinate of the center of the Gaussian function

    FWHM_min: float
        min Full Width at Half Maximum (FWHM) of the Gaussian function

    FWHM_max: float
        max Full Width at Half Maximum (FWHM) of the Gaussian function

    flux: float
        intensity (i.e., total flux) of the gaussian function. It is the value of the integral of the Gaussian

    angle: float
        orientation angle (measured counterclockwise from the positive x-axis) of the Gaussian function
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX

    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX
    
    sf: numpy array
        array containing the values of the slit width of the front windows
    
    sr: numpy array
        array containing the values of the slit width of the rear windows
    
    pf: numpy array
        array containing the pitch values of the front windows
    
    pr: numpy array
        array containing the pitch values of the rear windows
    
    phase_corr: numpy array
        array containing the phase correction factors (in deg) to be added to the visibility phases.
        They are the sum of the grid correction factors (retrieved from the windows phases) and the "ad hoc"
         correction factors, due to instrument distorsions
    
    pixel_area: numpy array
        value of the area of the considered pixels (e.g., top row, bottom row, small, or combination of them) in mm^2
    
    pixel_phase_factor: numpy array
        phase correction factor due to the integration over the pixel area (in deg.)
    
    OUTPUT
    A: array containing the values of the counts recorded by the "A" pixels of each detector
    
    B: array containing the values of the counts recorded by the "B" pixels of each detector
    
    C: array containing the values of the counts recorded by the "C" pixels of each detector
    
    D: array containing the values of the counts recorded by the "D" pixels of each detector
    """
    
    # Define parameters
    P = pixel_area*(sf*sr)/(pf*pr)
    E = 4.*np.sqrt(2.)/np.pi**3*(pf*pr)/(sf*sr)*np.sin(np.pi*sf/pf)*np.sin(np.pi*sr/pr)
    
    # Compute amplitude and phase visibility
    vis = ComputeVisibilitiesGaussian(xc, yc, FWHM_max, FWHM_min, flux, angle, u, v)
    
    re_vis = vis[0:len(vis)//2]
    im_vis = vis[len(vis)//2:len(vis)]
    
    amp_vis   = np.sqrt(re_vis**2+im_vis**2)
    phase_vis = np.arctan2(im_vis,re_vis)
    
    # Compute counts
    phase = phase_vis - (phase_corr + pixel_phase_factor)/180.*np.pi
    A = P*(flux - E*amp_vis*np.cos(phase))
    B = P*(flux - E*amp_vis*np.sin(phase))
    C = P*(flux + E*amp_vis*np.cos(phase))
    D = P*(flux + E*amp_vis*np.sin(phase))
    
    
    return A,B,C,D


def ComputeVisFromCounts(xc, yc, FWHM_max, FWHM_min, flux, angle, u, v,
                         sf, sr, pf, pr, phase_corr, pixel_area, pixel_phase_factor,
                         add_noise=False):
    """
    Function for computing the visibilities measured by STIX when the emitting source is a Gaussian
    function. It is based on Equation (18) of "The STIX imaging concept" (Massa et al., Solar Physics, 2023)
    
    INPUTS
    xc: float
        x coordinate of the center of the Gaussian function

    yc: float
        y coordinate of the center of the Gaussian function

    FWHM_min: float
        min Full Width at Half Maximum (FWHM) of the Gaussian function

    FWHM_max: float
        max Full Width at Half Maximum (FWHM) of the Gaussian function

    flux: float
        intensity (i.e., total flux) of the gaussian function. It is the value of the integral of the Gaussian

    angle: float
        orientation angle (measured counterclockwise from the positive x-axis) of the Gaussian function

    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX

    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX

    sf: numpy array
        array containing the values of the slit width of the front windows

    sr: numpy array
        array containing the values of the slit width of the rear windows

    pf: numpy array
        array containing the pitch values of the front windows

    pr: numpy array
        array containing the pitch values of the rear windows

    phase_corr: numpy array
        array containing the phase correction factors (in deg) to be added to the visibility phases.
        They are the sum of the grid correction factors (retrieved from the windows phases) and the "ad hoc"
        correction factors, due to instrument distorsions

    pixel_area: numpy array
        value of the area of the considered pixels (e.g., top row, bottom row, small, or combination of them) in mm^2

    pixel_phase_factor: numpy array
        phase correction factor due to the integration over the pixel area (in deg.)
    
    OUTPUT
    vis: array containing the real and the imaginary parts of the visibilities corresponding to the simulated Gaussian source
    
    sigamp: array containing the uncertainty on the visibility amplitudes corresponding to the simulated Gaussian sources
    """
    
    A,B,C,D = ComputeCountsGaussian(xc, yc, FWHM_max, FWHM_min, flux, angle, u, v,
                                    sf, sr, pf, pr, phase_corr, pixel_area, pixel_phase_factor)
    
    if add_noise:
        
        cond_add_noise=True
        
        while cond_add_noise:
        
            A = np.random.poisson(A).astype('float64')
            B = np.random.poisson(B).astype('float64')
            C = np.random.poisson(C).astype('float64')
            D = np.random.poisson(D).astype('float64')
            
            cond_add_noise = (((C-A) == 0) & ((D-B) == 0)).any()

    
    dA = np.sqrt(A)
    dB = np.sqrt(B)
    dC = np.sqrt(C)
    dD = np.sqrt(D)

    # Visibility amplitudes
    vis_cmina = C-A
    vis_dminb = D-B

    dcmina = np.sqrt(dC**2 + dA**2)
    ddminb = np.sqrt(dD**2 + dB**2)
    
    P = pixel_area*(sf*sr)/(pf*pr)
    E = 4.*np.sqrt(2.)/np.pi**3*(pf*pr)/(sf*sr)*np.sin(np.pi*sf/pf)*np.sin(np.pi*sr/pr)

    vis_amp = np.sqrt(vis_cmina**2 + vis_dminb**2)
    sigamp  = np.sqrt( (np.divide(vis_cmina, vis_amp, out=np.zeros_like(vis_cmina), where=vis_amp!=0)*dcmina)**2
                      +(np.divide(vis_dminb, vis_amp, out=np.zeros_like(vis_dminb), where=vis_amp!=0)*ddminb)**2 )
    
    vis_amp = vis_amp/(2.*P*E)
    sigamp  = sigamp/(2.*P*E)

    # Visibility phases
    vis_phase = np.arctan2(vis_dminb,vis_cmina)+ (pixel_phase_factor + phase_corr)/180.*np.pi
    
    # Compute visibility values
    vis = np.zeros(2*len(u))
    
    vis[0:len(u)]        = vis_amp*np.cos(vis_phase)
    vis[len(u):2*len(u)] = vis_amp*np.sin(vis_phase)

    return vis, sigamp


def generate_powerlaw(x_min, x_max, alpha, n_samples=1):
    """
    Function for generating samples according to a powerlaw distribution defined between a minimum and a maximum value

    INPUTS:
    x_min: float
        minimum value attained by the simulated distribution

    x_max: float
        maximum value attained by the simulated distribution

    alpha: float
        powerlaw index

    n_samples: int
        number of samples to be generated. Default, 1

    OUTPUT
    array containing samples distributed according to a powerlaw distribution

    """

    # Generate uniform distribution between 0 and 1
    eps = 1e-3
    y = np.random.uniform(low=eps, high=1.0-eps, size=n_samples)

    beta = (1-alpha) / (x_max**(-alpha+1) - x_min**(-alpha+1)) 

    F_y = np.power(((1-alpha) / beta * y + x_min**(-alpha+1)), 1/(-alpha+1))

    return F_y

def SimulateConfig(n_sources, u, v, sf, sr, pf, pr, phase_corr, pixel_area, pixel_phase_factor,
                   add_noise = False, fov = 257., FWHM_max_min = 10, FWHM_max_max_1 = 100, 
                   FWHM_max_max_2 = 50, ecc_min = 0.3, dist_min = 25, dist_max = 120, 
                   flux_min = 1000, flux_max = 1200000, alpha_flux = 2, dynamic_range = 0.3):
    """
    Function for simulating the visibility values corresponding to a random configuration
    consisting of several Gaussian sources
    
    INPUTS:
    
    n_sources: int,
        number of Gaussian sources in the simulated configuration
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX

    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX

    sf: numpy array
        array containing the values of the slit width of the front windows

    sr: numpy array
        array containing the values of the slit width of the rear windows

    pf: numpy array
        array containing the pitch values of the front windows

    pr: numpy array
        array containing the pitch values of the rear windows

    phase_corr: numpy array
        array containing the phase correction factors (in deg) to be added to the visibility phases.
        They are the sum of the grid correction factors (retrieved from the windows phases) and the "ad hoc"
        correction factors, due to instrument distorsions

    pixel_area: numpy array
        value of the area of the considered pixels (e.g., top row, bottom row, small, or combination of them) in mm^2

    pixel_phase_factor: numpy array
        phase correction factor due to the integration over the pixel area (in deg.)
    
    KEYWORDS:
    
    add_noise: boolean
        If True, random noise is added to the visibility values
    
    fov: float
        Size [arcsec] of the simulated Field-of-View (FOV). Default, 257 [arcsec]
    
    FWHM_max_min: float
        Minimum value for the maximum FWHM of each source [arcsec]. Default, 10 [arcsec]
    
    FWHM_max_max_1: float
        Maximum value for the maximum FWHM [arcsec] of the source in the case the configuration consists of a single source.
        Default, 100 [arcsec]
    
    FWHM_max_max_2: float
        Maximum value for the maximum FWHM [arcsec] of the source in the case the configuration consists of two or more sources
        Default, 100 [arcsec]
    
    ecc_min: float
        Minimum value for the eccentricity (i.e., min FWHM / max FWHM) of each source. Default, 0.3
    
    dist_min: float
        Minimum distance between different Gaussian sources [arcsec]. Default, 25.
    
    dist_max: float
        Maximum distance between the sources [arcsec]. Default, 120.
    
    flux_min: float
        Minimum value for the flux of the first simulated source [counts s^-1 keV^-1 cm^-2 arcsec^-2]. Default, 1000
                     
    flux_max: float
        Maximum value for the flux of the first simulated source [counts s^-1 keV^-1 cm^-2 arcsec^-2]. Default, 1200000

    alpha_flux: float
        Powerlaw index used for simulating the configuration flux. Default, 2

    dynamic_range: float
        Minimum value of the ratio between the flux of the first simulated source and the flux of the other sources
        (if n_sources > 1). Default, 0.3
    
    OUTPUTS:
    
    vis: float array of dimension 2*len(u) containing the real and the imaginary part of the visibility values
    
    xc: float array of dimension n_sources containing the x coordinates of the center of the simulated Gaussian sources
    
    yc: float array of dimension n_sources containing the y coordinates of the center of the simulated Gaussian sources
    
    FWHM_max: float array of dimension n_sources containing the values of the max FWHM of the simulated Gaussian sources
    
    FWHM_min: float array of dimension n_sources containing the values of the min FWHM of the simulated Gaussian sources
    
    flux: float array of dimension n_sources containing the values of the flux of the simulated Gaussian sources
    
    orientation: float array of dimension n_sources containing the orientation angle of the simulated Gaussian sources
    
    sigamp: array containing the estimated std of the stochastic error affecting the visibility amplitudes.
    """
    
    if n_sources == 1:
    
        xc = []
        yc = []
        FWHM_max = []
        FWHM_min = []
        flux = []
        orientation = []

        FWHM_max.append(np.random.uniform(low=FWHM_max_min, high=FWHM_max_max_1))
        FWHM_min.append(np.random.uniform(low=max([FWHM_max_min,FWHM_max[0]*ecc_min]), high=FWHM_max[0]))
        
        xc.append(np.random.uniform(low=-fov/2.+3./2.*FWHM_max[0], high=fov/2.-3./2.*FWHM_max[0]))
        yc.append(np.random.uniform(low=-fov/2.+3./2.*FWHM_max[0], high=fov/2.-3./2.*FWHM_max[0]))

        flux.append(generate_powerlaw(flux_min, flux_max, alpha_flux))
        
        orientation.append(np.random.uniform(low=0, high=180))

        vis, sigamp = ComputeVisFromCounts(xc[0], yc[0], FWHM_max[0], FWHM_min[0], flux[0], orientation[0],
                                           u, v, sf, sr, pf, pr, phase_corr, pixel_area, pixel_phase_factor,
                                           add_noise=add_noise)
    
    else:
        
        xc = []
        yc = []
        FWHM_max = []
        FWHM_min = []
        flux = []
        orientation = []
        
        xc.append(np.random.uniform(low=-fov/2.+3./2.*FWHM_max_max_2, high=fov/2.-3./2.*FWHM_max_max_2))
        yc.append(np.random.uniform(low=-fov/2.+3./2.*FWHM_max_max_2, high=fov/2.-3./2.*FWHM_max_max_2))
        
        FWHM_max.append(np.random.uniform(low=FWHM_max_min, high=FWHM_max_max_2))
        FWHM_min.append(np.random.uniform(low=max([FWHM_max_min,FWHM_max[0]*ecc_min]), high=FWHM_max[0]))

        flux.append(generate_powerlaw(flux_min, flux_max, alpha_flux))
        orientation.append(np.random.uniform(low=0, high=180))

        vis, sigamp = ComputeVisFromCounts(xc[0], yc[0], FWHM_max[0], FWHM_min[0], flux[0], orientation[0], u, v,
                                           sf, sr, pf, pr, phase_corr, pixel_area, pixel_phase_factor,
                                           add_noise=add_noise)
    
        for i in range(1,n_sources):

            cond = True

            while cond:

                x_new = np.random.uniform(low=-fov/2.+3./2.*FWHM_max_max_2, high=fov/2.-3./2.*FWHM_max_max_2)
                y_new = np.random.uniform(low=-fov/2.+3./2.*FWHM_max_max_2, high=fov/2.-3./2.*FWHM_max_max_2)

                cond = []
                dist = []
                for j in range(len(xc)):

                    this_dist = np.sqrt((xc[j] - x_new)**2 + (yc[j] - y_new)**2)

                    if (this_dist >= dist_min+FWHM_max_min) and (this_dist <= dist_max):
                        cond.append(True)
                    else:
                        cond.append(False)
                        
                    dist.append(this_dist)

                cond = not all(cond)

            dist=np.array(dist)
            
            xc.append(x_new)
            yc.append(y_new)
            
            FWHM_max_new = np.random.uniform(low=FWHM_max_min, high=max([FWHM_max_min, 
                                             min([FWHM_max_max_2,np.min(dist - np.array(FWHM_max))])]))                       
            FWHM_max.append(FWHM_max_new)
            
            FWHM_min.append(np.random.uniform(low=max([FWHM_max_min,FWHM_max_new*ecc_min]), high=FWHM_max_new))
            
            flux.append(flux[0]*np.random.uniform(low=dynamic_range, high=1.))
            orientation.append(np.random.uniform(low=0, high=180))

            vis_new, sigamp_new = ComputeVisFromCounts(xc[i], yc[i], FWHM_max[i], FWHM_min[i], flux[i], 
                                                       orientation[i], u, v, sf, sr, pf, pr, phase_corr, 
                                                       pixel_area, pixel_phase_factor, add_noise=add_noise)

            vis += vis_new
            sigamp = np.sqrt(sigamp**2 + sigamp_new**2)
    
    xc = np.array(xc)
    yc = np.array(yc)
    FWHM_max = np.array(FWHM_max)
    FWHM_min = np.array(FWHM_min)
    flux = np.array(flux)
    orientation = np.array(orientation)
    
    return xc, yc, FWHM_max, FWHM_min, flux, orientation, vis, sigamp
