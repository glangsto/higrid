#!/usr/bin/env python3
#program to grid hydrogen observations into a data cube.
#this program uses multi-tasking to read the spectra.
#HISTORY
#26Mar18 GIL remove extra prints
#26Mar16 GIL move ra-pix to center of the image
#26Mar14 GIL reshape output matrix, make calibration consistent.
#26Mar11 GIL option to put only one pixel
#26Mar10 GIL fix coordinate gridding
#26Mar09 GIL third initial version.

import numpy as np
from dataclasses import dataclass
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import interp1d

# modules from analyze
import radioastronomy
import gainfactor as gf
import tsys
import hotcold

@dataclass
class Config:
    velocity: bool
    baseline: bool
    pixel: bool
    rest_freq_mhz: float
    nchan: int
    xa0: int           # these indices are before interpolation
    xa: int            # so fit can occure outside data gridded
    xb: int
    xbe: int
    processCount: int
    beam_sigma: float  #
    freq_axis_topo: object
    vel_axis_kms: object
    cel_wcs: object
#    wcs: object
    naxis: np.ndarray
    crpix: np.ndarray
    cdelt: np.ndarray
    crval: np.ndarray
# pre-computed gains
    gains: np.ndarray

# ---------------------------------------------------------
# INPUT READER
# ---------------------------------------------------------
def load_spectrum_file(fname):
    rs = radioastronomy.Spectrum()
    rs.read_spec_ast( fname)
    inten = rs.ydataA/rs.nave
    freq = rs.xdata * 1.E-6    # covert to MHz

    az = rs.telaz
    el = rs.telel
    ra = rs.ra
    dec = rs.dec
        
    utc = str( rs.utc)
    parts = utc.split(' ')
    nparts = len( parts)
    if nparts > 1:
        utc = parts[0] + 'T' + parts[1]

    lat = rs.tellat
    lon = rs.tellon
    try:
        height = rs.telalt
    except:
        height = 800.

    telescope = EarthLocation(lat=lat * u.deg, lon=lon * u.deg,
                              height=height * u.m)
    obstime = Time(utc, format="isot", scale="utc")
    return ra, dec, obstime, telescope, freq, inten
    # end of load_spectrum_file()
    
# convert frequency to velocity
def topo_freq_to_lsr_velocity(freq_topo_mhz, ra_deg, dec_deg, obstime,
                              telescope_location, rest_freq):
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    v_corr = sky.radial_velocity_correction(
        kind="lsrk", obstime=obstime, location=telescope_location
    ).to(u.km/u.s)

    nu_topo = freq_topo_mhz * u.MHz
    c = 299792.458 * u.km/u.s

    nu_lsr = nu_topo * (1.0 - v_corr / c)
    vel_lsr = (rest_freq - nu_lsr) / rest_freq * c

    return vel_lsr.to(u.km/u.s).value
    # end of topo_freq_to_lsr_velocity()
    
def processSpectrum(args):
    fname, cfg = args
    
    ra, dec, obstime, telescope, freq, inten = load_spectrum_file(fname)

    # now calibrate:
    inten = inten * cfg.gains

    rest_freq = cfg.rest_freq_mhz * u.MHz

    nData = len(inten)
    fitOrder = 1
    
    if cfg.processCount == 1:
        print("cfg.naxis: ", cfg.naxis.astype(int))
        print("cfg.cdelt: ", cfg.cdelt)
        print("cfg.crpix: ", cfg.crpix)
        print("cfg.crval: ", cfg.crval)
        print("%4d: %12.3f - %12.3f -> %.3f - %.3f" %
              (cfg.processCount, freq[cfg.xa0], freq[cfg.xa],
               freq[cfg.xb], freq[cfg.xbe]))
        
    if cfg.baseline:
        baseline = gf.fit_range( freq[0:nData],
                                 inten[0:nData], 
                                 cfg.xa0, cfg.xa, cfg.xb, cfg.xbe,
                                 fitOrder)
        # now subtract of baseline
        inten = inten - baseline
    
    # Spectral interpolation (velocity or frequency)
    if cfg.velocity:
        # take frequency axis values and make an identical size velocity values
        vel_lsr_kms = topo_freq_to_lsr_velocity(
            freq, ra, dec, obstime, telescope, rest_freq
        )
        # prepare to put data on a grid.
        # create interpolation function interp()
        interp = interp1d(vel_lsr_kms, inten, bounds_error=False,
                          fill_value=0.0, kind='cubic')
        # now take the function and interpolate to output array
        spec_interp = interp(cfg.vel_axis_kms)
    else:
        interp = interp1d(freq, inten, bounds_error=False,
                          fill_value=0.0, kind='cubic')
        spec_interp = interp(cfg.freq_axis_topo)

    # --- KEY CHANGE: use WCS to get projected pixel position ---
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    x_pix, y_pix = cfg.cel_wcs.world_to_pixel(sky)  # float pixel coords
#    print(cfg.cel_wcs)
    naxis = cfg.naxis.astype(int)
    n2    = cfg.naxis/2.
    n2    = n2.astype(int)

    # Build Gaussian kernel in pixel space
    ix = np.arange(naxis[0])
    iy = np.arange(naxis[1])  
    # create a arrays of x,y distances from center pixel
    cdx = (x_pix - cfg.crpix[0])*cfg.cdelt[0]
    cdy = (y_pix - cfg.crpix[1])*cfg.cdelt[1]
    dx = ((ix - cfg.crpix[0])*cfg.cdelt[0]) - cdx
    dy = ((iy - cfg.crpix[1])*cfg.cdelt[1]) - cdy
#    print("ra,dec %.2f,%2f ->  %.2f,%.2f -> dx,dy=%.2f,%.2f" %
 #         (ra, dec, x_pix, y_pix, cdx, cdy))
    # calculate max offset^2 for kernel from center pixel 5 arcmin 
    maxd2 = 2.*6.*6.
    # 3 d cube but 2 d weight
    cube_part   = np.zeros((naxis[2], naxis[1], naxis[0]), dtype=float)
    weight_part = np.zeros((naxis[1], naxis[0]), dtype=float)

    # if coordinate is outside the grid, just return zeros
    x_pix = int(x_pix + .5)
    if x_pix < 0 or x_pix >= naxis[0]:
        return cube_part, weight_part
    y_pix = int(y_pix + .5)
    if y_pix < 0 or y_pix >= naxis[1]:
        return cube_part, weight_part

    # if only gridding a single pixel
    if cfg.pixel:
        for i in range(cfg.nchan):
            cube_part[i, y_pix, x_pix] = spec_interp[i]
            weight_part[y_pix, x_pix]  = 1.
    else:
        # else create a whole image with a gaussian
        gauss_x = np.exp(-0.5 * (dx / cfg.beam_sigma) ** 2)
        gauss_y = np.exp(-0.5 * (dy / cfg.beam_sigma) ** 2)
        outer = np.outer(gauss_y, gauss_x)  # (Dec, RA)
        kernel = np.zeros_like(outer)  # (Dec, RA)
        # the kernel
        # limit kernal to circle around center pixel
        for j in range(naxis[1]):
            d2y = dy[j]*dy[j]
            for k in range( naxis[0]):
                d2 = d2y + dx[k]*dx[k]
                if d2 < maxd2:
#        kernel = np.outer(gauss_y, gauss_x)  # (Dec, RA)
                    kernel[j,k] = outer[j,k]
                # else keep kernel at zero
        #sometimes get NANS, do not use in sum
        if np.isnan(kernel).any():
            # just return zeros with zero weight
            return cube_part, weight_part
        # normally normalize
        # now multiply the whole spectrum by kernel and kernel

        for i in range(cfg.nchan):
            cube_part[i, :, :]   = kernel * spec_interp[i]

        # 2-d weight
        weight_part[:, :] = kernel

#    if cfg.processCount % 100 == 0:
#        print("x: %.2f %.2f %.2f %.1f" % (ix[n2[0]], x_pix, dx, cfg.crpix[0]))
#        print("y: %.2f %.2f %.2f %.1f" % (iy[n2[1]], y_pix, dy, cfg.crpix[1]))

    cfg.processCount = cfg.processCount + 1
    
    return cube_part, weight_part
