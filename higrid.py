#!/usr/bin/env python3
#program to grid hydrogen observations into a data cube.
#this program uses multi-tasking to read the spectra.
#HISTORY
#26Mar19 GIL show max min values in resulting cube
#26Mar16 GIL move re-pix to center of the image
#26Mar14 GIL reshape output matrix, make calibration consistent.
#26Mar13 GIL add diagnostic print
#26Mar11 GIL sort out gridding problems
#26Mar10 GIL try to fix coordinate gridding
#26Mar09 GIL third initial version.

import numpy as np
import argparse
from multiprocessing import Pool
from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.wcs.utils import pixel_to_skycoord
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import interp1d
from spectral_cube import SpectralCube
from tqdm import tqdm
# special gridding tool stuff
from processing import processSpectrum, Config

# modules from analyze
import radioastronomy
import tsys
import hotcold

# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid spectral line observations into a spectral cube."
    )

    parser.add_argument("--freq-min", type=float, default=1419.75,
                        help="Minimum topocentric frequency (MHz)")
    parser.add_argument("--freq-max", type=float, default=1422.75,
                        help="Maximum topocentric frequency (MHz)")
    parser.add_argument("--vel-min", type=float, default = -200.,
                        help="Minimum velocity (km/sec)")
    parser.add_argument("--vel-max", type=float, default = +200.,
                        help="Maximum velocity (km/sec)")
    parser.add_argument("--nchan", type=int, default=50,
                        help="Number of spectral channels")

    parser.add_argument("--hot", type=str,
                        help="Hot Calibration File")
    parser.add_argument("--cold", type=str,
                        help="Cold Calibration File")
    
    parser.add_argument("--ra-min", type=float, default=0.0)
    parser.add_argument("--ra-max", type=float, default=360.)
    parser.add_argument("--dec-min", type=float, default=-30.)
    parser.add_argument("--dec-max", type=float, default=140.)
    parser.add_argument("--pix", type=float, default=3.,
                        help="Spatial pixel size (deg)")

    parser.add_argument("--beam-fwhm", type=float, default=8.0,
                        help="Gaussian beam FWHM (deg)")

    parser.add_argument("--thot", type=float, default=295.0,
                        help="Estimated hot load temperature (K)")
    parser.add_argument("--tcold", type=float, default=5.0,
                        help="Estimated cold load temperature (K)")

    parser.add_argument("--baseline", action="store_true", 
                        help="Option to fit a linear baseline subtraction")

    parser.add_argument("--pixel", action="store_true", 
                        help="Option only grid one pixel, not beam")

    parser.add_argument("--rest-freq", type=float, default=1420.40575177,
                        help="Rest frequency for velocity conversion (MHz)")

    parser.add_argument("--velocity", action="store_true",
                        help="Convert to LSRK velocity cube instead of frequency cube")

    parser.add_argument("--output-cube", type=str,
                        default="spectral_cube.fits",
                        help="Output FITS cube filename")
    parser.add_argument("--output-m0", type=str, default="moment0.fits",
                        help="Output moment-0 FITS filename")

    parser.add_argument(
        "--projection",
        type=str,
        default="AIT",
        choices=["CAR", "TAN", "MOL", "AIT"],
        help="Sky projection for gridding: CAR, TAN, MOL, AIT"
    )
    
    parser.add_argument("files", nargs="+",
                        help="List of FITS files containing spectra")

    return parser.parse_args()
    # end of parser()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    """
    grid a list of files each containing spectrum.
    """
    args = parse_args()

    freq_axis_topo = np.linspace(args.freq_min, args.freq_max,
                                 args.nchan) * u.MHz
    n_freq = len( freq_axis_topo)
    
    rest_freq = args.rest_freq * u.MHz

    processCount = 0
    
    print("frequncies, min, max, ref: %.6f, %.6f, %.6f" %
          (freq_axis_topo[0].value,
           freq_axis_topo[n_freq-1].value,
           rest_freq.value))
    
    dec_grid = np.arange(args.dec_min, args.dec_max, args.pix)
    # for dec assume min < max always
    ddec = args.dec_max - args.dec_min
    avedec = (args.dec_max + args.dec_min)/2.
    avera = (args.ra_min + args.ra_max)/2.
    dra   = args.ra_max - args.ra_min

    # if the map does not cross zero RA
    if args.ra_min < args.ra_max:
        ra_grid = np.arange(args.ra_min, args.ra_max, args.pix)
    else:    
        # ra is wrapping 360!
        # example if ramin = 350 and ramax = 30
        # then dra = 30 - 350 = -320, but want avera in range 0 to 360
        # so dra = dra + 360 = 40 and avera = (350 + 30)/2 = 380/2 = 160
        # but want avera = 10,  avera = (350 + (30 + 360))/2 = 740/2 = 370
        # avera = avera - 360 = 10.
        dra = dra + 360.
        avera = (args.ra_min + args.ra_max + 360.)/2
        if avera > 360:
            avera = avera - 360.
        ra_grid = np.arange(0., dra, args.pix) + args.ra_min

    n_ra = len(ra_grid)
    n_dec = len(dec_grid)
    # now check for coordinate wrap
    for i in range(n_ra):
        if ra_grid[i] > 360.:
            ra_grid[i] = ra_grid[i] - 360.

    print("Ra : %8.2f %8.2f D: %8.2f %d" %
          (args.ra_min, args.ra_max, dra, n_ra)) 
    print("Dec: %8.2f %8.2f D: %8.2f %d" %
          (args.dec_min, args.dec_max, ddec, n_dec)) 

    if processCount == 0:
        print("Image size: %4d,%4d,%4d" % (n_ra, n_dec, args.nchan))
    cube_data = np.zeros((args.nchan, n_dec, n_ra), dtype=float)
    cube_moment = np.zeros((n_dec, n_ra), dtype=float)
    weight_data = np.zeros_like(cube_moment)

    beam_sigma = args.beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    c = 299792.458 * u.km/u.s
    if args.velocity:
        vel_axis = (rest_freq - freq_axis_topo) / rest_freq * c
        vel_axis_kms = vel_axis.to(u.km/u.s).value
    else:
        vel_axis_kms = None

    # now need the observations for calibration
    hot  = radioastronomy.Spectrum()
    cold = radioastronomy.Spectrum()

    # try to read calibration files
    doCalib = True  # assume if calibration files are present, then calibrate
    try:
        if args.hot != None:
            hot.read_spec_ast( args.hot)
            hv = hot.ydataA/hot.nave
        else:
            doCalib = False
        if args.cold != None:
            cold.read_spec_ast( args.cold)
            cv = cold.ydataA/cold.nave
        else:
            doCalib = False
    except:
        print("\nCan not read calibration Files\n")
        print("Hot,Cold Files: %s,%s" % (args.hot,args.cold))
        doCalib = False

    # if no calibration files provided, get spectral length from a file
    if doCalib == False:
        afile = args.files[0]
        try:
            cold.read_spec_ast(afile)
            cv = cold.ydataA/cold.nave
            nin= len(cv)
            print("Reading first file: nChan = %d" % nchan)
            # if no hot+cold, then gains are unity
            gains = np.zeros_like(cv) + 1.
            # in this case the first spectra will be the reference
            # the spectrum will be subtracted from all observations
        except:
            print("Can not read the first spectra file!")
            print("File Name: %s" % afile)
            print("Exiting! \n\n")
            exit()
    # if here was able to read a file
    nin = len(cv)
    nc = int(nin/2)
    
    # now compute ranges for baseline calc. ignore 10% on each end
    xa0 = int(nin/10)
    xa = 2*xa0
    xb = nin - xa
    xbe = nin - xa0

    if args.baseline:
        print("Fitting a Linear baseline")
    else:
        print("Not removing a baseline")

    # if computing baseline range from velocities
    if args.velocity and args.vel_min != None and args.vel_max != None:
        print("Selecting indicies from velocities: %.3f,%.3f" %
              (args.vel_min, args.vel_max))
        xv = cold.xdata
        nc = int(len( cold.xdata)/2)   # center channel
              
        print("freqs: %d: %.3f" % (xa0, xv[xa0]))
        print("freqs: %d: %.3f" % (xa, xv[xa]))
        print("freqs: %d: %.3f" % (nc, xv[nc]))
        print("freqs: %d: %.3f" % (xb, xv[xb]))
        print("freqs: %d: %.3f" % (xbe, xv[xbe]))

        # from the gr-radio_astro get calibration codes for indicies
        vel, xa0, xa, xb, xbe = hotcold.velocity_indecies( cold, 25,
                                                           args.vel_max,
                                                           args.vel_min)
        
        print("Indicies for baseline: %d - %d and %d - %d" %
              (xa0, xa, xb, xbe))
        print("velocities: %d: %.3f" % (xa0, vel[xa0]))
        print("velocities: %d: %.3f (%.3f)" % (xa, vel[xa], args.vel_min))
        print("velocities: %d: %.3f" % (nc, vel[nc]))
        print("velocities: %d: %.3f (%.3f)" % (xb, vel[xb], args.vel_max))
        print("velocities: %d: %.3f" % (xbe, vel[xbe]))

    # if found and read calibration files, compute gains
    if doCalib:
        print("Computing Kelvins/couunt with Hot=%.2fK anc Cold=%.2f" % (
            args.thot, args.tcold))
        gains, gainAve, tRxMiddle, tRms, tStdA, tStdB = \
            hotcold.compute_gain( hv, cv, xa0, xa, xb, xbe, args.thot, args.tcold)
        print("Calibration tRx: %.3f +/- %.3f" % (tRxMiddle, tRms))
        # will mutiply by 1 over gain
        gains = 1./gains
        
    print("gains: %d: %.3f" % (xa0, gains[xa0]))
    print("gains: %d: %.3f" % (xa, gains[xa]))
    print("gains: %d: %.3f" % (nc, gains[nc]))
    print("gains: %d: %.3f" % (xb, gains[xb]))
    print("gains: %d: %.3f" % (xbe, gains[xbe]))

    
    # prepare to write a data cube.
    w = WCS(naxis=3)

    # Choose projection
    proj = args.projection.upper()
    if proj == "CAR":
        ctype1, ctype2 = "RA---CAR", "DEC--CAR"
    elif proj == "TAN":
        ctype1, ctype2 = "RA---TAN", "DEC--TAN"
    elif proj == "MOL":
        ctype1, ctype2 = "RA---MOL", "DEC--MOL"
    elif proj == "AIT":
        ctype1, ctype2 = "RA---AIT", "DEC--AIT"
    else:
        print("Projection %s not recognized" % proj)
        
    # Spectral axis
    if args.velocity:
        ctype3 = "VELO-LSR"
        cunit3 = "km/s"
        crval3 = vel_axis_kms[0]
        cdelt3 = vel_axis_kms[1] - vel_axis_kms[0]
    else:
        ctype3 = "FREQ"
        cunit3 = "MHz"
        crval3 = freq_axis_topo[0].value
        cdelt3 = freq_axis_topo[1].value - freq_axis_topo[0].value

    w.wcs.ctype = [ctype1, ctype2, ctype3]
    w.wcs.cunit = ["deg", "deg", cunit3]

    # now prepare to write FITS header
    w.wcs.crval = [avera, avedec, crval3 ]
    w.wcs.cdelt = [-args.pix, args.pix, cdelt3]
    w.wcs.crpix = [int(n_ra/2.), int(n_dec/2.), 1]
#    w.wcs.latpole = 90.
    
    naxis = np.zeros(3)
    naxis[0] = n_ra
    naxis[1] = n_dec
    naxis[2] = args.nchan
    print("Data Cube Shape: ", naxis)

    hdu = fits.PrimaryHDU(cube_data, header=w.to_header())
    hdu.writeto( args.output_cube, overwrite=True)

    # Extract 2D celestial WCS for RA/Dec → pixel
    cel_wcs = w.celestial

    print(cel_wcs)

    # Prepare config for workers
    cfg = Config(
        velocity=args.velocity,
        baseline=args.baseline,
        pixel=args.pixel,
        rest_freq_mhz=args.rest_freq,
        nchan=args.nchan,         # number of OUTPUT channels
        xa0=xa0,                  # these are the baseline fitting
        xa=xa,                    # arguments BEFORE interpolation
        xb=xb,                    # two ranges, one on either side of line
        xbe=xbe,
        processCount=processCount,
        beam_sigma=beam_sigma,
        freq_axis_topo=freq_axis_topo,
        vel_axis_kms=vel_axis_kms,
        cel_wcs=cel_wcs,
        naxis=naxis,
        crpix=w.wcs.crpix,
        cdelt=w.wcs.cdelt,
        crval=w.wcs.crval,
        # add precalcualted calibration steps
        gains=gains,
    )
    # print info every few spectra
    processCount = 0

    jobs = [(fname, cfg) for fname in args.files]

    # here the multi-tasking work is done
    with Pool() as pool:
        for cube_part, weight_part in tqdm(
            pool.imap_unordered(processSpectrum, jobs),
            total=len(jobs),
            desc="Gridding spectra",
        ):
            cube_data += cube_part
            weight_data += weight_part
            
            processCount = processCount + 1
            
# when here, all data have been processed.
# Normalize
    mask = weight_data > 0
    for i in range(args.nchan):
        cube_data[i, mask] /= weight_data[mask]

# google on FITS vs python data order        
# If your FITS file has dimensions (NAXIS1, NAXIS2, NAXIS3) of shape (X, Y, Z),
# the resulting Python NumPy array will have a shape of (Z, Y, X)
# (or (axis 0, axis 1, axis 2)).
# that's why the output array is transposed!!!!


    hdu = fits.PrimaryHDU(cube_data, header=w.to_header())
    hdu.writeto( args.output_cube, overwrite=True)

# show maximum and minum values in resulting cube
    cshape = cube_data.shape
    print(cshape)
    amax = np.argmax(cube_data)
    cmax, dmax, rmax = np.unravel_index( amax, cshape)
    print( "Grid Max: %10.3f K at %d,%d,%d" %
           (cube_data[cmax,dmax,rmax], rmax, dmax, cmax))
    # Convert pixel to sky
    coords = pixel_to_skycoord(rmax, dmax, w)
    print("Max RA, Dec: %.2f, %.2f" % (coords.ra.deg, coords.dec.deg))
    amin = np.argmin(cube_data)
    cmin, dmin, rmin = np.unravel_index( amin, cshape)
    print( "Grid Min: %10.3f K at %d,%d,%d" %
           (cube_data[cmin,dmin,rmin], rmin, dmin, cmin))
    coords = pixel_to_skycoord(rmin, dmin, w)
    print(f"Min RA, Dec: %.2f, %.2f" % (coords.ra.deg, coords.dec.deg))

    # now diagnose header prameters
    header = w.to_header()
#    print(header)

    cube = cube_data.transpose(2, 1, 0)

    cube_sc = SpectralCube( data=cube * u.K,  wcs=w )

    moment0 = cube_sc.moment(order=0)
#
    w0 = WCS(naxis=2)
#    print("Moment Cube Shape: ",moment0.value)
    hdu2 = fits.PrimaryHDU(moment0.value, header=moment0.wcs.to_header())
    hdu2.writeto( args.output_m0, overwrite=True )

    return
    # end of main()
    

if __name__ == "__main__":
    main()
