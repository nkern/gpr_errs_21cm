"""Perform tiling and gridding of 21 cm simulation cubes to HEALPix maps.

Author: Piyanat Kittiwisit (piyanat.kittiwisit@asu.edu)
Created: August 19, 2013

"""
from __future__ import print_function, division

import argparse
import os

import numpy as np
import healpy as hp
from astropy.cosmology import WMAP9


def cube2hpx(simfile, freq, nside=4096, sim_res=7.8125,
             sim_size=(128, 128, 128), healpix_coord_files=None):
    """
    Parameters
    ----------
    simfile: string
        Name of a temperature simulation cube.
    freq: float
        Frequency of interest in MHz.
    nside: integer
        NSIDE of the output HEALPix image. Must be a valid NSIDE for HEALPix.

    """
    # Read in and interpolate simulation cubes to the redshift of interest.
    if isinstance(simfile, str):
        cube = np.load(simfile)
    else:
        cube = simfile

    # Determine the radial comoving distance r to the comoving shell at the
    # frequency of interest.
    f21 = 1420.40575177  # MHz
    z21 = f21 / freq - 1
    dc = WMAP9.comoving_distance(z21).value

    # Get the vector coordinates (vx, vy, vz) of the HEALPIX pixels.
    if not healpix_coord_files:
        healpix_coord_files = 'healpix_coord_N{:d}.npy'.format(nside)
    if os.path.isfile(healpix_coord_files):
        vx, vy, vz = np.load(healpix_coord_files)
    else:
        vx, vy, vz = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))

    # Translate vector coordinates to comoving coordinates and determine the
    # corresponding cube indexes (xi, yi, zi). For faster operation, we will
    # use the mod function to determine the nearest neighboring pixels and
    # just grab the data points from those pixels instead of doing linear
    # interpolation.
    xi = np.mod(np.around(vx * dc / sim_res).astype(int), sim_size[0])
    yi = np.mod(np.around(vy * dc / sim_res).astype(int), sim_size[1])
    zi = np.mod(np.around(vz * dc / sim_res).astype(int), sim_size[2])
    out = np.asarray(cube[xi, yi, zi])

    return out
    #hp.write_map(hpxfile, out, fits_IDL=False, dtype=np.float64, coord='C')
    # TODO: Add unit convertion option that multiply healpix map by some factors
    # TODO: Add history
    # TODO: Add BUNIT


if __name__ == '__main__':
    convert_string = lambda string: [int(s) for s in string.split()]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('simfile', type=str,
                        help='Name of an input temperature simulation cube file.')
    parser.add_argument('fitsfile', type=str,
                        help='Name of an output healpix file.')
    parser.add_argument('freq', type=float,
                        help='Frequency of interest.')
    parser.add_argument('--nside', type=int, default=4096,
                        help='nside of the output healpix image.')
    parser.add_argument('--sim_size', type=convert_string, nargs=3,
                        default='128 128 128',
                        metavar=('xsize', 'ysize', 'zsize'),
                        help='Number of (x, y, z) pixels of the input simulation cube')
    parser.add_argument('--sim_res', type=float, default=7.8125,
                        help='Pixel size of the simulation cube in Mpc/h')
    parser.add_argument('--read_column', '--col', type=str,
                        help='Column in simfile to read')
    args = parser.parse_args()
    cube2hpx(args.simfile, args.fitsfile, args.freq, nside=args.nside,
             sim_res=args.sim_res, sim_size=args.sim_size)
