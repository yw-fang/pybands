#!/usr/bin/env python

import sys
import os
import re
import numpy as np
from optparse import OptionParser

############################################################
__version__ = "1.0"
############################################################

def find_vasp_files(vasp_file):
    try:
        file = open(vasp_file, 'r')
    except IOError:
        print('%s may not exist!' % vasp_file)
        sys.exit()

def WeightFromPro(infile='PROCAR', whichAtom=None, spd=None, lsorbit=False):
    """
    Contribution of selected atoms to the each KS orbital
    """


    # assert os.path.isfile(infile), '%s cannot be found!' % infile
    find_vasp_files(vasp_file=infile)
    FileContents = [line for line in open(infile) if line.strip()]

    # when the band number is too large, there will be no space between ";" and
    # the actual band number. A bug found by Homlee Guo.
    # Here, #kpts, #bands and #ions are all integers
    nkpts, nbands, nions = [int(xx) for xx in re.sub('[^0-9]', ' ', FileContents[1]).split()]

    if spd:
        Weights = np.asarray([line.split()[1:-1] for line in FileContents
                              if not re.search('[a-zA-Z]', line)], dtype=float)
        Weights = np.sum(Weights[:,spd], axis=1)
    else:
        Weights = np.asarray([line.split()[-1] for line in FileContents
                              if not re.search('[a-zA-Z]', line)], dtype=float)

    nspin = Weights.shape[0] / (nkpts * nbands * nions)
    nspin /= 4 if lsorbit else 1

    if lsorbit:
        Weights.resize(nspin, nkpts, nbands, 4, nions)
        Weights = Weights[:,:,:,0,:]
    else:
        Weights.resize(nspin, nkpts, nbands, nions)

    if whichAtom is None:
        return np.sum(Weights, axis=-1)
    else:
        whichAtom = [xx - 1 for xx in whichAtom]
        return np.sum(Weights[:,:,:,whichAtom], axis=-1)



############################################################
def get_bandInfo(inFile = 'OUTCAR'):
    """
    extract band energies from OUTCAR
    """
    find_vasp_files(vasp_file=inFile)
    outcar = [line for line in open(inFile) if line.strip()]
    print(type(outcar))
    print(outcar[0:10])

    for ii, line in enumerate(outcar):
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])

        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])

        if "k-points in reciprocal lattice and weights" in line:
            Lvkpts = ii + 1

        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 1
            # break

    # basis vector of reciprocal lattice
    # B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]],

    # When the supercell is too large, spaces are missing between real space
    # lattice constants. A bug found out by Wei Xie (weixie4@gmail.com).
    B = np.array([line.split()[-3:] for line in outcar[ibasis:ibasis+3]],
                 dtype=float)
    # k-points vectors and weights
    tmp = np.array([line.split() for line in outcar[Lvkpts:Lvkpts+nkpts]],
                   dtype=float)
    vkpts = tmp[:,:3]
    wkpts = tmp[:,-1]

    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2
    bands = []
    # vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'k-point' in line:
            # vkpts += [line.split()[3:]]
            continue
        bands.append(float(line.split()[1]))

    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))

    if os.path.isfile('KPOINTS'):
        kp = open('KPOINTS').readlines()

    if os.path.isfile('KPOINTS') and kp[2][0].upper() == 'L':
        Nk_in_seg = int(kp[1].split()[0])
        Nseg = nkpts // Nk_in_seg
        vkpt_diff = np.zeros_like(vkpts, dtype=float)

        for ii in range(Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            vkpt_diff[start:end, :] = vkpts[start:end,:] - vkpts[start,:]

        kpt_path = np.linalg.norm(np.dot(vkpt_diff, B), axis=1)
        # kpt_path = np.sqrt(np.sum(np.dot(vkpt_diff, B)**2, axis=1))
        for ii in range(1, Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            kpt_path[start:end] += kpt_path[start-1]

        # kpt_path /= kpt_path[-1]
        kpt_bounds =  np.concatenate((kpt_path[0::Nk_in_seg], [kpt_path[-1],]))
    else:
        # get band path
        vkpt_diff = np.diff(vkpts, axis=0)
        kpt_path = np.zeros(nkpts, dtype=float)
        kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
        # kpt_path /= kpt_path[-1]

        # get boundaries of band path
        xx = np.diff(kpt_path)
        kpt_bounds = np.concatenate(([0.0,], kpt_path[1:][np.isclose(xx, 0.0)], [kpt_path[-1],]))

    return kpt_path, bands, Efermi, kpt_bounds


############################################################
def command_line_arg():
    usage = "usage: %prog [options] arg1 arg2"
    par = OptionParser(usage=usage, version= __version__)

    par.add_option('-f', '--file',
            action='store', type="string",
            dest='filename', default='OUTCAR',
            help='location of OUTCAR')

    par.add_option('--procar',
            action='store', type="string", dest='procar',
            default='PROCAR',
            help='location of the PROCAR')

    par.add_option('-z', '--zero',
            action='store', type="float",
            dest='efermi', default=None,
            help='energy reference of the band plot')

    par.add_option('-o', '--output',
            action='store', type="string", dest='bandimage',
            default='bands-narrow.pdf',
            help='output image name, "bands.pdf" by default')

    par.add_option('-k', '--kpoints',
            action='store', type="string", dest='kpts',
            default=None,
            help='kpoint path')

    par.add_option('-s', '--size', nargs=2,
            action='store', type="float", dest='figsize',
            default=(6.0, 8.0),
            help='figure size of the output plot')

    par.add_option('-y', nargs=2,
            action='store', type="float", dest='ylim',
           default=(-0.5, 0.5),
            help='energy range of the band plot')

    par.add_option('--lw',
            action='store', type="float", dest='linewidth',
            default=2.0,
            help='linewidth of the band plot')

    par.add_option('--dpi',
            action='store', type="int", dest='dpi',
            default=360,
            help='resolution of the output image')

    par.add_option('--occ',
            action='append', type="string", dest='occ',
            default=[],
            help='orbital contribution of each KS state')

    par.add_option('--occL',
            action='store_true', dest='occLC',
            default=False,
            help='use Linecollection or Scatter to show the orbital contribution')

    par.add_option('--occLC_cmap',
            action='store', type='string', dest='occLC_cmap',
            default='jet',
            help='colormap of the line collection')

    par.add_option('--occLC_lw',
            action='store', type='float', dest='occLC_lw',
            default=2.0,
            help='linewidth of the line collection')

    par.add_option('--occLC_cbar_pos',
            action='store', type='string', dest='occLC_cbar_pos',
            default='top',
            help='position of the colorbar')

    par.add_option('--occLC_cbar_size',
            action='store', type='string', dest='occLC_cbar_size',
            default='3%',
            help='size of the colorbar, relative to the axis')

    par.add_option('--occLC_cbar_pad',
            action='store', type='float', dest='occLC_cbar_pad',
            default=0.02,
            help='pad between colorbar and axis')

    par.add_option('--occM',
            action='append', type="string", dest='occMarker',
            default=[],
            help='the marker used in the plot')

    par.add_option('--occMs',
            action='append', type="int", dest='occMarkerSize',
            default=[],
            help='the size of the marker')

    par.add_option('--occMc',
            action='append', type="string", dest='occMarkerColor',
            default=[],
            help='the color of the marker')

    par.add_option('--spd',
            action='store', type="string", dest='spdProjections',
            default=None,
            help='Spd-projected wavefunction character of each KS orbital.')

    par.add_option('--lsorbit',
            action='store_true', dest='lsorbit',
            help='Spin orbit coupling on, special treament of PROCAR')

    par.add_option('-q', '--quiet',
            action='store_true', dest='quiet',
            help='not show the resulting image')


    return  par.parse_args( )



def main():

    opts, args = command_line_arg()

    if opts.occ:

        Nocc = len(opts.occ)
        occM  = ['o' for ii in range(Nocc)]
        occMc = ['r' for ii in range(Nocc)]
        occMs = [20  for ii in range(Nocc)]
        for ii in range(min(len(opts.occMarker), Nocc)):
            occM[ii] = opts.occMarker[ii]
        for ii in range(min(len(opts.occMarkerSize), Nocc)):
            occMs[ii] = opts.occMarkerSize[ii]
        for ii in range(min(len(opts.occMarkerColor), Nocc)):
            occMc[ii] = opts.occMarkerColor[ii]
        opts.occMarker = occM
        opts.occMarkerColor = occMc
        opts.occMarkerSize = occMs

        whts = []
        for occ in opts.occ:
            alist = np.array(occ.split(), dtype=int)
            nlist = [x for x in alist if not x == -1]
            cmark, = np.where(alist == -1)
            for ii in cmark:
                nlist += range(alist[ii + 1], alist[ii + 2] + 1)
            occAtom = set(nlist)

            # occAtom = [int(x) for x in opts.occ.split()]
            # whts = WeightFromPro(opts.procar, whichAtom = occAtom)
            if opts.spdProjections and (Nocc == 1):
                angularM = [int(x) for x in opts.spdProjections.split()]
                # print angularM
                whts.append(WeightFromPro(opts.procar, whichAtom=occAtom,
                    spd=angularM, lsorbit=opts.lsorbit))
            else:
                whts.append(WeightFromPro(opts.procar, whichAtom=occAtom,
                    lsorbit=opts.lsorbit))
    else:
        whts = None

    kpath, bands, efermi, kpt_bounds = get_bandInfo(opts.filename)
    if opts.efermi is None:
        bands -= efermi
    else:
        bands -= opts.efermi
############################################################
if __name__ == '__main__':
    main()




