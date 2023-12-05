import os
import healpy as hp
import numpy as np
import pymaster as nmt
import pickle
import math
import time
from pixell import enmap, reproject

from matplotlib import pyplot as plt

plt.close('all')

# start_script = time.time()

data_dir = os.environ['ACTDIR']

nside = 2048
ell_min = 20
ell_max = 4096
delta_ell = 300

nbins_tomo = 4

kappa_dir = os.path.join(data_dir, 'data/act/')
gamma_dir = os.path.join(data_dir, 'data/des/DESY3_blind_cat_e1e2_maps/')

kappa_60_map_names = [
                        'coadd_4splitlensingmapMV.fits',
                        'coadd_4splitlensingmapMVPOL.fits',
                        'coadd_4splitlensingmapnomaskTT.fits',
                        'coadd_4splitlensingmapTT.fits',
                        'coaddc_4splitlensingmapMV.fits',
                        'kappa_MV_alms.fits',
                        'kappa_MVPOL_alms.fits',
                        # 'kappa_TT_alms.fit',
                    ]

# kappa_map_name = 'coadd_4splitlensingmapMV.fits'
kappa_mask_name = 'masks/act_mask_20220316_GAL060_rms_70.00_d2sk.fits'

g1_map_name = 'blind_DESY3_meansub_Rcorrected_g1_map_binX_nside2048.fits'
g2_map_name = 'blind_DESY3_meansub_Rcorrected_g2_map_binX_nside2048.fits'

count_map_name = 'count_map_binX_nside2048.fits'
weight_map_name = 'weight_map_binX_nside2048.fits'

# list of null type: filename
# baseline (+curl): v_all_MV_subtractDR5_recal
# 40% mask:v_all_MV_40sub
# TT: v_all_TTsub
# MVPOL: v_all_MVPOLsub
# MVPOL 40% mask ): v_all_MVPOL40
# f090 MV: v_90_MVsub
# f090 TT: v_90_TTsub
# f150 MV:v_150_MVsub
# f150 TT: v_150_TTsub
# f150-f090 MV: f_15090_MVsub
# f150-f090 TT   f_15090_TTsub

# t_maps_read = time.time()

## set up cl binning
nbins = math.ceil((ell_max - ell_min) / delta_ell) + 1

ell_bin_edges = []
ell_bin_edges.append(0)
for i in np.arange(0, nbins - 1, 1):
    if i == 0:
        ell_bin_edges.append(ell_min)
    else:
        ell_bin_edges.append(ell_bin_edges[i] + delta_ell)
ell_bin_edges.append(ell_max)
lh_edges = np.asarray(ell_bin_edges)

ells = np.arange(0, ell_max  +1, 1)
bpws = -1*np.ones_like(ells, dtype = 'int')
weights = np.zeros_like(ells, dtype = 'float64')

for i in np.arange(0, nbins, 1):
    bpws[lh_edges[i]:lh_edges[i + 1]] = i
    weights[lh_edges[i]:lh_edges[i + 1]] = 1.

print('Getting binning...')
# t_bins_start = time.time()
ell_bins = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights, lmax=ell_max)
# t_bins_end = time.time()
print('Done.')

## set up gamma field (used for all)
gamma_field = {}

for ibin in np.arange(1, nbins_tomo + 1):

    print(f'DES bin {ibin} field...')

    gamma_mask = hp.read_map(os.path.join(gamma_dir, weight_map_name.replace('binX', f'bin{ibin}')))

    g1_field_name = g1_map_name.replace('binX', f'bin{ibin}').rstrip('.fits')
    g2_field_name = g2_map_name.replace('binX', f'bin{ibin}').rstrip('.fits')

    gamma_field_fname = g1_field_name.replace('g1', 'gamma')
    gamma_field_fname = os.path.join(gamma_dir, gamma_field_fname + '.pkl')

    # load des shear maps
    g1_map = hp.read_map(os.path.join(gamma_dir, g1_map_name.replace('binX', f'bin{ibin}')))
    g2_map = hp.read_map(os.path.join(gamma_dir, g2_map_name.replace('binX', f'bin{ibin}')))

    gamma_field[ibin] = nmt.NmtField(gamma_mask, [g1_map, g2_map], masked_on_input=True)


# load act mask
kappa_mask = enmap.read_map(os.path.join(kappa_dir, kappa_mask_name))
kappa_mask = reproject.map2healpix(kappa_mask,
                                   nside=nside, lmax=None, out=None, rot=None,
                                   spin=[0,2], method="spline", order=0,
                                   extensive=False, bsize=100000,
                                   nside_mode="pow2", boundary="constant", verbose=False)

# t_fields_end = time.time()
# print('Done in {}s.'.format(t_fields_start - t_fields_end))

for kappa_map_name in kappa_60_map_names:

    ## set up kappa field 
    print('Getting fields...')
    # t_fields_start = time.time()

    kappa_field_name = kappa_map_name.rstrip('.fits') + '_' + kappa_mask_name.rstrip('.fits')
    kappa_field_fname = os.path.join(kappa_dir, kappa_field_name + '.pkl')

    # load act null test map
    kappa_alms = hp.read_alm(os.path.join(kappa_dir, kappa_map_name)).astype('complex128')
    kappa_map = hp.alm2map(np.nan_to_num(kappa_alms), nside=nside)


    kappa_field = nmt.NmtField(kappa_mask, [kappa_map], masked_on_input=True)

    ## set up workspace
    workspace_kappagamma = {}
    ## compute coupling matix
    print('Getting coupling matrix...')
    # t_coupling_start = time.time()
    for ibin in np.arange(1, nbins_tomo + 1):

        workspace_fname = f'workspace_kappagamma_bin{ibin}.fits'

        if os.path.exists(workspace_fname):
            workspace_kappagamma[ibin] = nmt.NmtWorkspace()
            workspace_kappagamma[ibin].read_from(workspace_fname)
        else:
            workspace_kappagamma[ibin] = nmt.NmtWorkspace()
            workspace_kappagamma[ibin].compute_coupling_matrix(kappa_field, gamma_field[ibin], ell_bins)
            workspace_kappagamma[ibin].write_to(workspace_fname)
    # t_coupling_end = time.time()
    # print('Done in {}s.'.format(t_coupling_start - t_coupling_end))

    cl_coupled = {}
    cl_decoupled = {}

    ## compute cls
    for ibin in np.arange(1, nbins_tomo + 1):

        print('Getting coupled Cls...')
        # t_coupledcl_start = time.time()
        cl_coupled[ibin] = nmt.compute_coupled_cell(gamma_field[ibin], kappa_field)
        # t_coupledcl_end = time.time()
        # print('Done in {}s.'.format(t_coupledcl_start - t_coupledcl_end))

        print('Getting decoupled Cls...')
        # t_decoupledcl_start = time.time()
        cl_decoupled[ibin] = workspace_kappagamma[ibin].decouple_cell(cl_coupled[ibin])[0]
        # t_decoupledcl_end = time.time()
        # print('Done in {}s.'.format(t_decoupledcl_start - t_decoupledcl_end))

    plt.close('all')
    plt.figure(1, figsize=(4.5, 3.75))
    for ibin in np.arange(1, nbins_tomo + 1):
        plt.plot(ell_bins.get_effective_ells(), cl_decoupled[ibin], '.-')

    ax = plt.gca()
    plt.xlim([0,ells.max()])
    plt.ylim([-5.e-12, 5.e-12])

    ax.fill_between(ells, ax.get_ylim()[0], ax.get_ylim()[1], where=(ells > 1900), color='silver', alpha=0.5)
    ax.fill_between(ells, ax.get_ylim()[0], ax.get_ylim()[1], where=(ells < 100), color='silver', alpha=0.5)

    np.savetxt(f'./output/{kappa_map_name}_ellcl.dat', np.column_stack([ell_bins.get_effective_ells(), cl_decoupled]))

    plt.axhline(0.0, color='k')#, linestyle='dashed', alpha=0.4)
    plt.xlabel('$\ell$')
    plt.ylabel('$\Delta C^{XY}_{\ell}$')
    plt.savefig(f'figures/{kappa_map_name}_null.png', dpi=300, bbox_inches='tight')
