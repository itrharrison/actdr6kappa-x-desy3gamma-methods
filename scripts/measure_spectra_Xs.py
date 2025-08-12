import os
import math
import numpy as np
import yaml
import pickle

import sacc
import healpy as hp
import pymaster as nmt
from pixell import enmap, reproject

from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

# data_dir = '/scratch/c.c1025819/actdr6_nulls/'
data_dir = './'

s = sacc.Sacc()
s_data = s.load_fits('../actdr4kappa-x-desy3gamma-data/data/UNBLINDED_ACTPlanck_tSZfree_ACTDR4-kappa_DESY3-gamma_data_simCov.fits')
ell_data, cl_data = s_data.get_ell_cl(tracer1='gs_des_bin4', tracer2='ck_act', data_type='cl_20')
bpw = s_data.get_bandpower_windows([18, 19, 20, 21, 22, 23])

nside = 2048
# binning = nmt.NmtBin.from_nside_linear(nside, 4)
# binning = nmt.NmtBin.from_lmax_linear(1900, 300)
# binning = nmt.NmtBin(nside=nside, bpws=bpw.values, ells=bpw.values, weights=bpw.weight, lmax=1900)

elmax = 2200
mf_lmin = 100
delta_l = 300

nbins = math.ceil((elmax-mf_lmin)/delta_l) + 1
#nbins = math.ceil(elmax/delta_l)

edgs = []
edgs.append(0)
for i in np.arange(0, nbins-1, 1):
    if i == 0:
        edgs.append(mf_lmin)
    else:
        edgs.append(edgs[i]+delta_l)
edgs.append(elmax)
ledges = np.asarray(edgs)
print('ledges', ledges)

ells = np.arange(0, elmax+1, 1)
bpws = -1*np.ones_like(ells, dtype = 'int')
weights = np.zeros_like(ells, dtype = 'float64')
weights_ell = np.zeros_like(ells, dtype = 'float64')

for i in np.arange(0, nbins, 1):
    bpws[ledges[i]:ledges[i+1]] = i
    # weights[ledges[i]:ledges[i+1]] = 1.
    weights[ledges[i]:ledges[i+1]] = np.arange(ledges[i], ledges[i+1], 1)**2

# binning = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights, lmax=elmax)

binning = nmt.NmtBin(bpws=bpws, ells=ells, weights=weights, lmax=elmax)
# binning = nmt.NmtBin.from_lmax_linear(elmax, delta_l)#, f_ell = ells*ells)#, f_ell=np.arange(1, 6143 + 2))

kappa_alms = hp.read_alm('data/act/kappa_alm_data_act_dr6_lensing_v1_baseline.fits').astype('complex128')
kappa_map = hp.alm2map(np.nan_to_num(kappa_alms), nside=nside)

stellar_map_4096 = hp.read_map(os.path.join(data_dir, 'data/psf_stellar_density_fracdet_binned_1024_nside_4096_cel.fits.gz'))
stellar_map = hp.ud_grade(stellar_map_4096, nside)
stellar_map[stellar_map<0] = 0.0

ext_map_4096 = hp.read_map(os.path.join(data_dir, 'data/ebv_sfd98_fullres_nside_4096_nest_equatorial.fits.gz'))
ext_map = hp.ud_grade(ext_map_4096, nside)

mask_pwr = 2

kappa_mask = hp.read_map(os.path.join(data_dir, 'data/act/masks/act_GAL060_mask_healpy_nside=2048.fits'))
# kappa_mask_dr4 = hp.read_map(os.path.join(data_dir, 'data/act/hp_nside2048_lmax6000_act_dr4.01_s14s15_D56_lensing_mask.fits'))
kappa_mask = np.where(kappa_mask > 1, 1, kappa_mask)
kappa_mask = np.where(kappa_mask < 1e-2, 0, kappa_mask)

# kappa_map_weight = np.mean(kappa_mask**2.)
# kappa_map_weight_dr4 =  0.5709974
# kappa_map_weight = 0.12760189732901422
# kappa_map_weight = 0.06225266366704071

kappa_mask = kappa_mask**mask_pwr

for ibin in np.arange(1,5):
    g1_map = hp.read_map(os.path.join(data_dir, f'data/des/DESY3_cat_e1e2_maps/DESY3_meansub_Rcorrected_g1_map_bin{ibin}_nside2048.fits'))
    g2_map = hp.read_map(os.path.join(data_dir, f'data/des/DESY3_cat_e1e2_maps/DESY3_meansub_Rcorrected_g2_map_bin{ibin}_nside2048.fits'))
    mod_g_map = np.sqrt(g1_map**2. + g2_map**2.)

    gamma_mask = hp.read_map(os.path.join(data_dir, f'data/des/DESY3_blind_cat_e1e2_maps/weight_map_bin{ibin}_nside2048.fits'))
    # gamma_mask = np.where(gamma_mask > 1, 1, gamma_mask)
    # gamma_mask = np.where(gamma_mask < 1e-2, 0, gamma_mask)

    aposcale = 0.5
    gamma_mask = nmt.mask_apodization(gamma_mask, aposcale, apotype="C1")
    kappa_mask = nmt.mask_apodization(kappa_mask, aposcale, apotype="C1")

    comb_mask = kappa_mask * gamma_mask

    print('Done.')

    wsp_s1s2 = nmt.NmtWorkspace()
    wsp_s1s1 = nmt.NmtWorkspace()
    # wsp_moi = nmt.NmtWorkspace()

    print('Fields...')
    # kappa_field_moi = nmt.NmtField(kappa_mask, [kappa_map], spin=0, masked_on_input=True)#, beam=None, masked_on_input=True)
    # gamma_field_moi = nmt.NmtField(gamma_mask, [g1_map, g2_map], spin=2, masked_on_input=True)#, purify_b=False, beam=None)
    kappa_field = nmt.NmtField(kappa_mask, [kappa_map], beam=None, masked_on_input=True, lmax=elmax)#, beam=None, masked_on_input=False)
    gamma_field = nmt.NmtField(gamma_mask, [-1.0 * g1_map, g2_map], purify_b=False, beam=None, lmax=elmax)#, purify_b=False, beam=None)
    stellar_field = nmt.NmtField(gamma_mask, [stellar_map], beam=None, masked_on_input=False, lmax=elmax)#, purify_b=False, beam=None)
    ext_field = nmt.NmtField(gamma_mask, [ext_map], beam=None, masked_on_input=False, lmax=elmax)#, purify_b=False, beam=None)
    print('Done.')

    print('Coupling...')
    wsp_s1s2.compute_coupling_matrix(kappa_field, gamma_field, binning)
    wsp_s1s1.compute_coupling_matrix(kappa_field, stellar_field, binning)
    # wsp_moi.compute_coupling_matrix(kappa_field_moi, gamma_field_moi, binning)

    Bbl = wsp_s1s1.get_bandpower_windows().squeeze()
    plt.figure()
    for ibp in np.arange(7):
        plt.plot(ells, Bbl[ibp])

    print('Done.')

    print('Measuring...')
    # cl_kappagamma = compute_master(kappa_field, gamma_field, wsp)
    # cl_kappagamma_ana = hp.anafast([kappa_map * comb_mask, g1_map * comb_mask, g2_map * comb_mask])
    # cl_kappagamma_ana = binning.bin_cell(cl_kappagamma_ana[3,:2201])
    # cl_kappagamma = nmt.workspaces.compute_full_master(kappa_field, gamma_field, binning)
    # cl_kappagamma_moi = compute_master(kappa_field_moi, gamma_field_moi, wsp_moi)

    cl_kappastellar = compute_master(kappa_field, stellar_field, wsp_s1s1)
    cl_kappaext = compute_master(kappa_field, ext_field, wsp_s1s1)

    cl_gammastellar = compute_master(stellar_field, gamma_field, wsp_s1s2)
    cl_gammaext = compute_master(ext_field, gamma_field, wsp_s1s2)

    cl_stellarstellar = compute_master(stellar_field, stellar_field, wsp_s1s1)
    cl_extext = compute_master(ext_field, ext_field, wsp_s1s1)

    print('Done.')

    outobj = {
            'ell' : binning.get_effective_ells(),
            'cl_kappastellar' : cl_kappastellar,
            'cl_kappaext' : cl_kappaext,
            'cl_gammastellar': cl_gammastellar,
            'cl_gammaext' : cl_gammaext,
            'cl_stellarstellar' : cl_stellarstellar,
            'cl_extext' : cl_extext
            }

    outfile = f'./data/Xs_spectra_bin{ibin}.pkl'

    pickle.dump(outobj, open(outfile, 'wb'))
