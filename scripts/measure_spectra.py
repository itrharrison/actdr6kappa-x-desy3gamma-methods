import os
import math
import numpy as np
import yaml

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

for i in np.arange(0, nbins, 1):
    bpws[ledges[i]:ledges[i+1]] = i
    weights[ledges[i]:ledges[i+1]] = 1.

binning = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights, lmax=elmax)

cl_theory = np.loadtxt('./figures/cl_unbinned_actplanck.txt')
cl_theory = cl_theory.reshape([4, 1901])
cl_theory_bin = cl_theory[-1, :]
cl_theory_ells = np.arange(len(cl_theory_bin))
cls_to_bin = np.interp(np.arange(binning.lmax + 1), cl_theory_ells, cl_theory_bin)
cl_fid = binning.bin_cell(np.array([cls_to_bin]))[0]

print('Reading...')
# kappa_alms = hp.read_alm('data/act/kappa_alm_data_act_dr6_lensing_v1_baseline.fits').astype('complex128')
# kappa_map = hp.alm2map(np.nan_to_num(kappa_alms), nside=nside)
kappa_map = hp.read_map('data/act/hp_nside2048_lmax6000_act_planck_dr4.01_s14s15_D56_lensing_kappa.fits')

g1_map = hp.read_map('data/des/DESY3_cat_e1e2_maps/DESY3_meansub_Rcorrected_g1_map_bin4_nside2048.fits')
g2_map = hp.read_map('data/des/DESY3_cat_e1e2_maps/DESY3_meansub_Rcorrected_g2_map_bin4_nside2048.fits')

mask_pwr = 2

kappa_mask = hp.read_map('data/act/masks/act_GAL060_mask_healpy_nside=2048.fits')
# kappa_mask = hp.read_map('data/act/hp_nside2048_lmax6000_act_dr4.01_s14s15_D56_lensing_mask.fits')
kappa_mask = np.where(kappa_mask > 1, 1, kappa_mask)
kappa_mask = np.where(kappa_mask < 1e-2, 0, kappa_mask)

kappa_mask = kappa_mask**mask_pwr

gamma_mask = hp.read_map('data/des/DESY3_blind_cat_e1e2_maps/weight_map_bin4_nside2048.fits')
# gamma_mask = np.where(gamma_mask > 1, 1, gamma_mask)
# gamma_mask = np.where(gamma_mask < 1e-2, 0, gamma_mask)

print('Done.')

wsp = nmt.NmtWorkspace()
wsp_moi = nmt.NmtWorkspace()

print('Fields...')
# kappa_field_moi = nmt.NmtField(kappa_mask, [kappa_map], spin=0, masked_on_input=True)#, beam=None, masked_on_input=True)
# gamma_field_moi = nmt.NmtField(gamma_mask, [g1_map, g2_map], spin=2, masked_on_input=True)#, purify_b=False, beam=None)
kappa_field = nmt.NmtField(kappa_mask, [kappa_map], beam=None, masked_on_input=False)#, beam=None, masked_on_input=False)
gamma_field = nmt.NmtField(gamma_mask, [g1_map, g2_map], purify_b=False, beam=None)#, purify_b=False, beam=None)
print('Done.')

print('Coupling...')
wsp.compute_coupling_matrix(kappa_field, gamma_field, binning)
# wsp_moi.compute_coupling_matrix(kappa_field_moi, gamma_field_moi, binning)

print('Done.')

print('Measuring...')
cl_kappagamma = compute_master(kappa_field, gamma_field, wsp)
# cl_kappagamma_moi = compute_master(kappa_field_moi, gamma_field_moi, wsp_moi)
print('Done.')

plt.close('all')
plt.figure(1, figsize=(4.5, 3.75))
plt.axhline(0, color='k', linestyle='dashed')
plt.plot(binning.get_effective_ells(), 1.e9 * cl_fid, '.-', label='Theory Cl')
plt.plot(binning.get_effective_ells(), 1.e9 * cl_kappagamma[0], 'o', mfc='none', label='Ian Cl')
# plt.plot(binning.get_effective_ells(), 1.e9 * cl_kappagamma_moi[0], 'o', label='Ian Cl Mask on input')
plt.plot(ell_data, 1.e9 * cl_data, '+', label='Shabbir Cl')
plt.xlim([0,2000])
plt.ylim([-1, 6])
plt.legend()

plt.savefig('./figures/test_cl.png', dpi=300, bbox_inches='tight')
