import os
import math
import numpy as np
import yaml
import pickle

import sacc
import healpy as hp
import pymaster as nmt
from pixell import enmap, reproject
import kmeans_radec

from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots
plt.ion()

def IndexToDeclRa(index, nside=2048):
    theta,phi=hp.pixelfunc.pix2ang(nside,index)
    return -np.degrees(theta-np.pi/2.),np.degrees(np.pi*2.-phi)

def DeclRaToIndex(decl,RA, nside=2048):
    return hp.pixelfunc.ang2pix(nside,np.radians(-decl+90.),np.radians(360.-RA))

data_dir = './'

nside_dgrade = 2048

# kappa_mask = hp.read_map(os.path.join(data_dir, 'data/act/masks/act_GAL060_mask_healpy_nside=2048.fits'))
kappa_mask = hp.read_map(os.path.join(data_dir, 'data/act/hp_nside2048_lmax6000_act_dr4.01_s14s15_D56_lensing_mask.fits'))
kappa_mask = np.where(kappa_mask > 1, 1, kappa_mask)
kappa_mask = np.where(kappa_mask < 1e-2, 0, kappa_mask)

wt_map = hp.read_map(os.path.join(data_dir, f'data/des/DESY3_blind_cat_e1e2_maps/weight_map_bin1_nside2048.fits'))
gamma_mask = np.where(wt_map == hp.UNSEEN, 0, wt_map)
gamma_mask = np.where(gamma_mask > 1, 1, gamma_mask)
gamma_mask = np.where(gamma_mask < 1e-2, 0, gamma_mask)

# kappa_mask = hp.read_map('./data/combined_mask.fits')

comb_mask = hp.ud_grade(kappa_mask*gamma_mask, nside_dgrade)

ra, dec = IndexToDeclRa(np.arange(len(comb_mask)), nside=nside_dgrade)

maxra = ra[comb_mask > 0].max()
minra = ra[comb_mask > 0].min()

maxdec = dec[comb_mask > 0].max()
mindec = dec[comb_mask > 0].min()

nregions = 28

ra_initial, dec_initial = kmeans_radec.test.generate_randoms_radec(minra, maxra, mindec, maxdec, nregions)

X = np.vstack((ra[comb_mask>0], dec[comb_mask>0])).T
Xunmask = np.vstack((ra, dec)).T

km = kmeans_radec.kmeans_sample(X, nregions)

# plt.figure(1)
# kmeans_radec.test.plot_centers(km.centers, plt.gca())
# plt.show()

region_label = np.nan * np.ones_like(comb_mask)

region_label[comb_mask>0] = kmeans_radec.find_nearest(X, km.centers)

# plt.figure(2)
hp.mollview(region_label, title='Jackknife Regions', cmap='gist_ncar')
plt.show()
