import os
import math
import numpy as np
import yaml

import healpy as hp
import pymaster as nmt
from pixell import enmap, reproject


class NullSpectrum():

    def __init__(self, config_fname):

        with open(config_fname, 'r') as file:
            ini = yaml.safe_load(file)

        self.nside = ini['config']['nside']
        self.ell_min = ini['config']['ell_min']
        self.ell_max = ini['config']['ell_max']
        self.delta_ell = ini['config']['delta_ell']
        self.nsims = ini['config']['nsims']

        self.dirs = ini['dirs']
        self.tracer_config = ini['tracer']
        self.nulltests = ini['nulltests']

        # check if environment variable has been specified for root dir
        if self.dirs['root'].startswith('${'):
            self.dirs['root'] = os.environ.get(self.dirs['root'].lstrip('${').rstrip('}'))
        
        self.setup_binning()

    def setup_binning(self):
        '''Set up the uniform ell binning of the null test spectrum.

        Creates a namaster NmtBin object and stores it as self.ell_bins
        Weights are uniform and bandpowers are square.
        '''

        print('Getting binning...')
        nbins = math.ceil((self.ell_max - self.ell_min) / self.delta_ell) + 1

        ell_bin_edges = []
        ell_bin_edges.append(0)
        for i in np.arange(0, nbins - 1, 1):
            if i == 0:
                ell_bin_edges.append(self.ell_min)
            else:
                ell_bin_edges.append(ell_bin_edges[i] + self.delta_ell)
        ell_bin_edges.append(self.ell_max)
        lh_edges = np.asarray(ell_bin_edges)

        ells = np.arange(0, self.ell_max  +1, 1)
        bpws = -1*np.ones_like(ells, dtype = 'int')
        weights = np.zeros_like(ells, dtype = 'float64')

        for i in np.arange(0, nbins, 1):
            bpws[lh_edges[i]:lh_edges[i + 1]] = i
            weights[lh_edges[i]:lh_edges[i + 1]] = 1.

        self.ell_bins = nmt.NmtBin(nside=self.nside, bpws=bpws, ells=ells, weights=weights, lmax=self.ell_max)
        print('Done.')

    def setup_tracer_fields(self, masked_on_input=True):
        '''Set up the namaster NmtField corresponding to the non-ACT tracer maps being 
        used in the cross-correlation.

        Creates a namaster NmtField object for each bin_tag specified in the config file and stores 
        it as self.tracer_fields[bin_tag]
        '''

        print('Getting tracer fields...')

        self.tracer_spin = self.tracer_config['tracer_spin']
        self.tracer_nbins = self.tracer_config['tracer_nbins']
        self.tracer_fields = {}

        tracer_dir = os.path.join(self.dirs['root'], self.dirs['tracer_dir'])

        for ibin, bin_tag in enumerate(self.tracer_config['tracer_bin_tags']):

            print(f'{bin_tag}...')

            tracer_mask_fname = self.tracer_config['tracer_masks'][bin_tag]

            tracer_mask = hp.read_map(os.path.join(tracer_dir, tracer_mask_fname))

            tracer_maps = []

            for ispin, spin_tag in enumerate(self.tracer_config['tracer_spin_tags']):
                tracer_map_fname = self.tracer_config['tracer_maps'][bin_tag][spin_tag]
                tracer_maps.append(hp.read_map(os.path.join(tracer_dir, tracer_map_fname)))
                
            self.tracer_fields[bin_tag] = nmt.NmtField(tracer_mask, tracer_maps,
                                                       masked_on_input=masked_on_input)

        print('Done.')

    def setup_act_field(self, nulltest, masked_on_input=True, remove_mask_apod=False):
        '''Set up the namaster NmtField corresponding to the ACT kappa map being used in 
        the cross-correlation.

        Applies a transfer function read from a file if specified in nulltest['transfer_function']

        Parameters
        ----------
        nulltest
            dict containing the specification of the null test

        Returns
        -------
        kappa_field
            The namaster NmtField object
        '''

        print('Getting ACT field...')

        kappa_dir = self.dirs['kappa_dir']
        kappa_map_fname = nulltest['kappa_map']
        kappa_mask_fname = nulltest['kappa_mask']

        kappa_alms = hp.read_alm(os.path.join(kappa_dir, kappa_map_fname)).astype('complex128')

        try:
            kappa_mask = enmap.read_map(os.path.join(kappa_dir, 'masks', kappa_mask_fname))
            kappa_mask = reproject.map2healpix(kappa_mask,
                                               nside=self.nside, lmax=None, out=None, rot=None,
                                               spin=[0,2], method="spline", order=0,
                                               extensive=False, bsize=100000,
                                               nside_mode="pow2", boundary="constant", verbose=False)
        except ValueError:
            kappa_mask = hp.read_map(os.path.join(kappa_dir, 'masks', kappa_mask_fname))**2.

        if remove_mask_apod:
            kappa_mask = np.piecewise(kappa_mask, [kappa_mask>=1-1e-10, kappa_mask<1-1e-10], [1, 0])

        if 'transfer_function' in nulltest.keys():
            transfer_dir = self.dirs['transfer_dir']
            transfer_name = nulltest['transfer_function']
            
            transfer_function = np.loadtxt(os.path.join(transfer_dir, transfer_name))
            transfer_function = np.nan_to_num(transfer_function, nan=1.0, posinf=1., neginf=1.) # first two values are numerically not defined

            kappa_alms = hp.almxfl(kappa_alms, transfer_function)

        kappa_map = hp.alm2map(np.nan_to_num(kappa_alms), nside=self.nside)

        kappa_field = nmt.NmtField(kappa_mask, [kappa_map], masked_on_input=masked_on_input)

        print('Done.')

        return kappa_field


    def setup_workspaces(self, nulltest):
        '''Set up the namaster NmtWorkspaces for each bin_tag corresponding to the combined masks being 
        used in the cross-correlation.

        Uses a workspace filename of [workspace_dir]/[tracer_mask]_[kappa_mask]_wsp.fits and looks 
        for a file already named this to load if it already exists. Otherwise creates a new workspace and
        saves it there.


        Parameters
        ----------
        nulltest
            dict containing the specification of the null test

        '''

        print('Getting coupling matrix...')

        workspaces = {}

        for ibin, bin_tag in enumerate(self.tracer_config['tracer_bin_tags']):

            tracer_mask_fname = self.tracer_config['tracer_masks'][bin_tag]
            kappa_mask_fname = nulltest['kappa_mask']

            workspace_fname =  os.path.join(self.dirs['workspace_dir'], '{}_{}_wsp.fits').format(tracer_mask_fname.rstrip('.fits'), kappa_mask_fname.rstrip('.fits')) # unique for each mask combination

            if os.path.exists(workspace_fname):
                print(f'Reading workspace for {bin_tag} from {workspace_fname} ...')
                workspaces[bin_tag] = nmt.NmtWorkspace()
                workspaces[bin_tag].read_from(workspace_fname)
            else:
                print(f'Creating new workspace for {bin_tag} and saving at {workspace_fname} ...')
                workspaces[bin_tag] = nmt.NmtWorkspace()
                workspaces[bin_tag].compute_coupling_matrix(nulltest['kappa_field'], self.tracer_fields[bin_tag], self.ell_bins)
                workspaces[bin_tag].write_to(workspace_fname)

        print('Done.')

    def measure_cls(self, nulltest):

        print('Measuring Cls...')

        cl_coupled = {}
        cl_decoupled = {}

        for ibin, bin_tag in enumerate(self.tracer_config['tracer_bin_tags']):

            print(f'{bin_tag}...')

            tracer_mask_fname = self.tracer_config['tracer_masks'][bin_tag]
            kappa_mask_fname = nulltest['kappa_mask']

            workspace_fname =  os.path.join(self.dirs['workspace_dir'], '{}_{}_wsp.fits').format(tracer_mask_fname.rstrip('.fits'), kappa_mask_fname.rstrip('.fits')) # unique for each mask combination

            workspace = nmt.NmtWorkspace()
            workspace.read_from(workspace_fname)

            cl_coupled[bin_tag] = nmt.compute_coupled_cell(self.tracer_fields[bin_tag], nulltest['kappa_field'])
            cl_decoupled[bin_tag] = workspace.decouple_cell(cl_coupled[bin_tag])[0]

            import pdb; pdb.set_trace()

        print('Done.')

        return cl_decoupled


if __name__ == '__main__':

    import pickle

    nullspectra = NullSpectrum(config_fname='scripts/null_list.yaml')

    nullspectra.setup_binning()
    nullspectra.setup_tracer_fields()

    for null_test in nullspectra.nulltests:

        print('Starting null test {}...'.format(null_test['name']))

        null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['name']))

        null_test['kappa_field'] = nullspectra.setup_act_field(null_test)
        
        nullspectra.setup_workspaces(null_test)
        cl_decoupled = nullspectra.measure_cls(null_test)

        pickle.dump(cl_decoupled, open(null_test_cl_fname, 'wb'))


