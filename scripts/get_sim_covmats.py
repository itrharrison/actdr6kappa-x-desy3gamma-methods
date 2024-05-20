import os
import yaml
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

from mpi4py import MPI

from get_spectra import NullSpectrum

class NullCovMat():

    def __init__(self, config_fname):

        with open(config_fname, 'r') as file:
            ini = yaml.safe_load(file)

        self.nsims = ini['config']['nsims']
        self.dirs = ini['dirs']
        self.nulltests = ini['nulltests']
        self.config_fname = config_fname

        # check if environment variable has been specified for root dir
        if self.dirs['root'].startswith('${'):
            self.dirs['root'] = os.environ.get(self.dirs['root'].lstrip('${').rstrip('}'))

    def get_spectra_set(self, sims_cross_mode='data'):
        
        nullspectra = NullSpectrum(config_fname=self.config_fname)

        nullspectra.setup_binning()
        nullspectra.setup_tracer_fields()

        # distribute this nicely

        for null_test in nullspectra.nulltests:

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # rank = 0
            # size = 1

            print('Starting null test {}...'.format(null_test['name']))

            sim_tags_list = np.arange(rank, self.nsims, size)

            for isim, sim_tag in enumerate(sim_tags_list):

                print('Getting Cls from sim {} on proc {}...'.format(sim_tag + 1, rank))

                null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls_sim_{}.pkl'.format(null_test['name'], sim_tag + 1))

                null_test['kappa_map'] = null_test['sim_maps'] + '_{:d}.fits'.format(sim_tag + 1)
                null_test['kappa_field'] = nullspectra.setup_act_field(null_test, sim_map=True)

                if sims_cross_mode=='sims':
                    nullspectra.setup_tracer_fields(isim=isim)
                else:
                    print('!! Crossing kappa sims with tracer data... !!')

                nullspectra.setup_workspaces(null_test)

                cl_decoupled = nullspectra.measure_cls(null_test)

                pickle.dump(cl_decoupled, open(null_test_cl_fname, 'wb'))

    def make_covmat(self, plot_dir=None):

        nullspectra = NullSpectrum(config_fname=self.config_fname)

        cl_theory = np.loadtxt('./figures/cl_unbinned_actplanck.txt') # improve this
        cl_theory = cl_theory.reshape([4, 1901])

        cl_fid = {}

        for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):

            cl_theory_bin = cl_theory[int(bin_tag.lstrip('bin')) - 1]
            cl_theory_ells = np.arange(len(cl_theory_bin))
            cls_to_bin = np.interp(np.arange(nullspectra.ell_bins.lmax + 1), cl_theory_ells, cl_theory_bin)
            cl_fid[bin_tag] = nullspectra.ell_bins.bin_cell(np.array([cls_to_bin]))[0]

        for null_test in nullspectra.nulltests:

            n_tracer_bins = len(nullspectra.tracer_config['tracer_bin_tags'])

            sim_tags_list = np.arange(self.nsims)

            spectra_arr = np.zeros([n_tracer_bins, nullspectra.ell_bins.get_n_bands(), self.nsims])

            if not null_test['map_null']:
                baseline_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['baseline_name']))
                baseline_cl_decoupled = pickle.load(open(baseline_cl_fname, 'rb'))

            for isim, sim_tag in enumerate(sim_tags_list):

                null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls_sim_{}.pkl'.format(null_test['name'], sim_tag + 1))
                spectra = pickle.load(open(null_test_cl_fname, 'rb'))

                for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):

                    spectra_arr[ibin, :, isim] = spectra[bin_tag]

                    if not null_test['map_null']:
                        spectra_arr[ibin, :, isim] = (spectra[bin_tag] - baseline_cl_decoupled[bin_tag]) / cl_fid[bin_tag]

            # import pdb; pdb.set_trace()

            self.covmat_list = {}

            for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):

                self.covmat_list[bin_tag] = np.cov(spectra_arr[ibin])

            self.covmat_list['total'] = np.cov(spectra_arr.reshape(nullspectra.ell_bins.get_n_bands() * n_tracer_bins, self.nsims))

            null_test_covmat_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_covmat.pkl'.format(null_test['name']))

            pickle.dump(self.covmat_list, open(null_test_covmat_fname, 'wb'))

            if plot_dir:

                plt.close('all') # tidy up any unshown plots

                # data_spectra = spectra_arr[:,:,0] # goose this for now

                null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['name']))

                data_spectra = pickle.load(open(null_test_cl_fname, 'rb'))

                plt.figure(1, figsize=(2 * 4.5, 2 * 3.75))

                # import pdb; pdb.set_trace()

                plt.subplot(221)
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin1']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[0], color='k', alpha=0.01)
                plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin1'], yerr=err_bars, c='C0')
                plt.ylim([-2.e-9, 2.e-9])

                plt.subplot(222)
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin2']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[1], color='k', alpha=0.01)
                plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin2'], yerr=err_bars, c='C1')
                plt.ylim([-2.e-9, 2.e-9])

                plt.subplot(223)
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin3']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[2], color='k', alpha=0.01)
                plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin3'], yerr=err_bars, c='C2')
                plt.ylim([-2.e-9, 2.e-9])

                plt.subplot(224)
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin4']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[3], color='k', alpha=0.01)
                plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin4'], yerr=err_bars, c='C3')
                plt.ylim([-2.e-9, 2.e-9])

                plt.xlabel('$\ell$')
                # if null_test['map_null']:
                #     plt.ylim([-1., 5.])
                #     plt.ylabel('$C^{\kappa_{\\rm ' + null_test['name'].replace('_', '\_') + '}\gamma}_{\ell}$')
                # else:
                #     plt.ylim([-1., 1.])
                #     plt.ylabel('($C^{\kappa_{\\rm ' + null_test['name'].replace('_', '\_') + '}\gamma}_{\ell} - C^{\kappa_{\\rm ' + null_test['baseline_name'].replace('_', '\_') + '}\gamma}_{\ell}) / C^{\kappa\gamma, {\\rm th.}}_{\ell}$')

                # plt.subplot(132)
                # plt.imshow(self.covmat_list['total'], origin='lower')

                # plt.subplot(133)
                # plt.imshow(self.corrcoeff_list['total'], origin='lower')

                plt.axhline(0.0, color='k')#, linestyle='dashed', alpha=0.4)

                plt.savefig(os.path.join(plot_dir, 'covmat_{}.png'.format(null_test['name'])), dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    import pickle

    nullcovmats = NullCovMat(config_fname='./scripts/null_list.yaml')

    # nullcovmats.make_covmat(plot_dir='./figures')

    nullcovmats.get_spectra_set(sims_cross_mode='sims')

