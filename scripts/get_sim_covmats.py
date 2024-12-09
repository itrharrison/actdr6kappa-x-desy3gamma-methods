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
        self.blind = ini['config']['blind']
        self.analysis_range = ini['config']['analysis_range']
        self.config_fname = config_fname

        # check if environment variable has been specified for root dir
        if self.dirs['root'].startswith('${'):
            self.dirs['root'] = os.environ.get(self.dirs['root'].lstrip('${').rstrip('}'))

    def _plot_analysis_range(self, ax, ells):
        ax.fill_between(np.arange(ells.max()), ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(np.arange(ells.max()) > self.analysis_range[1]),
                    color='silver', alpha=0.5)
        ax.fill_between(np.arange(ells.max()), ax.get_ylim()[0], ax.get_ylim()[1],
                    where=(np.arange(ells.max()) < self.analysis_range[0]),
                    color='silver', alpha=0.5)

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

                null_test['kappa_map'] = null_test['sim_maps'] + '_{:04d}.fits'.format(sim_tag + 1)

                null_test['kappa_field'] = nullspectra.setup_act_field(null_test, sim_map=True)

                if sims_cross_mode=='sims':
                    nullspectra.setup_tracer_fields(isim=isim + 1)
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

            if 'transfer_function' in null_test.keys():
                transfer_dir = self.dirs['transfer_dir']
                transfer_name = null_test['transfer_function']
            
                transfer_function = np.loadtxt(os.path.join(self.dirs['root'], transfer_dir, transfer_name))
                transfer_function = np.nan_to_num(transfer_function, nan=1.0, posinf=1., neginf=1.) # first two values are numerically not defined

                transfer_function = nullspectra.ell_bins.bin_cell(transfer_function)

            else:
                transfer_function = np.ones_like(cl_fid[bin_tag])

            if 'transfer_function_baseline' in null_test.keys():
                transfer_dir = self.dirs['transfer_dir']
                transfer_name = null_test['transfer_function']
            
                transfer_function_baseline = np.loadtxt(os.path.join(self.dirs['root'], transfer_dir, transfer_name))
                transfer_function_baseline = np.nan_to_num(transfer_function_baseline, nan=1.0, posinf=1., neginf=1.) # first two values are numerically not defined

                transfer_function_baseline = nullspectra.ell_bins.bin_cell(transfer_function_baseline)

            else:
                transfer_function_baseline = np.ones_like(cl_fid[bin_tag])

            n_tracer_bins = len(nullspectra.tracer_config['tracer_bin_tags'])

            sim_tags_list = np.arange(self.nsims)

            spectra_arr = np.zeros([n_tracer_bins, nullspectra.ell_bins.get_n_bands(), self.nsims])


            for isim, sim_tag in enumerate(sim_tags_list):

                null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls_sim_{}.pkl'.format(null_test['name'], sim_tag + 1))
                spectra = pickle.load(open(null_test_cl_fname, 'rb'))

                for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):

                    spectra_arr[ibin, :, isim] = spectra[bin_tag] / transfer_function

                    if not null_test['map_null']:
                        baseline_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls_sim_{}.pkl'.format(null_test['baseline_name'], sim_tag + 1))
                        baseline_cl_decoupled = pickle.load(open(baseline_cl_fname, 'rb'))
                        spectra_arr[ibin, :, isim] = (spectra[bin_tag] / transfer_function - baseline_cl_decoupled[bin_tag] / transfer_function_baseline) / cl_fid[bin_tag]

                        # import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()

            self.covmat_list = {}
            self.corrcoef_list = {}

            for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):

                self.covmat_list[bin_tag] = np.cov(spectra_arr[ibin])
                self.corrcoef_list[bin_tag] = np.corrcoef(spectra_arr[ibin])

            self.covmat_list['total'] = np.cov(spectra_arr.reshape(nullspectra.ell_bins.get_n_bands() * n_tracer_bins, self.nsims))
            self.corrcoef_list['total'] = np.corrcoef(spectra_arr.reshape(nullspectra.ell_bins.get_n_bands() * n_tracer_bins, self.nsims))

            null_test_covmat_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_covmat.pkl'.format(null_test['name']))
            null_test_corrcoef_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_corrcoef.pkl'.format(null_test['name']))

            pickle.dump(self.covmat_list, open(null_test_covmat_fname, 'wb'))
            pickle.dump(self.corrcoef_list, open(null_test_corrcoef_fname, 'wb'))

            if plot_dir:

                plt.close('all') # tidy up any unshown plots

                # data_spectra = spectra_arr[:,:,0] # goose this for now

                null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['name']))

                plt.figure(1, figsize=(2 * 4.5, 2 * 3.75))

                if null_test['map_null']:
                    plt.suptitle(null_test['name'].split('-')[0])
                    ylabel = '$C^{\kappa_{\\rm ' + null_test['name'].split('-')[0].replace('_', '\_') + '}\gamma}_{\ell}$'
                    ylims = [-0.5e-9, 0.5e-9]
                    data_spectra = pickle.load(open(null_test_cl_fname, 'rb'))
                else:
                    plt.suptitle('{}$-${}'.format(null_test['name'].split('-')[0], null_test['baseline_name'].split('-')[0]))
                    ylabel = '($C^{\kappa_{\\rm ' + null_test['name'].split('-')[0].replace('_', '\_') + '}\gamma}_{\ell} - C^{\kappa_{\\rm ' + null_test['baseline_name'].split('-')[0].replace('_', '\_') + '}\gamma}_{\ell}) / C^{\kappa\gamma, {\\rm th.}}_{\ell}$'
                    ylims = [-5, 5]
                    baseline_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls.pkl'.format(null_test['baseline_name']))
                    
                    data_spectra = {}

                    for ibin, bin_tag in enumerate(nullspectra.tracer_config['tracer_bin_tags']):
                        data_spectra[bin_tag] = (pickle.load(open(null_test_cl_fname, 'rb'))[bin_tag] / transfer_function - pickle.load(open(baseline_cl_fname, 'rb'))[bin_tag] / transfer_function_baseline ) / cl_fid[bin_tag]

                # import pdb; pdb.set_trace()

                plt.subplot(221)
                plt.title('Bin 1')
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin1']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[0], color='k', alpha=0.01)
                if self.blind:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), np.zeros_like(data_spectra['bin1']), yerr=err_bars, c='C0')
                else:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin1'], yerr=err_bars, c='C0')
                plt.ylabel(ylabel)
                plt.xlabel('$\ell$')
                plt.ylim(ylims)
                plt.xlim([0, nullspectra.ell_bins.get_effective_ells().max()])
                self._plot_analysis_range(plt.gca(), nullspectra.ell_bins.get_effective_ells())

                plt.subplot(222)
                plt.title('Bin 2')
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin2']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[1], color='k', alpha=0.01)
                if self.blind:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), np.zeros_like(data_spectra['bin2']), yerr=err_bars, c='C1')
                else:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin2'], yerr=err_bars, c='C1')
                # plt.ylabel(ylabel)
                plt.xlabel('$\ell$')
                plt.ylim(ylims)
                plt.xlim([0, nullspectra.ell_bins.get_effective_ells().max()])
                self._plot_analysis_range(plt.gca(), nullspectra.ell_bins.get_effective_ells())

                plt.subplot(223)
                plt.title('Bin 3')
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin3']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[2], color='k', alpha=0.01)
                if self.blind:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), np.zeros_like(data_spectra['bin3']), yerr=err_bars, c='C2')
                else:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin3'], yerr=err_bars, c='C2')
                plt.ylabel(ylabel)
                plt.xlabel('$\ell$')
                plt.ylim(ylims)
                plt.xlim([0, nullspectra.ell_bins.get_effective_ells().max()])
                self._plot_analysis_range(plt.gca(), nullspectra.ell_bins.get_effective_ells())

                plt.subplot(224)
                plt.title('Bin 4')
                plt.axhline(0, color='k', linestyle='dashed', alpha=0.4)
                err_bars = np.sqrt(np.diag(self.covmat_list['bin4']))
                plt.plot(nullspectra.ell_bins.get_effective_ells(), spectra_arr[3], color='k', alpha=0.01)
                if self.blind:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), np.zeros_like(data_spectra['bin4']), yerr=err_bars, c='C3')
                else:
                    plt.errorbar(nullspectra.ell_bins.get_effective_ells(), data_spectra['bin4'], yerr=err_bars, c='C3')
                # plt.ylabel(ylabel)
                plt.xlabel('$\ell$')
                plt.ylim(ylims)
                plt.xlim([0, nullspectra.ell_bins.get_effective_ells().max()])
                self._plot_analysis_range(plt.gca(), nullspectra.ell_bins.get_effective_ells())

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
                # plt.imshow(self.corrcoef_list['total'], origin='lower')

                plt.axhline(0.0, color='k')#, linestyle='dashed', alpha=0.4)

                plt.subplots_adjust(hspace=0.3)

                if null_test['map_null']:
                    plt.savefig(os.path.join(plot_dir, 'covmat_{}.png'.format(null_test['name'])), dpi=300, bbox_inches='tight')
                else:
                    plt.savefig(os.path.join(plot_dir, 'covmat_{}-{}.png'.format(null_test['name'], null_test['baseline_name'])), dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    import pickle

    nullcovmats = NullCovMat(config_fname='./scripts/null_list_publicmaps.yaml')

    # nullcovmats.make_covmat(plot_dir='./figures')
    nullcovmats.get_spectra_set(sims_cross_mode='sims')

