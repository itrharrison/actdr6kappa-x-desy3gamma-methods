import os
import yaml
import pickle

from get_spectra import NullSpectrum

class NullCovMat():

    def __init__(self, config_fname):

        with open(config_fname, 'r') as file:
            ini = yaml.safe_load(file)

        self.nsims = ini['config']['nsims']
        self.dirs = ini['dirs']
        self.nulltests = ini['nulltests']

        # check if environment variable has been specified for root dir
        if self.dirs['root'].startswith('${'):
            self.dirs['root'] = os.environ.get(self.dirs['root'].lstrip('${').rstrip('}'))

    def get_spectra_set(self):
        
        nullspectra = NullSpectrum(config_fname=self.config_fname)

        nullspectra.setup_binning()
        nullspectra.setup_tracer_fields()

        # distribute this nicely

        for null_test in nullspectra.nulltests:

            print('Starting null test {}...'.format(null_test['name']))

            for isim in np.arange(self.nsims):

                print('Gettign Cls from sim {}...'.format(isim))

                null_test_cl_fname = os.path.join(nullspectra.dirs['root'], nullspectra.dirs['cl_output_dir'], '{}_cls_sim_{}.pkl'.format(null_test['name'], isim))

                null_test['kappa_map'] = null_test['sims'] + '_{:i}.fits'.format(isim)
                null_test['kappa_field'] = nullspectra.setup_act_field(null_test, sim_map=True)

                nullspectra.setup_workspaces(null_test)

                cl_decoupled = nullspectra.measure_cls(null_test)

                pickle.dump(cl_decoupled, open(null_test_cl_fname, 'wb'))
