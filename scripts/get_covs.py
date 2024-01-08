import os
import math
import numpy as np
import yaml
import pickle

import healpy as hp
import pymaster as nmt
from pixell import enmap, reproject

from get_spectra import NullSpectrum

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('Setting up...')
    # read setup
    nullspectra = NullSpectrum(config_fname='scripts/null_list.yaml')

    # make sure binning and workspaces are set up
    nullspectra.setup_binning()
    nullspectra.setup_tracer_fields()

    baseline_test = nullspectra.nulltests[0]

    baseline_test['kappa_field'] = nullspectra.setup_act_field(baseline_test)

    nullspectra.setup_workspaces(baseline_test)
    print('Done.')
else:
    nullspectra = None
    baseline_test = None

nullspectra = comm.bcast(nullspectra, root=0)
baseline_test = comm.bcast(baseline_test, root=0)

comm.barrier()

# us the sim map instead, 400 times
# tasks are distributed here
sim_tags_list = np.arange(rank, nullspectra.nsims, size)

for sim_tag in enumerate(sim_tags_list):

    print(f'Starting on sim {sim_tag}')

    sim_map_name = baseline_test['sim_maps'].replace('sim_tag', sim_tag)
    baseline_test['kappa_map'] = os.path.join(root, nullspectra.dirs['sim_dir'], sim_map_name)
    baseline_test['kappa_field'] = nullspectra.setup_act_field(baseline_test)

    cl_decoupled[sim_tag] = nullspectra.measure_cls(baseline_test)

# consolidate results here
comm.barrier()
tot_cl_decoupled = comm.reduce(cl_decoupled, op=lambda x,y: x|y, root=0)
comm.barrier()

if rank == 0:
    # form the covmat and output
    pickle.dump(tot_cl_decoupled, open('./test_cov.pkl', 'wb'))

