#from __future__ import print_function
import sys
import numpy as np
from mpi4py import MPI
from LQ_Grid_Problem import LQ_Grid_Problem as SP
from APIS_Grid2 import APIS_Grid as Smoother

comm = MPI.COMM_WORLD

Iters = 25
case_identifier = sys.argv[1]

meta_params = {}
meta_params["Iters"] = Iters
meta_params["steps_between_obs"] = 100
meta_params["N_particles"] = 2000     # particles per worker
meta_params["learning_rate"] = 0.075
meta_params["anneal_threshold"] = 10 # in number of particles
meta_params["anneal_factor"] = 1.1

#Instantiate objects
smproblem = SP(comm,meta_params)
apis = Smoother(smproblem)

for itr in np.arange(Iters):
    apis.generate_particles()
    apis.get_statistics(itr)
    apis.adapt_initialization()
    apis.update_controller()
    
apis.save_data() #Only saves if the flag -save is given in arg.sys