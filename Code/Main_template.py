#from __future__ import print_function
import sys
import numpy as np
from mpi4py import MPI
from SmoothingProblem_template import LQ_Problem as SP
from APIS import APIS as Smoother

comm = MPI.COMM_WORLD
case_identifier = sys.argv[1]

#TODO: move meta_params definition into a module to keep main template clean
meta_params = {}
meta_params["Iters"] = 25
meta_params["steps_between_obs"] = 100
meta_params["N_particles"] = 200     # particles per worker
meta_params["learning_rate"] = 0.2
meta_params["anneal_threshold"] = 10 # in number of particles
meta_params["anneal_factor"] = 1.1

Iters = meta_params["Iters"]

#Instantiate objects
smproblem = SP(comm,meta_params)
apis = Smoother(smproblem)

for itr in np.arange(Iters):
    apis.generate_particles()
    apis.get_statistics(itr)
    apis.adapt_initialization()
    apis.update_controller()
    
apis.save_data() #Only saves if the flag -save is given in arg.sys
