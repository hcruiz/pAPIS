#from __future__ import print_function
import sys
from time import time
import numpy as np
from mpi4py import MPI
from MultDimBM_Problem import MultDim_BM_Problem as SP #from Test_Problem import Test_Problem as SP #
from APIS import APIS as Smoother

comm = MPI.COMM_WORLD

Iters = 200
case_identifier = None #sys.argv[1]

meta_params = {}
meta_params["Iters"] = Iters
meta_params["steps_between_obs"] = 100.
meta_params["N_particles"] = 2000     # particles per worker
meta_params["learning_rate"] = 0.03
meta_params["anneal_threshold"] = 100 # in number of particles
meta_params["anneal_factor"] = 1.15
ess_threshold = 1.01
params2update =  ['None']#['sigma_dyn'] #
regularizer= ['L1']
#Instantiate objects
smproblem = SP(comm,meta_params,case_identifier)
apis = Smoother(smproblem)

if comm.Get_rank()==0: 
    print "Number of iterations:", Iters
    N_t = apis.timepoints
    nr_rois = smproblem.nr_rois
    #array_pmeanZ = np.zeros([Iters,nr_rois,N_t])
    #array_pmeanBold = np.zeros([Iters,nr_rois,N_t])
    #array_meanZ = np.zeros([Iters,nr_rois,N_t])
    #array_meanBold = np.zeros([Iters,nr_rois,N_t])
    array_OLC = np.zeros([Iters,smproblem.dim_control,N_t])
    array_feedbackMatrix = np.zeros([Iters,smproblem.dim_control,smproblem.dim_control,N_t])
    
    start_time = time()
for itr in np.arange(Iters):
    if comm.Get_rank()==0: print "Iteration ",itr
    apis.generate_particles()
    apis.get_statistics(itr)
    apis.adapt_initialization()
    apis.update_controller() #'movavg',itr gives controll parameters that are half the value of the optimal ones, but it does forget the initial noise
    ####################################
    #mask = ~np.eye(smproblem.feedback_term.shape[0],dtype=bool)
    #smproblem.feedback_term[mask] = 0.  # A diagonal parametrization of the controller is important to avoid ESS decay
    ####################################
    #if comm.Get_rank()==0: print "Eigenvals of feedback matrix: ",np.linalg.eigvals(np.sum(smproblem.feedback_term,axis=2))
    apis.update_parameters(itr,ess_threshold,*params2update)
    apis.posterior_obssignal()
    
    local_meanZ = np.mean(apis.Particles[:,0,:],axis=0)
    global_meanZ = np.zeros(apis.Particles.shape[-1])
    comm.Reduce(local_meanZ,global_meanZ)  
    
    local_mean_obSignal = np.mean(apis.local_obsSignal,axis=0)
    global_mean_obsSignal = np.zeros(apis.local_obsSignal.shape[1:])
    comm.Reduce(local_mean_obSignal,global_mean_obsSignal)
    
    if comm.Get_rank()==0:
        #print global_meanZ.shape, global_mean_obsSignal.shape
        #array_pmeanZ[itr] = global_meanZ
        #array_pmeanBold[itr] = global_mean_obsSignal
        
        #array_meanZ[itr] = apis.mean_post[0]
        #array_meanBold[itr] = apis.mean_postObsSignal
        
        array_OLC[itr] = smproblem.openloop_term
        array_feedbackMatrix[itr] = smproblem.feedback_term
    
if comm.Get_rank()==0:
    elapsed = (time() - start_time)
    print 'Elapsed time for ',Iters,' iterations: ', elapsed, 'sec' 
    print 'Per itaration is on average:',elapsed/Iters
    
apis.save_data() #Only saves if the flag -save is given in arg.sys
if comm.Get_rank()==0 and "-save" in sys.argv:
    #apis.save_var("array_pmeanZ",array_pmeanZ)
    #apis.save_var("array_pmeanBold",array_pmeanBold)
    
    #apis.save_var("array_meanZ",array_meanZ)
    #apis.save_var("array_meanBold",array_meanBold)
    
    apis.save_var("array_OLC",array_OLC)
    apis.save_var("array_feedbackMatrix",array_feedbackMatrix)
