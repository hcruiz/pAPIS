#from __future__ import print_function
import sys
import os
#sys.path.append("../../")
import numpy as np
from mpi4py import MPI
import scipy.io as sio
import copy

"""
Created on Sat April 22 2017

The class APIS generates trajectories/particles integrating the stochastic system defined in the Smoothing_Problem class, it estimates its costs and unnorm_weights and using these, it estimates the control parameters defined in Smoothing_Problem. (In the future: It also trains the model of the system using the EM-algorithm)

This class:
    -Initializes all data variables, e.g. particles, costs (S=V+C_u,V=neg_logLikelihood,C_u), weights, ESS, ESS_raw, mean_post, var_post, norm_psi, the correlation matrices between the basis functions corr_basis=H and the correlations between noise realizations and basis_functions corr_basisnoise=dQ(h).
    -Estimates all above quantities from the particles.
    -Implements annealing
    -Implements a resampling funtion to get the smoothing particles according to the weights
    (-Implements the EM algorithm in update_parameters)

An istance of the class is initialized in the main program and gets as argument an instance of Smoothing_Problem class. particles are generated, weights and all necessary estimations for the updates are computed. A resampling function to get the smoothing particles is also defined.
NOTE: The functions with the prefix nn in their name, use only the weights exp(-S) that are NOT normalized!
@author: HCRuiz
"""

class APIS(object):
    
    def __init__(self,smproblem):
        self.smp = smproblem
        self.comm = self.smp.comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.N_particles = self.smp.N_particles
        self.total_number_particles = self.comm.Get_size()*self.N_particles
        if self.rank==0: print "Total Number of Particles: ", self.total_number_particles
        self.learning_rate = self.smp.meta_params["learning_rate"]
        self.anneal_threshold = self.smp.meta_params["anneal_threshold"]
        
        self.anneal_factor = self.smp.meta_params["anneal_factor"]
        self.dt = self.smp.dt
        self.timepoints = self.smp.timepoints
        self.dim = self.smp.dim
        self.dim_control = self.smp.dim_control
        self.dim_obs = self.smp.dim_obs
        self.sigma_dyn = self.smp.sigma_dyn
        ## Define the IS initialization parameters; this are updated to the posterior mean and stdev in adapt_initialization() ##
        self.mean_q0 = self.smp.mean_prior0
        self.sigma_q0 = self.smp.sigma_prior0
        
        if self.rank==0: 
            self.step_olc = np.zeros_like(self.smp.openloop_term)
            self.step_fbck = np.zeros_like(self.smp.feedback_term)
            print "anneal_threshold = ", self.anneal_threshold 
            print "dt = ",self.dt
            print "NOTICE: only the root=0 has the global statistics!"
            if "-save" not in sys.argv: print "WARNING: Data will NOT be saved!!"
         
        # Initialize general variables
        self.Iters = self.smp.meta_params["Iters"] ## TODO: clean code; instead of methods receiving the iteration number, add a counter from 0 to Iters
        self.particle_stats_exist = False
        if self.rank==0:
            self.rawESS_itrs, self.ESS_itrs = np.zeros(self.Iters), np.zeros(self.Iters)
            self.meanS0_itrs, self.meanStateCost_itrs, self.meanControlCost_itrs = np.zeros(self.Iters), np.zeros(self.Iters), np.zeros(self.Iters)
            self.meanCost_itrs, self.varCost_itrs = np.zeros(self.Iters), np.zeros(self.Iters)
            self.varS0_itrs, self.varStateCost_itrs, self.varControlCost_itrs = np.zeros(self.Iters), np.zeros(self.Iters), np.zeros(self.Iters)
            self.covS0V_itrs, self.covS0Cu_itrs, self.covVCu_itrs = np.zeros(self.Iters), np.zeros(self.Iters), np.zeros(self.Iters)
            self.wS0_itrs = np.zeros(self.Iters)
            self.wStateCost_itrs, self.wControlCost_itrs, self.wCost_itrs = np.zeros(self.Iters),np.zeros(self.Iters),np.zeros(self.Iters)
            self.wvarCost_itrs = np.zeros(self.Iters)
            # Storage for parameters
            self.SigmaDyn_itrs = np.zeros((self.Iters,self.sigma_dyn.shape[0]))
            self.SigmaDyn_itrs[:,:] = np.diag(self.sigma_dyn)

    ############################################################ FUNCTIONS ###########################################################
    
    ##############################
    ######### Get Data ###########
    ##############################
    def generate_particles(self):
        if self.rank==0:print "Generating particles.."
        #############################
        ##### Initialize Data #######
        #############################
        self.Particles = np.zeros([self.N_particles,self.dim,self.timepoints])
        self.local_obsSignal = np.zeros([self.N_particles,self.dim_obs,self.timepoints])
        
        #self.Noise = np.zeros([self.N_particles,self.dim_control,self.timepoints])
        self.U_xt = np.zeros([self.N_particles,self.dim_control,self.timepoints])
        self.Basis_xt = np.zeros([self.N_particles,self.dim_control,self.timepoints])

        self.local_Vparticles = np.zeros([self.N_particles,]) # State cost
        self.local_Cuparticles = np.zeros([self.N_particles,]) # Control cost
        self.unnorm_local_weights = np.zeros([self.N_particles,])

        self.corr_basis = np.zeros([self.dim_control,self.dim_control,self.timepoints]) 
        self.corr_basisnoise = np.zeros([self.dim_control,self.dim_control,self.timepoints]) 

        self.annealed_psi = None
        self.norm_psi = None 
        self.ESS = None
        self.ESS_raw = None
        
        #Initialize particles
        self.Particles[:,:,0], self.S0 = self._initialize_particles()
        self.Noise = np.sqrt(self.dt)*np.dot(self.sigma_dyn,np.random.randn(self.N_particles,self.dim_control,self.timepoints))

        #Propagate particles and estimate cost for each particle    
        for t_index in range(self.timepoints-1):
            state = self.Particles[:,:,t_index] 
            noise = self.Noise[:,:,t_index]
            #if self.rank==0: print t_index
            new_state, u, h_xt = self.smp.integration_step(state,noise,t_index)
            
            self.Particles[:,:,t_index+1] = new_state
            self.U_xt[:,:,t_index] = u
            self.Basis_xt[:,:,t_index] = h_xt
            # Compute Control and State Costs
            self.local_obsSignal[:,:,t_index] = self.smp.obs_signal(self.Particles[:,:,t_index],t_index)
            self.local_Vparticles += self.smp.neg_logLikelihood(self.local_obsSignal[:,:,t_index], t_index)
            upn = (u/2.)*self.dt + noise.T
            invSig2 = np.linalg.pinv(np.dot(self.sigma_dyn,self.sigma_dyn))
            u_invsig = np.dot(u,invSig2)
            self.local_Cuparticles += np.sum(u_invsig*upn,axis=1)
        
        self.local_obsSignal[:,:,t_index+1] = self.smp.obs_signal(self.Particles[:,:,t_index+1],t_index+1)
        self.local_Vparticles += self.smp.neg_logLikelihood(self.local_obsSignal[:,:,t_index+1], t_index+1)
        
        self.local_Sparticles = self.local_Vparticles + self.local_Cuparticles + self.S0
        self.global_Sparticles = np.array(self.comm.gather(self.local_Sparticles,root=0))
        #print "self.global_Sparticles is ", self.global_Sparticles, "in rank ",self.rank 
        
        self._get_nnweights() # Results are saved in self.unnorm_local_weights
        
        self.get_psi() #Gets both annealed and raw normalization constant psi
        self.posterior_mean()
        self.posterior_var()
        
        
    def _initialize_particles(self):
        #if self.rank==0:print "Initializing particles with axis-aligned Gaussian"
        #if self.rank==0: print "Mean q0",self.mean_q0,"Sigma q0",self.sigma_q0 
        P0 = self.mean_q0 + self.sigma_q0*np.random.randn(self.N_particles,self.dim)
        S0 = self._neglog_p0(P0) - self._neglog_q0(P0)
        return P0, S0
    
    def _get_nnweights(self): 
        
        if self.rank==0:
            temperature = 1. # This is the temperature maybe temperature can be chosen better, e.g. temperature ~ max(S)
            globalS = self.global_Sparticles.reshape((self.total_number_particles,))
            #print "shape of globalS: ",globalS.shape
            globalS -= np.min(globalS) # for numerical stability
            self._compute_ess(globalS)
            self.ESS_raw = self.ESS
            print "Raw ESS: ", self.ESS_raw,"\t anneal_threshold = ",self.anneal_threshold
            while self.ESS < self.anneal_threshold:
                temperature *= self.anneal_factor
                annealed_globalS = globalS/temperature
                self._compute_ess(annealed_globalS)
                sys.stdout.write('*')
                sys.stdout.flush()
                if temperature > 500000:
                    assert 0==1, "Annealing failed!"
                elif self.ESS >= self.anneal_threshold: sys.stdout.write('\n')
            
            #print "ESS = ",self.ESS#,"\t anneal_threshold = ",self.anneal_threshold
            self.unnorm_global_weights = np.exp(-globalS/temperature)
            sendbuf = self.unnorm_global_weights
            #print "global variable ",self.unnorm_global_weights
        else:
            sendbuf =  None

        self.comm.Scatter(sendbuf, self.unnorm_local_weights, root=0)
        #print  "Local variable ",self.unnorm_local_weights, "in worker ", self.rank
        
    def _compute_ess(self,globalS):
        #Compute the weights
        w = np.exp(-globalS)
        w = w/np.sum(w)
        assert np.isclose(np.sum(w),1.), "Weights are not normalized!"
        #Compute the (annealed) ESS
        self.ESS = 1./np.sum(w**2)
        self.ESS /= self.total_number_particles 
    
    def get_psi(self):
        self.annealed_psi = self.nnweighted_sum(np.ones([self.N_particles,1]))
        if self.rank==0:
            globalS = self.global_Sparticles.reshape((self.total_number_particles,))
            self.norm_psi = np.exp(-globalS).sum()
        #print self.norm_psi        
    
    def nnweighted_sum(self,x_local):
        '''Estimates the weighted sum of x by first computing the weighted sum with the local weights and local data; then it passes the result to root=0 and root summs all local results. Only root gives the global weighted sum as answer, all other workers give a string saying "Only root has this!"
        '''
        assert x_local.shape[0]==self.N_particles, "Wrong dimensionalities! The 0th dimension must be "+str(self.N_particles)+"but it is "+str(x_local.shape[0])
        #print x_local.shape,"x_local shape"
        ax = np.arange(len(x_local.shape))
        if len(x_local.shape) > 2: ax[0],ax[-2] = ax[-2],ax[0] 
        #print x_local.transpose(*ax).shape,"x_local transpose"
        wx_local = np.dot(self.unnorm_local_weights,x_local.transpose(*ax))
        min_in_rank =  np.absolute(np.min(wx_local))
        #if np.isclose(min_in_rank,10**(-10)):
            #print "Min in rank :",min_in_rank, "in rank ", self.rank
            #sys.stdout.flush()
            
        if self.rank==0:
            lst = [self.size]
            shape_lst = [a for a in x_local.shape[1:]]
            lst += shape_lst
            #print lst
            all_summands_list = np.zeros(lst)
            min_all = np.zeros(self.size)
        else:
            all_summands_list = None
            min_all = None
                
        self.comm.Gather(wx_local,all_summands_list,root=0)
        self.comm.Gather(min_in_rank,min_all,root=0)
        all_summands = np.array(all_summands_list)
        
        if self.rank==0: 
            #print "shape of all summands:",all_summands.shape
            #print "min all_summands \n",np.min(np.absolute(all_summands))
            #print "Min in ranks :",min_all
            #print "###########################################"
            #sys.stdout.flush()
            wx_global = all_summands.sum(axis=0)
            #print "Type of np array:",wx_global.dtype
        else: wx_global = "Only root has this!"
        return wx_global
    
    def posterior_mean(self):
        '''Has dimensions (dim,timepoints)
        '''
        nn_wm = self.nnweighted_sum(self.Particles)
        if self.rank==0:
            self.mean_post = nn_wm/self.annealed_psi
        else: self.mean_post = "Only root has mean_post!"
        #self.mean_post = self.comm.bcast(self.mean_post,root=0)    
        #print "mean_post =",self.mean_post
        
    def posterior_var(self):
        '''Has dimensions (dim,timepoints)
        '''
        xx = self.Particles**2
        nn_wxx = self.nnweighted_sum(xx)
        if self.rank==0:
            self.var_post = nn_wxx/self.annealed_psi - self.mean_post**2 
        else: self.var_post = "Only root has var_post!"
        #self.var_post = self.comm.bcast(self.var_post,root=0)
        #print "var_post =",self.var_post
    
    def posterior_obssignal(self):
        nn_wobssignal = self.nnweighted_sum(self.local_obsSignal)
        nn_wos2 = self.nnweighted_sum(self.local_obsSignal**2)
            
        if self.rank==0:
            self.mean_postObsSignal = nn_wobssignal/self.annealed_psi
            self.var_postObsSignal = nn_wos2/self.annealed_psi - self.mean_postObsSignal**2 
        else: 
            self.mean_postObsSignal = "Only root has mean_postObsSignal!"
            self.var_postObsSignal = "Only root has var_postObsSignal!"
            
    def get_statistics(self,itr): 
        ''' Gets the particle and weighted statistics of costs in each iteration & the posterior of observation signal at the last iteration'''
        ##################################### PARTICLE STATISTICS #################################################################
        self.particle_stats_exist = True
        self.global_S0 = np.array(self.comm.gather(self.S0,root=0))
        self.global_Vparticles = np.array(self.comm.gather(self.local_Vparticles,root=0))
        self.global_Cuparticles = np.array(self.comm.gather(self.local_Cuparticles,root=0))

        if self.rank==0:             
            self.rawESS_itrs[itr], self.ESS_itrs[itr] = self.ESS_raw, self.ESS
            
            self.global_S0 = self.global_S0.reshape((self.total_number_particles,))
            self.global_Vparticles = self.global_Vparticles.reshape((self.total_number_particles,))
            self.global_Cuparticles = self.global_Cuparticles.reshape((self.total_number_particles,))
            
            self.meanS0 = np.mean(self.global_S0)
            self.meanStateCost, self.meanControlCost = np.mean(self.global_Vparticles), np.mean(self.global_Cuparticles)
            
            self.meanS0_itrs[itr] = self.meanS0
            self.meanStateCost_itrs[itr], self.meanControlCost_itrs[itr] = self.meanStateCost, self.meanControlCost
            self.meanCost_itrs[itr] = self.meanStateCost + self.meanControlCost + self.meanS0
            
            self.varCost_itrs[itr] = np.var(self.global_Sparticles)
            C = np.vstack((self.global_S0, self.global_Vparticles, self.global_Cuparticles))
            covCostMatrix = np.cov(C)
            #print "covCostMatrix.shape = ", covCostMatrix.shape          
            self.varS0_itrs[itr] = covCostMatrix[0,0]
            self.varStateCost_itrs[itr], self.varControlCost_itrs[itr] = covCostMatrix[1,1], covCostMatrix[2,2]
            self.covS0V_itrs[itr] = covCostMatrix[0,1]
            self.covS0Cu_itrs[itr] = covCostMatrix[0,2]
            self.covVCu_itrs[itr] = covCostMatrix[1,2]
            
        ########################################## WEIGHTED STATISTICS ##################################################################
        
        nn_wS0 = self.nnweighted_sum(self.S0)
        nn_wV = self.nnweighted_sum(self.local_Vparticles)
        nn_wCu = self.nnweighted_sum(self.local_Cuparticles)
        nn_wS2 = self.nnweighted_sum(self.local_Sparticles**2)
        
        if self.rank==0:
            self.wS0_itrs[itr] = nn_wS0/self.annealed_psi
            self.wStateCost_itrs[itr] = nn_wV/self.annealed_psi
            self.wControlCost_itrs[itr] = nn_wCu/self.annealed_psi
            self.wCost_itrs[itr] = self.wStateCost_itrs[itr] + self.wControlCost_itrs[itr]
            
            self.wvarCost_itrs[itr] = nn_wS2/self.annealed_psi - self.wCost_itrs[itr]**2
            #TODO: weighted correlations
        
        ########################################## POSTERIOR OF OBSERVATION SIGNAL ######################################################
        if itr==(self.Iters-1):
            nn_wobssignal = self.nnweighted_sum(self.local_obsSignal)
            nn_wos2 = self.nnweighted_sum(self.local_obsSignal**2)
            
            if self.rank==0:
                self.mean_postObsSignal = nn_wobssignal/self.annealed_psi
                self.var_postObsSignal = nn_wos2/self.annealed_psi - self.mean_postObsSignal**2 
            else: 
                self.mean_postObsSignal = "Only root has mean_postObsSignal!"
                self.var_postObsSignal = "Only root has var_postObsSignal!"
                
    def _get_nncorrelations(self):
        if self.rank==0:
            self.hh_t = np.zeros([self.dim_control,self.dim_control,self.timepoints])
            self.hdW_t = np.zeros([self.dim_control,self.dim_control,self.timepoints])
            self.dW_t = np.zeros([self.dim_control,self.timepoints])
        
        for d in np.arange(self.dim_control): #loop over controlled dimensions (and dims of noise)
            dW_t_d = self.nnweighted_sum(self.Noise[d,:,:])
            if self.rank==0: self.dW_t[d,:] = dW_t_d
            for k in np.arange(self.dim_control): #loop over basis functions
                hh_t_dk = self.nnweighted_sum(self.Basis_xt[:,k,:]*self.Basis_xt[:,d,:])
                hdW_t_dk = self.nnweighted_sum(self.Basis_xt[:,k,:]*self.Noise[d,:,:])
                if self.rank==0:
                    self.hh_t[d,k,:] = hh_t_dk
                    self.hdW_t[d,k,:] = hdW_t_dk
    
    ##############################    
    ### Adaptive Initialization ##
    ##############################
    def adapt_initialization(self):
        if self.rank==0:
            self.mean_post0 = self.mean_post[:,0]
            self.var_post0 = self.var_post[:,0]
            if any(self.var_post0<=0): 
                print "ERROR: Variance is NEGATIVE!!"
                print "Minimum Variance posterior at t=0:",np.min(self.var_post0)
                sys.stdout.flush()
                self.comm.Abort()
        else: 
            self.mean_post0 = None
            self.var_post0 = None
        self.mean_post0 = self.comm.bcast(self.mean_post0,root=0)        
        self.var_post0 = self.comm.bcast(self.var_post0,root=0)
        
        assert (self.var_post0>0).any(), "Variance of posterior at t=0 is zero in dim. "+str(range(self.dim)[np.isclose(self.var_post0,0)])
        #if any(self.var_post0<=0): 
        #    "ERROR: Variance is NEGATIVE!!"
        #    self.comm.Abort()
            
        self.mean_q0 = self.mean_post0
        self.sigma_q0 = np.sqrt(self.var_post0)
        #if self.rank==0: print "Initialization updated: mean_q0 =",self.mean_q0, "sigma_q0 =", self.sigma_q0
    
    def _neglog_p0(self,state):
        '''Negative log-prior at initial time. It's implemented as an axis-aligned Gaussian. Other priors are possible, but need implementation. 
        '''
        state_minus_mean = state - self.smp.mean_prior0
        #print "Shape of (state_minus_mean/self.sigma_prior0",(state_minus_mean/self.smp.sigma_prior0).shape
        return 0.5*np.sum((state_minus_mean/self.smp.sigma_prior0)**2,axis=1)
    
    def _neglog_q0(self,state):
        '''Negative log. of proposal distribution at initial time. It's implemented as an axis-aligned Gaussian. Other importance samplers are possible, but need implementation. 
        '''
        state_minus_mean = state - self.mean_q0
        return 0.5*np.sum((state_minus_mean/self.sigma_q0)**2,axis=1)
    
    ##############################    
    ##### Train Controller #######
    ##############################
    def update_controller(self,*args):
        #if self.rank==0:print "Updating Control Parameters"
        self._update_basisfunct()
        self._update_control_params(*args)
        
    def _update_basisfunct(self):
        '''Here the mu_control and sigma_control parameters of the basis functions are updated with the new estimations
        '''
        if self.rank==0:
            self.smp.mu_control = self.mean_post[0:self.dim_control,:]
            self.smp.sigma_control = np.sqrt(self.var_post[0:self.dim_control,:])
        # Broadcast the updated variables to all workers
        self.smp.mu_control = self.comm.bcast(self.smp.mu_control,root=0)
        self.smp.sigma_control = self.comm.bcast(self.smp.sigma_control,root=0)
        #if self.rank==1: print "mu_control: ",self.smp.mu_control," in worker ", self.rank
        
    def _update_control_params(self,*args):
        self._get_nncorrelations() # Only root=0 has the correlations!!
        #Update_step_A = np.zeros([self.dim_control,self.dim_control,self.timepoints])
        if self.rank==0: 
            #print args
            for t in np.arange(self.hh_t.shape[-1]):
                H_inv = np.linalg.pinv(self.hh_t[:,:,t]/self.annealed_psi)
                if 'momentum' in args:
                    if t==0 and len(args)==2:
                        eta=args[1]
                    else:
                        eta=0.9
                        #self.learning_rate = 1. - eta
                        if t==0: print "Update with momentum. eta=",eta
                        #print "Shape of hdW_t:",self.hdW_t.shape
                        self.step_olc[:,t] = eta*self.step_olc[:,t] + self.learning_rate*self.dW_t[:,t]/(self.dt*self.annealed_psi)
                        self.smp.openloop_term[:,t] += self.step_olc[:,t]
                        self.step_fbck[:,:,t] = eta*self.step_fbck[:,:,t] + self.learning_rate*self.hdW_t[:,:,t]*H_inv/(self.dt*self.annealed_psi)
                        self.smp.feedback_term[:,:,t] += self.step_fbck[:,:,t]
                elif 'movavg' in args:
                    if t==0: 
                        itr = args[1]
                        print "Moving Average Update..." #This update decreases the off-diagonal terms s.t. initial noise is forgotten, but the resultig parameters are smaller (0.5) than with the standard update... 
                    self.smp.openloop_term[:,t] = (1.-self.learning_rate)*self.smp.openloop_term[:,t] + self.learning_rate*self.dW_t[:,t]/(self.dt*self.annealed_psi)
                    #self.smp.openloop_term[:,t] /= (1-(1.-self.learning_rate)**(itr+1.)) NOT WORKING!
                    self.smp.feedback_term[:,:,t] = (1.-self.learning_rate)*self.smp.feedback_term[:,:,t] + self.learning_rate*self.hdW_t[:,:,t]*H_inv/(self.dt*self.annealed_psi)
                else:
                    if t==0: print "Standard Update..."
                    #print "Shape of hdW_t:",self.hdW_t.shape
                    self.smp.openloop_term[:,t] += self.learning_rate*self.dW_t[:,t]/(self.dt*self.annealed_psi)
                    self.smp.feedback_term[:,:,t] += self.learning_rate*self.hdW_t[:,:,t]*H_inv/(self.dt*self.annealed_psi)   
                    #Although mathematically doesn't matter the normalization constant \psi, it is important for stability. Without normalization, the results vary much more!
                    #Update_step_A[:,:,t] = self.hdW_t[:,:,t]*H_inv/(self.dt*self.annealed_psi)
            #print "Update step feedback controller: \n",np.sum(Update_step_A,axis=2)
            #print "Eigenvals of feedback matrix: ",np.linalg.eigvals(np.sum(self.smp.feedback_term,axis=2))
        self.smp.openloop_term = self.comm.bcast(self.smp.openloop_term,root=0)
        self.smp.feedback_term = self.comm.bcast(self.smp.feedback_term,root=0)
    
    def smooth_controller(self,steps):
        if self.rank==0: print "Smoothing controller by "+str(2*steps+1)+"-point average"
        openloop_temp = np.zeros_like(self.smp.openloop_term)
        feedback_temp = np.zeros_like(self.smp.feedback_term)
        openloop_temp[:,0:steps] = self.smp.openloop_term[:,0:steps]
        feedback_temp[:,:,0:steps] = self.smp.feedback_term[:,:,0:steps]
        for t in np.arange(steps,self.timepoints-(steps+1)):
            openloop_temp[:,t] = np.mean(self.smp.openloop_term[:,t-steps:t+(steps+1)],axis=-1)
            feedback_temp[:,:,t] =  np.mean(self.smp.feedback_term[:,:,t-steps:t+(steps+1)],axis=-1)
        openloop_temp[:,-(steps+1):] = self.smp.openloop_term[:,-(steps+1):]
        feedback_temp[:,:,-(steps+1):] = self.smp.feedback_term[:,:,-(steps+1):]
        self.smp.openloop_term = openloop_temp
        self.smp.feedback_term = feedback_temp
        
    ##############################        
    ######## Train Model #########
    ##############################
    def update_parameters(self,itr,ess_threshold,*args):
        # TODO: put in a function that saves all parameters used in the model
        if self.rank==0  and itr==0 and ess_threshold>self.anneal_threshold:
            print "Updating Model Parameters if raw ESS >",ess_threshold

        if ess_threshold>self.anneal_threshold or ('-testing' in sys.argv): 
            if '-testing' in sys.argv and self.rank==0: 
                print "WARNING: Testing APIS; learning threshold might be smaller than annealing threshold"
            self.ESS_raw = self.comm.bcast(self.ESS_raw,root=0)
            # update noise term if sufficient ESS
            updating = self.ESS_raw>ess_threshold or ('-testing' in sys.argv)
            if 'sigma_dyn' in args and updating:
                ### Dynamic noise #####
                mean_t_uplusnoise2 = np.mean((self.U_xt*self.dt+np.swapaxes(self.Noise,0,1))**2,axis=2)
                #print "shape of mean_t_uplusnoise2:", mean_t_uplusnoise2.shape
                Sigdyn2 = self.nnweighted_sum(mean_t_uplusnoise2) #mean_t uplusnoise**2 is the weighted mean of the squared of uplusnoise over time
                if self.rank==0:
                    #print "old sigma: ",self.sigma_dyn
                    #### ONLY IMPLEMENTED FOR DIAGONAL Sigma_dyn!! ##########
                    Sigdyn2 /= self.annealed_psi*self.dt
                    mask = ~np.eye(self.sigma_dyn.shape[0],dtype=bool)
                    
                    sigma_dyn_proxy = copy.copy(self.sigma_dyn)
                    sigma_dyn_proxy[mask] = 1.
                    grad_sigmadyn = (Sigdyn2-self.sigma_dyn**2)/(2*sigma_dyn_proxy)
                    grad_sigmadyn[mask] = 0. 
                    #(self.timepoints/self.sigma_dyn**3)*(Sigdyn2-self.sigma_dyn**2) but use Hessian as learning rate
                    
                    print "Grad_SigmaDyn =",grad_sigmadyn
                    self.sigma_dyn += grad_sigmadyn
                    print "new sigma: ",self.sigma_dyn

                self.sigma_dyn = self.comm.bcast(self.sigma_dyn,root=0)
            
            # update parameters of observation signal if sufficient ESS
            if 'obs_signal' in args and self.ESS_raw>ess_threshold:
                ### Observation Signal #####
                local_GradLogL = self.smp.grad_obsSignal(self.Particles) #NxNum_obsParams
                GradLogL = self.nnweighted_sum(local_GradLogL) 
                #GradLogL is the weighted mean of the sum of gradients of neg. log-likelihood
                if self.rank==0: 
                    GradLogL /= self.annealed_psi
                    print "Grad Log-Likelihood:",GradLogL
                    #print "Shape of GradObs:", GradLogL.shape     
                GradLogL = self.comm.bcast(GradLogL,root=0) #broadcast the value from root=0 to all other workers           
                self.smp.update_obsSignal(itr,GradLogL) #updates all parameters of the observation signal
                
        if self.rank==0: self.SigmaDyn_itrs[itr] = np.diag(self.sigma_dyn)
            
    ####################################
    ######## Helper Functions ##########
    ####################################
    def save_data(self,*arg_list):
        '''Saves the standard data posterior mean & variance, observations, tobs, control parameters and (later) resampled particles.
        TODO?: If a list with variable names is given, it saves also those variables. If the variable does not exist it raises a warning'''
        if self.rank==0 and "-save" in sys.argv:
            index = sys.argv.index("-save")
            self.file_prefix = sys.argv[index+1]
            print "Data saved with prefix:", self.file_prefix
            self.save_var("meanPost",self.mean_post)
            self.save_var("varPost",self.var_post)
            self.save_var("norm_psi",self.norm_psi)
            ##TODO: self.save_var("smParticles",self.smParticles) 
            self.save_var("feedback_term",self.smp.feedback_term)
            self.save_var("openloop_term",self.smp.openloop_term)
            self.save_var("observations",self.smp.observations)
            self.save_var("t_obs",self.smp.t_obs)
            self.save_var("var_obs",self.smp.var_obs)
            self.save_var("meta_params",self.smp.meta_params)
            
            self.save_var("SigmaDyn_itrs",self.SigmaDyn_itrs)
            self.smp.save_data(self.save_var)
            if self.particle_stats_exist:
                self.save_var("meanPostObsSignal",self.mean_postObsSignal)
                self.save_var("rawESS_itrs",self.rawESS_itrs)
                self.save_var("ESS_itrs",self.ESS_itrs)
                self.save_var("meanS0_itrs",self.meanS0_itrs)
                self.save_var("meanStateCost_itrs",self.meanStateCost_itrs)
                self.save_var("meanControlCost_itrs",self.meanControlCost_itrs)
                self.save_var("meanCost_itrs",self.meanCost_itrs)

                self.save_var("varPostObsSignal",self.var_postObsSignal)
                self.save_var("varCost_itrs",self.varCost_itrs)        
                self.save_var("varS0_itrs",self.varS0_itrs)
                self.save_var("varStateCost_itrs",self.varStateCost_itrs)
                self.save_var("varControlCost_itrs",self.varControlCost_itrs)
                self.save_var("covS0V_itrs",self.covS0V_itrs)
                self.save_var("covS0Cu_itrs",self.covS0Cu_itrs)
                self.save_var("covVCu_itrs",self.covVCu_itrs)
                
                self.save_var("wS0_itrs",self.wS0_itrs)
                self.save_var("wStateCost_itrs",self.wStateCost_itrs)
                self.save_var("wControlCost_itrs",self.wControlCost_itrs)
                self.save_var("wCost_itrs",self.wCost_itrs)
                self.save_var("wvarCost_itrs",self.wvarCost_itrs)
                
            else: print "Only basic variables saved. To get all statistics call the APIS method get_statistics(itr)!"
            
        elif self.rank==0: print "WARNING: Data NOT Saved!!"
            
    def save_var(self,var_name,var):
            var_str = self.file_prefix+"_"+var_name
            print var_name+" saved!"
            np.save(var_str,var)
        
        
        
