import numpy as np
import sys
import os
#sys.path.append("../../")
import scipy.io as sio
#import random as rnd
"""
Created on Fr Jun 02 2017

The class fMRI_Problem defines the fMRI smoothing problem of a network of ROIS for the APIS class 

This class contains:
    -A function loading the data: The data must contain an array with the timestamps of each observation. With this, the time horizon and the integration step are defined. The user chooses the number of integrationsteps between observations
    -Definition of the model parameters: state and controlled dimension, dyn. and observation noise variance, etc
    -Defines needed functions for dynamics and cost: drift, diffusion, negLog_likelihood
    -Defines the controller: its basis functions h(x,t) and the parameters A(t) and b(t) for the feedback and open-loop term respectively
    -Defines the integration step and uses it to obtain the variance of the stationary distribution of the problem
    -Defines the prior with the mean given by the user and the variance estimated
    -Here, the communicator of mpi is also initialized for parallelization

An istance of the class is initialized in the main program and passed as argument to the APIS class where particles are generated, weights and all necessary estimations for the updates are computed. The training also happens in APIS and a resampling function is also defined there.

@author: HCRuiz
"""

class MultDim_BM_Problem(object):        
        
    def __init__(self,comm,meta_params,case_identifier):
        self.comm = comm
        #####################################
        ## Algorithmic specific parameters ##
        #####################################
        self.meta_params = meta_params
        steps_between_obs = meta_params["steps_between_obs"]
        self.N_particles = meta_params["N_particles"]
        self.learning_rate = meta_params["learning_rate"]
        self.Iters = meta_params["Iters"]
        #self.anneal_threshold = meta_params["anneal_threshold"]
        #self.anneal_factor = meta_params["anneal_factor"]
        self.case_identifier = case_identifier
        #####################################
        ############ Data ###################
        #####################################
        
        self._load_data() ## Customize this helper function to load your data!! (see bottom for helper functions)
        
        self.T = self.t_obs[-1] #defines horizon time
        assert self.t_obs[0] == 0., "t_obs[0] is assumed to be 0!"
        #print "t_obs[1]",self.t_obs[1]
        self.dt = self.t_obs[1]/steps_between_obs
        self.timepoints = int(steps_between_obs*(self.T/self.t_obs[1]) + 1)   
        if self.comm.Get_rank()==0: print "# of time points: ",self.timepoints
            
        #####################################
        ##### Model specific parameters #####
        #####################################

        ################################# Change these values for your problem ################################
        #### System & Dynamics ####
        self.dim = self.nr_rois
        self.dim_control = self.nr_rois
        self._load_model_params()
        
        #### Observations ####
        self.index_obs_dim = range(self.dim_obs)  # sets the index of the observation signal 
        #print self.index_obs_dim
        #### Prior ####
        self.mean_prior0 = np.zeros(self.nr_rois) 
        #print "mean_prior0", self.mean_prior0
        #######################################################################################################        
        #Define an auxiliary matrix to transform lower dimensional controllers to the full state of the system
        self._aux_mat = np.zeros([self.dim_control,self.dim]) 
        for d in range(self.dim_control): self._aux_mat[d,d] = 1.
        #######################################################################################################
        
        #Define sigma_prior by integrating uncontrolled dynamics
        self._get_sigma_prior0()
        #self.cov_prior = np.diag(self.sigma_prior**2) #not sure if needed
        #self.SIGMA_prior = np.linalg.inv(self.cov_prior)
        ######################################################################################################
        
        if self.comm.Get_rank()==0: 
            print "Dim. of the system: ",self.dim
            print "Dim. controlled: ",self.dim_control
            print "mean_prior0 =",self.mean_prior0
            print "sigma_prior0 =", self.sigma_prior0
            print "NOTICE:\nEstimating sigma_prior0 from the stationary distribution of uncontrolled dynamics; if system has no stationary distribution OR purely deterministic d.o.f.; modify method _get_sigma_prior0()"
            #print 'Dim of mean', mean_prior.shape
         
        
        #####################################
        ######## Control Parameters #########
        #####################################
        self.feedback_term = np.zeros([self.dim_control,self.dim_control,self.timepoints])
        self.openloop_term = np.zeros([self.dim_control,self.timepoints])
        #These are the statistics to define the basis functions
        self.mu_control = self.mean_prior0[0:self.dim_control,np.newaxis]*np.ones([self.dim_control,self.timepoints]) 
        self.sigma_control = self.sigma_prior0[0:self.dim_control,np.newaxis]*np.ones([self.dim_control,self.timepoints])
        if self.comm.Get_rank()==0: 
            print "NOTICE: The basis functions are normalized using the stats of prior process mean_prior0 & sigma_prior0"
        #########################################
        ####### Tests for dimensionality ########
        #########################################
        self._testing()

    #####################################
    ##### Model specific functions  #####
    #####################################   
    
    #### Defines the dynamics ####    
    ########################################### Change these methods to define your problem ############################################
    
    def Drift(self,state,t_index):
        # X = [z,s,f,v,q]
        Drift = np.zeros_like(state)
        return Drift
    
    def Diffusion(self,state,t_index):
        '''The Diffusion function is responsible of lifting the controlled & noise variables to the whole state for integration with the drift. Hence, it must have the shape (dim_control,dim). 
        WARNING: it is assumed that it is not dependent on the state. If this is not the case, the integration_step function needs to be modified, because in this case the Diffusion would hlso have a 3rd dimension (N_particles,dim_control,dim) and the dot-product will give (N_particles,N_particles,dim)!!
        '''
        return self._aux_mat

    def controller(self,state,t_index):
        h_xt = self.basis_functions(state,t_index)
        u = np.dot(h_xt,self.feedback_term[:,:,t_index].T) + self.openloop_term[:,t_index]
        return u, h_xt
    
    #####################################################################################################################################    
    def integration_step(self,state,noise,t_index, uncontrolled=False):
        #t_index = int(t/self.dt)
        if uncontrolled:
            new_state = state + self.Drift(state,t_index)*self.dt + np.dot(noise.T,self.Diffusion(state,t_index))
            return new_state
        else:
            u , h_xt = self.controller(state,t_index)
            new_state = state + self.Drift(state,t_index)*self.dt + np.dot((u*self.dt + noise.T),self.Diffusion(state,t_index))
            return new_state, u, h_xt 

    #### Defines the State-Cost ####
    ########################################### Change this method to define your problem ############################################
    def obs_signal(self,state,t_index):
        state_obs = state # this is the states on which the signal depends
                          
        return state_obs
    
    def neg_logLikelihood(self,obs_signal, t_index):
        t = t_index*self.dt
        #print "shape of BoldSignal:", obs_signal.shape
        if np.any(np.isclose(t,self.t_obs)):
            #print "shape observation:",self.observations[np.isclose(t,self.t_obs),:].shape
            #print "shape obs_signal:",obs_signal.shape
            negLL_vector=0.5*(obs_signal - self.observations[np.isclose(t,self.t_obs),:])**2/self.var_obs
            return np.sum(negLL_vector, axis=1) #This ensures that the returned results has the correct dimensions
        else:
            #print 'Cost is zero!'
            return np.zeros(obs_signal.shape[0])
    #################################################################################################################################

    #### Defines Basis Functions of Controller ####
    def basis_functions(self,state,t_index):
        '''Estimates the basis functions given the state. The basic implementation assumes linear feedback controller but other basis functions are possible. Becasue of the linear assumption, the number of basis functions is the same as the number of controlled dimensions. If a more general case is considered with k basis functions, the APIS class needs modification accordingly.
        '''
        h_xt = (state[:,0:self.dim_control] - self.mu_control[:,t_index])/self.sigma_control[:,t_index]
        return h_xt
    
    ##################################################################################################################################
    
    ####################################
    ### Functions for learning model ###
    ####################################
    def grad_obsSignal(self,X):
        #ObsError = np.zeros([self.N_particles,self.dim_obs])
        GradObs = np.zeros([self.N_particles,self.dim_obs,self.NumParam_ObsSignal])
        Maux = np.zeros_like(GradObs)
        
        state_obs = X[:,-2:,:] 
        v = state_obs[:,0,:]
        q = state_obs[:,1,:]
        #print "shape of v and q", v.shape, q.shape
        for t_index in np.arange(self.timepoints):
            t = t_index*self.dt
            if np.any(np.isclose(t,self.t_obs)): 
                obs_sig = self.obs_signal(X[:,:,t_index],t_index)
                ObsError = (self.observations[np.isclose(t,self.t_obs),:] - obs_sig)
                GradObs[:,:,0] = obs_sig/self.Vo
                GradObs[:,:,1] = (1.-q[:,t_index]).reshape(GradObs[:,:,1].shape)
                GradObs[:,:,2] = (1.-q[:,t_index]/v[:,t_index]).reshape(GradObs[:,:,1].shape)
                GradObs[:,:,3] = (1.-v[:,t_index]).reshape(GradObs[:,:,1].shape)
                Maux += ObsError[:,:,np.newaxis]*GradObs
        return np.sum(Maux,axis=1)/self.var_obs
    
    def update_obsSignal(self,itr,GradLogL):
        #BOLD Signal
        GradLogL *= self.var_obs
        self.Vo += GradLogL[0]
        self.k1 += GradLogL[1]
        self.k2 += GradLogL[2]
        self.k3 += GradLogL[3]
        
        if self.comm.Get_rank()==0: self.Bold_Params_itrs[:,itr] = np.array([self.Vo,self.k1,self.k2,self.k3])

        #self.Vo = self.comm.bcast(self.Vo,root=0)
        #self.k1 = self.comm.bcast(self.k1,root=0)
        #self.k2 = self.comm.bcast(self.k2,root=0)
        #self.k3 = self.comm.bcast(self.k3,root=0)

    ####################################
    ######## Helper Functions ##########
    ####################################
    ########################################### Change this method to retrieve your data ############################################
    def _load_data(self):
        ''' Loads data from files and defines the variables observations, t_obs, index_obs_dim and eventually var_obs
        Change as needed.
        '''
        ###################### DATA ID ###############################    
        dir_data = 'Data_SmoothingProblem/ToyNet/ToyData_BM_2ROIS.mat'
        #################### Load event times ######################## 
        self.data = sio.loadmat(dir_data) #the data is a directory like object with key 'tobs' for the time points and key time_series for the observed signal. It also contains the event_time, the observation noise (stdev)
        ##################### LOAD TIME-SERIES ###########################
        self.t_obs = self.data['tobs'][0,:] # t_obs must have shape (No. of observ,nr_ROIs)
        ### Observation noise estimated from fMRI time series ###
        self.var_obs = (self.data['std_obs'])**2  # this should be a vector of length nr_ROIs      
        ### fMRI time series ###
        time_series = self.data['time_series'] # time_Series is a nr_ROIsx(# of observ)-array containing all fMRI of one subject. We have a fMRI time series of length (# of observ,nr_ROIs).
        #print 'shape of time_series: ', time_series.shape 
        self.nr_rois = time_series.shape[0]

        try: self.dim_obs = time_series.shape[1] # dimensions of the observed signal
        except : self.dim_obs = 1
              
        self.observations = time_series.T#reshape([time_series.shape[1],self.dim_obs])  # observations must have shape (No. of observ,dim_obs)

        if self.comm.Get_rank()==0:
            print 'Time series: ToyNet', 'in ',dir_data, ' loaded'   
        
        ####################### DON'T CHNAGE!! ####################################################################################
        #print "Shape of observations and t_obs",self.observations.shape,self.t_obs.shape
        assert self.observations.shape[0]==self.t_obs.shape[0], "Number of observations and timestamps mismatch!"
        try : self.dim_obs = self.observations.shape[1] # dimensions of the observed signal
        except : self.dim_obs = 1
        
        if self.comm.Get_rank()==0:
            print 'Number of observations: ', self.observations.shape[0]
            print 'Dimension of observations: ', self.dim_obs
    ##################################################################################################################################
    def _load_model_params(self):
        noise_factor = 1.
        
        self.sigma_dyn = noise_factor*np.identity(self.dim_control)

        
    def _testing(self):
        state = np.random.rand(self.N_particles,self.dim)
        t_index = 0
        Nxdim = (self.N_particles,self.dim)
        Nxdim_control = (self.N_particles,self.dim_control)
        Nxdim_obs = (self.N_particles,self.dim_obs)
        # Test dimensions of Drift
        shape_drift = self.Drift(state,t_index).shape
        assert shape_drift == Nxdim, "Drift has not the correct dimensions "+str(Nxdim)+" but has "+str(shape_drift)
        #Test shape of basis_functions
        shape_basis = self.basis_functions(state,t_index).shape
        assert shape_basis == Nxdim_control, "basis_functions has not the correct dimensions "+str(Nxdim_control)+" but has "+str(shape_basis)
        #Check control dimensions
        u, h_xt = self.controller(state,t_index)
        shape_u = u.shape
        assert shape_u == Nxdim_control, "Control u has not the correct dimensions "+str(Nxdim_control)+" but has "+str(shape_u)
        #Test shape Diffusion
        dim_controlxdim = (self.dim_control,self.dim)
        shape_diffusion = self.Diffusion(state,t_index).shape
        assert shape_diffusion == dim_controlxdim,"Diffusion has not the correct dimensions "+str(dim_controlxdim)+" but has "+str(shape_diffusion)+". If the shape should be (N_particles,dim_control,dim), change integration_step!"
        #Test shape of dot prod: np.dot((u*self.dt + noise),self.Diffusion(state,t_index))
        noise =  np.random.rand(self.N_particles,self.dim_control)
        shape_dotprod = np.dot((u*self.dt + noise),self.Diffusion(state,t_index)).shape
        assert shape_dotprod == Nxdim, "dot(u+noise,Diffusion) has not the correct dimensions "+str(Nxdim)+" but has "+str(shape_dotprod)
        #Test dimension of observation signal and negLog_likelihood
        obsignal = self.obs_signal(state,t_index)
        shape_obsignal = obsignal.shape
        assert shape_obsignal == Nxdim_obs, "obs_signal doesn't return the right dimensions: "+str(Nxdim_obs)+" but has "+str(shape_obsignal)
        negLogL = self.neg_logLikelihood(obsignal,t_index)
        shape_negLogL = negLogL.shape
        assert shape_negLogL==(self.N_particles,),"neg_logLikelihood doesn't return the right dimensions: "+str((self.N_particles,))+" but has "+str(shape_negLogL)
        
    def _get_sigma_prior0(self):
        '''Integrates the uncontrolled system to determine the variance of the stationary distribution of the unontrolled process and returns its square root.
        '''
        self.sigma_prior0 = np.ones((self.dim,))
        #self.sigma_prior0[0] = self.sigma_dyn/np.sqrt(2.*self.A)
        assert self.sigma_prior0.shape == (self.dim,), "sigma_prior0 has the wrong dimensions "+str((self.dim,))+" but has "+str(self.sigma_prior0.shape)
        
    def save_data(self,save_var):
        pass
