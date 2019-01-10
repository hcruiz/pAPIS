import numpy as np
import sys
import os
#sys.path.append("../../")
import scipy.io as sio
#from LQ_ModelParams import *
from SmoothingProblem_template import LQ_Problem
"""
Created on Fri April 21 2017

The class Smoothing_Problem defines the problem for the APIS class

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

class LQ_Grid_Problem(LQ_Problem): 
    
    def __init__(self,comm,meta_params,*args):
        
        #LQ_Problem.__init__(self,comm,meta_params,*args)
        super(LQ_Grid_Problem,self).__init__(comm,meta_params,*args)        
        #####################################
        ######## Control Parameters #########
        #####################################
        
        self.num_steps = 200#4*140
        upper = 5.
        lower = -5.
        step = (upper-lower)/self.num_steps
        self.centers = np.arange(lower,upper+step/10.,step)
        self.num_basisfunc = len(self.centers)
        
        #print self.centers
        self.halfwidth = (self.centers[1]-self.centers[0])/2.
        self.lower_limit = self.centers - self.halfwidth
        self.upper_limit = self.centers + self.halfwidth
        print "Lowest limit of controller:",self.lower_limit[0]
        print "Highest limit of controller:",self.upper_limit[-1]
        self.INDIC_xt = np.zeros([self.N_particles,self.num_basisfunc,self.timepoints],dtype=bool)
        self.feedback_term = np.zeros([self.dim_control,self.num_basisfunc,self.timepoints])
        self.openloop_term = np.zeros([self.dim_control,self.timepoints])
        
        if self.comm.Get_rank()==0: print "GRID STEP:",step, "No. of Basis functions", self.num_basisfunc
        #print "############## HELLO WORLD! ###################"
        self._testing()

        
    def controller(self,state,t_index):
        h_xt = self.basis_functions(state,t_index)
        indicator_x = (state[:,0:self.dim_control] > self.lower_limit)*(state[:,0:self.dim_control] < self.upper_limit)
        #print "SHAPE of indicator_x",indicator_x.shape
        #print "SHAPE of h_xt",h_xt.shape
        lost_particles =  np.logical_not(np.any(indicator_x,axis=1))
        #print "Shape of lost_particles",lost_particles.shape, " any lost?",np.any(lost_particles)
        if np.any(lost_particles):
            #print "There were lost particles!"
            lower_lost = state[:,0:self.dim_control] <= self.lower_limit[0]
            higher_lost = state[:,0:self.dim_control] >= self.upper_limit[-1]
            #print "Sahpe of lower_lost:",lower_lost.shape
            indicator_x[lower_lost[:,0],0] = True
            indicator_x[higher_lost[:,0],-1] = True
            lost_particles_after = np.any(indicator_x,axis=1)
            assert np.any(lost_particles_after), "Something went wrong with the clipping"   
        #print indicator_x
        self.INDIC_xt[:,:,t_index] = indicator_x

        copies_feedback =h_xt*self.feedback_term[:,:,t_index]
        assert copies_feedback.shape==indicator_x.shape, "copies_feedback and indicator_xt must have same form"
        u = copies_feedback[indicator_x] + self.openloop_term[:,t_index] #open-loop term not needad as the signal can be learned in each a_kt
        u = u[:,np.newaxis] # self.A*u[:,np.newaxis] for the fMRI problem
        return u, h_xt
    
    #### Defines Basis Functions of Controller ####
    def basis_functions(self,state,t_index):
        '''Estimates the basis functions given the state. The basic implementation assumes linear feedback controller but other basis functions are possible. Becasue of the linear assumption, the number of basis functions is the same as the number of controlled dimensions. If a more general case is considered with k basis functions, the APIS class needs modification accordingly.
        '''
        h_xt = np.ones_like(state[:,0:self.dim_control])
        return h_xt
    
    def _load_data(self):
        ''' Loads data from files and defines the variables observations, t_obs, index_obs_dim and eventually var_obs
        Change as needed.
        '''
        try:
            ts = int(self.case_identifier) # For multiple time series
            if self.comm.Get_rank()==0: print 'This is case ', ts
        except:
            if self.comm.Get_rank()==0: print "No case specified to load data; this is the standard LQ example"
        
        self.observations = np.array([[0.],[7.5]]) # observations must have shape (No. of observ,dim_obs)
        self.t_obs = np.array([0.,1.]) # t_obs must have shape (No. of observ,)
        self.var_obs = 0.25
        
        ######################################### DON'T CHNAGE!! ###################################################################
        assert self.observations.shape[0]==self.t_obs.shape[0], "Number of observations and timestamps mismatch!"
        try : self.dim_obs = self.observations.shape[1] # dimensions of the observed signal
        except : self.dim_obs = 1
        
        if self.comm.Get_rank()==0:
            print 'Number of observations: ', self.observations.shape[0]
            print 'Dimension of observations: ', self.dim_obs