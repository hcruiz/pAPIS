import sys
import os
#sys.path.append("../../")
import numpy as np
from mpi4py import MPI
import scipy.io as sio
from APIS import APIS
"""
Created on 14 June 2017

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

class APIS_Grid(APIS):
    
    def __init__(self,smproblem):
        super(APIS_Grid,self).__init__(smproblem)
        self.num_basisfunc = self.smp.num_basisfunc
    ##############################    
    ##### Train Controller #######
    ##############################
    def update_controller(self):
        #if self.rank==0:print "Updating Control Parameters"
        self._update_control_params()
        
    def _update_control_params(self):
        for t in np.arange(self.smp.INDIC_xt.shape[-1]):
            x_indicator = self.smp.INDIC_xt[:,:,t]
            self._get_nncorrelations(x_indicator,t) # Only root=0 has the correlations!!
            if self.rank==0: 
                self.smp.feedback_term[:,:,t] += self.learning_rate*self.nw_hdW_grid/self.dt #this is the global normalized weighted averaged with the statistics from the particles in each controll cell
        self.smp.feedback_term = self.comm.bcast(self.smp.feedback_term,root=0)
        
    def _get_nncorrelations(self,x_indicator,t_index):
        '''
        - hdW_grid is a 3D-grid [Nparticles,Nbasisfunc,T] containing for each control cell [(x,t)-grid] the noise components of the particles in that cell.
        -  Weight_grid is a 3D-grid [Nparticles,Nbasisfunc,T] containing the un-normalized weights of all particles in each cell
        - The partilcles missing in each control cell have the corresponding value 0. for both the weights and the noise components
        '''
        # Get weights as 3D-grid
        self._get_Weight_grid(x_indicator) 
        self.norm_grid = self._nnweighted_sum(np.ones_like(self.Weight_grid))
        #self.test_normalization()
        
        # Get noise realizations as a 3D-grid
        hdW_grid = self._get_hdW_grid(x_indicator,t_index)
        self.nnw_hdW_grid = self._nnweighted_sum(hdW_grid)
        
        if self.rank==0: 
            #self.nw_hdW_grid = np.zeros([self.dim_control,self.num_basisfunc,self.timepoints])
            #print "Shape INDIC_xt",self.smp.INDIC_xt.shape
            #print self.norm_grid
            self.norm_grid[self.norm_grid==0.] = 1.
            #print self.norm_grid
            self.nw_hdW_grid = self.nnw_hdW_grid/self.norm_grid
            #print "Shape of self.nw_hdW_grid",self.nw_hdW_grid.shape
            #print "Shape of self.norm_grid", self.norm_grid.shape

    def _nnweighted_sum(self,x_local):
        #print x_local.shape,"x_local shape", self.norm_local_gridweights.shape, "shape of local_gridweights"
        assert x_local.shape==self.Weight_grid.shape, "Dimension mismatch!"
        
        ax = np.arange(len(x_local.shape))
        if len(x_local.shape) > 2: ax[0],ax[-2] = ax[-2],ax[0] 
        #print x_local.transpose(*ax).shape,"x_local transpose"
        wx_local = np.sum(self.Weight_grid*x_local,axis=0)
        #print "Shape wx_local:",wx_local.shape
        all_summands = np.array(self.comm.gather(wx_local,root=0))
        if self.rank==0: 
            #print "shape of all summands:",all_summands.shape
            wx_global = all_summands.sum(axis=0)
        else: wx_global = "Only root has wx_global!"
        return wx_global
        
    def _get_Weight_grid(self,x_indicator):
        #print "Shape x_indicator",x_indicator.shape
        self.Weight_grid = np.zeros(x_indicator.shape)
        aux_grid = np.zeros_like(self.Weight_grid)
        aux_grid[x_indicator] = 1.
        localS_grid = aux_grid*self.local_Sparticles[:,np.newaxis]
        #print "Shape of localS_grid:",localS_grid.shape
        self.Weight_grid[x_indicator] = np.exp(-localS_grid[x_indicator])
        #print "Shape of Weight_grid",self.Weight_grid.shape
        
    def test_normalization(self):
        #print "Shape of norm_grid", self.norm_grid.shape
        #print "Shape of Weight_grid",self.Weight_grid.shape
        normalized_localweights_grid = self.Weight_grid/self.norm_grid
        #print "Shape of normalized_localweights_grid:",normalized_localweights_grid.shape
        sum_local = np.sum(normalized_localweights_grid,axis=0)
        all_summands = np.array(self.comm.gather(sum_local,root=0))
        if self.rank==0: 
            #print "shape of all summands:",all_summands.shape
            global_sum_normalized_weights = all_summands.sum(axis=0)
            #print global_sum_normalized_weights
            #print np.sum(global_sum_normalized_weights,axis=0)
            
    def _get_hdW_grid(self,x_indicator,t_index):
        hdW_grid = np.zeros(x_indicator.shape)
        aux_grid = np.zeros_like(hdW_grid)
        aux_grid[x_indicator] = 1.
        #print self.Noise[:,:,t_index].shape, x_indicator.shape
        hdW_grid = aux_grid*np.swapaxes(self.Noise[:,:,t_index],0,1)
        #print hdW_grid.shape
        return hdW_grid
        