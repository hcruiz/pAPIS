# pAPIS: Parallel Adaptive Path Integral Sampler
The [Adaptive Path-Integral Smoother/Sampler (APIS)](https://arxiv.org/abs/1605.00278) is a latent process sampler targeting the posterior distribution over a continuous-time state-space model (SSM) given a cost landscape, e.g. as end-cost or as a time series. The algorithm is based on the path integral control theory. For more detail on the theory see [here](https://arxiv.org/abs/physics/0505066). This repo has a parallel implementation of APIS using mpi4py. 

In what follows, I will give a short description of the algorithm and the modules. This is description is work in progress, so if you have questions or feedback I will very much appreciate if you drop me a line.

## Modules
There are three principal modules to work with. The *main* module [Main_template.py](https://github.com/hcruiz/pAPIS/blob/master/Code/Main_template.py) is the top-level module that you run to learn a controller and sample from the posterior adaptively. 

The main script instantiates two objects, the smoothing problem defining the SSM and the APIS object, which performs the sampling and control estimation steps. In addition, the main script also initializes the MPI.COMM_WORLD object for spawning jobs on different cores.
