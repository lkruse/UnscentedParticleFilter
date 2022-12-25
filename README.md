An implementation of the unscented particle filter, as presented in [*Van Der Merwe, R., Doucet, A., De Freitas, N., & Wan, E. (2000). The Unscented Particle Filter. Advances in Neural Information Processing Systems, 13.*](https://proceedings.neurips.cc/paper/2000/hash/f5c3dd7514bf620a1b85450d2ae374b1-Abstract.html) The UPF algorithm uses an unscented Kalman filter to generate the importance proposal distributions for the particle set. The UKF allows the particle filter to incorporate the latest observation in the update routine. 

The UPF algorithm is tested on two state estimation tasks: a synthetic scalar estimation task and a nonholonomic robotic motion estimation task. Each experiment can be run in its corresponding notebook.

<p align="center">
  <img src="/figs/synthetic_time_series.png" width="400" /> 
  <img src="/figs/nonholonomic_robot.png" width="400" />
</p>

The UKF code is inspired by the code in [Chapter 19](/https://algorithmsbook.com/files/chapter-19.pdf) of *Kochenderfer, M. J., 
Wheeler, T. A., & Wray, K. H. (2022). Algorithms for Decision Making. MIT Press.*
