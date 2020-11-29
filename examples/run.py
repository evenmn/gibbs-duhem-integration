from gibbsduhem import GibbsDuhem
from lammps_simulator.computer import CPU

rho1 = 0.97
rho2 = 0.00022
T_init = 370
p_init = 0.36
p_final = 1.01325

gibbsduhem = GibbsDuhem(T_init, p_init)
gibbsduhem.set_box1(rho1, 6, 6, 6)
gibbsduhem.set_box2(rho2, 6, 6, 6)
gibbsduhem.run(computer=CPU(num_procs=4, lmp_exec="lmp_mpi"), dbeta=1e-5, write="output.dat")
