from gibbsduhem import GibbsDuhem
from lammps_simulator.computer import CPU

rho1 = 0.97
rho2 = 0.00022
T_init = 370
p_init = 0.36
p_final = 1.01325

gibbsduhem = GibbsDuhem(T_init, p_init)
gibbsduhem.set_box_A(rho1, 6, 6, 6)
gibbsduhem.set_box_B(rho2, 6, 6, 6)

computer = CPU(num_procs=30, lmp_exec="lmp_mpi")
gibbsduhem.run(computer=computer, dT=0.1, write="output.dat")
