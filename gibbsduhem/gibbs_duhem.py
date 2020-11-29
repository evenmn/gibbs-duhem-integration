import os
import time
import subprocess
import numpy as np
import pandas as pd
from io import StringIO
from lammps_logfile import File
from lammps_analyzer import average
from lammps_simulator import Simulator
from lammps_simulator.computer import CPU


def write_water(charge, volume=None, massdensity=None,
                numberdensity=None, filename="water.lammps"):
    """Write lammps data file for a single water molecule
    """
    assert [numberdensity, massdensity, volume].count(None) == 2, \
        "Either volume or density has to be given"

    if numberdensity is not None:
        volume = 1 / numberdensity
    if massdensity is not None:
        molmass = 18.01528
        volume = molmass / massdensity

    cell_length = volume**(1/3)
    assert cell_length > 2.1, "unit cell needs a length > 2.1 Å"

    with open(filename, 'w') as f:
        f.write("LAMMPS 'data.' description\n\n")
        f.write("        3 atoms\n")
        f.write("        2 bonds\n")
        f.write("        1 angles\n\n")
        f.write("        2 atom types\n")
        f.write("        1 bond types\n")
        f.write("        1 angle types\n\n")
        f.write(f"     0.0 {cell_length:>4.1f}     xlo xhi\n")
        f.write(f"     0.0 {cell_length:>4.1f}     ylo yhi\n")
        f.write(f"     0.0 {cell_length:>4.1f}     zlo zhi\n\n")
        f.write("Atoms\n\n")
        f.write(f"     1    1   2  {charge[2]:>7.4f}   1.55000    1.55000    1.50000\n")
        f.write(f"     2    1   1  {charge[1]:>7.4f}   1.55000    2.30695    2.08588\n")
        f.write(f"     3    1   1  {charge[1]:>7.4f}   1.55000    0.79305    2.08588\n\n")
        f.write("Bonds\n\n")
        f.write("      1   1     1     2\n")
        f.write("      2   1     1     3\n\n")
        f.write("Angles\n\n")
        f.write("      1   1     2     1     3\n")


def check_squeue(user):
    """Returns the job IDs of user
    """
    output = subprocess.check_output(['squeue', '-u', user])
    stuff = output.split()
    stuff = stuff[8:]
    job_ids = np.asarray(stuff[::8], dtype=int)
    return job_ids


def latest_job_id(user):
    """Return job ID of the previously submitted job
    """
    output = subprocess.check_output(['squeue', '-u', user])
    data = output.split()
    data = data[8:]
    job_ids = np.asarray(data[::8], dtype=int)
    return np.max(job_ids)


class ClapeyronEquation:
    """Clapeyron equation,

                              dh
    f(beta, p; dh, dv) = - ---------
                           beta p dv
    """
    def __init__(self, dh, dv):
        self.dh, self.dv = dh, dv

    def __call__(self, beta, p):
        return - self.dh / (beta * p * self.dv)


class GibbsDuhem:
    """Gibbs Duhem integration to find boiling point and
    vaporization enthalpy for a certain pressure

    :param N1: number of molecules in box 1
    :type N1: int
    :param N2: number of molecules in box 2
    :type N2: int
    :param rho1: mass density of box 1, given in g/cm^3
    :type rho1: float
    :param rho2: mass density of box 2, given in g/cm^3
    :type rho2: float
    :param beta_init: initial beta-value, beta = 1/kT
    :type beta_init: float
    :param p_init: initial pressure value given in bar
    :type p_init: float
    :param p_final: final pressure value
    :type p_final: float
    """

    def __init__(self, T_init, p_init, wd="gibbs-duhem", overwrite=False):
        wd_prop = wd    # proposed working directory
        if overwrite:
            try:
                os.makedirs(wd)
            except FileExistsError:
                pass
        else:
            ext = 0
            repeat = True
            while repeat:
                try:
                    os.makedirs(wd_prop)
                    repeat = False
                except FileExistsError:
                    ext += 1
                    wd_prop = wd + f"_{ext}"
        os.chdir(wd_prop)

        self.beta = 1/T_init
        self.p = p_init
        self.rho1 = None
        self.rho2 = None
        self.n1 = None
        self.n2 = None
        self.restart1 = None
        self.restart2 = None

    def set_box1(self, rho, nx, ny, nz):
        """Set system box 1
        """
        self.rho1 = rho
        self.n1 = (nx, ny, nz)

    def set_box2(self, rho, nx, ny, nz):
        """Set system box 2
        """
        self.rho2 = rho
        self.n2 = (nx, ny, nz)

    def _run_init(self, rho, n, box_id):
        """Run simulation to initialize/equilibrate system
        """

        datafile = "H2O.molecule"
        paramfile = "H2O.TIP4P"
        restartfile = "restart.bin"

        var = {'press': self.p,
               'temp': 1/self.beta,
               'seed': 68885,
               'datafile': datafile,
               'paramfile': paramfile,
               'restartfile': restartfile,
               'nx': n[0],
               'ny': n[1],
               'nz': n[2]}

        charge = {2: -1.1128, 1: 0.5564}
        write_water(charge, massdensity=rho, filename=datafile)

        # get LAMMPS script
        this_dir, this_filename = os.path.split(__file__)
        lammps_script = os.path.join(this_dir, "../lammps/tip4p/gibbsduhem/in.init")
        paramfile = os.path.join(this_dir, "../lammps/tip4p/gibbsduhem/H2O.TIP4P")

        # run system
        sim = Simulator(directory=f"init_{box_id}", overwrite=False)
        wd = sim.wd
        sim.copy_to_wd(datafile, paramfile)
        sim.set_input_script(lammps_script, **var)
        sim.run(self.computer)

        if box_id == 1:
            self.restart1 = wd + restartfile
        else:
            self.restart2 = wd + restartfile

    def _run_npt(self, restart, box_id):
        """Run NPT simulation in LAMMPS to find equilibration enthalpy
        and value

        :param rho: initial density
        :type rho: float
        :param N: number of molecules in simulation
        :type N: int
        :param p: pressure in simulation
        :type p: float
        :param T: temperature in system
        :type T: float
        """

        paramfile = 'H2O.TIP4P'
        restartfileout = "restart.bin"

        var = {'press': self.p,
               'temp': 1/self.beta,
               'paramfile': paramfile,
               'restartfilein': '../' + restart,
               'restartfileout': restartfileout}

        # get LAMMPS script
        this_dir, this_filename = os.path.split(__file__)
        lammps_script = os.path.join(this_dir, "../lammps/tip4p/gibbsduhem/in.restart")
        paramfile = os.path.join(this_dir, "../lammps/tip4p/gibbsduhem/H2O.TIP4P")

        # run system
        sim = Simulator(directory=f"iter_{box_id}", overwrite=False)
        wd = sim.wd
        sim.copy_to_wd(paramfile)
        sim.set_input_script(lammps_script, **var)
        sim.run(self.computer)

        if box_id == 1:
            self.restart1 = wd + "/" + restartfileout
        else:
            self.restart2 = wd + "/" + restartfileout

        if self.computer.slurm:
            job_id = latest_job_id(self.user)

            # wait for the job to finish
            job_done = False
            while not job_done:
                current_job_ids = check_squeue(self.user)
                if job_id not in current_job_ids:
                    job_done = True
                else:
                    time.sleep(5)

        # analyze simulations
        logger = File(wd + "/log.lammps")
        volume = logger.get("Volume", run_num=1)
        enthalpy = logger.get("Enthalpy", run_num=1)
        density = logger.get("Density", run_num=1)
        atoms = logger.get("Atoms", run_num=1)
        molecules = atoms[-1] // 3

        volume_avg = average(volume, 10)
        enthalpy_avg = average(enthalpy, 10)
        density_avg = average(density, 10)

        molar_volume = volume_avg[-1] / molecules
        molar_enthalpy = enthalpy_avg[-1] / molecules

        return molar_volume, molar_enthalpy, density_avg[-1]

    def _compute_f(self, p, dh, dv):
        """Compute f=-dh/(beta p dv) using the Gibbs-Duhem equation

        :param p: pressure
        :type p: float
        :param dh: enthalpy difference
        :type dh: float
        :param dv: volume difference
        :type dv: float
        :returns: fugacity
        :rtype: float
        """
        conversion_factor = 0.0039687835
        return conversion_factor * dh / (self.beta * p * dv)

    def _predict_p(self, f):
        """Predict pressure based on one fugacity value f

        :param f: fugacity
        :type f: float
        :returns: predicted pressure
        :rtype: float
        """
        return self.p * np.exp(self.dbeta * f)

    def _correct_p(self, f0, f1):
        """Compute pressure based on two fugacity values f1 and f2

        :param f0: fugacity at previous timestep
        :type f0: float
        :param f1: fugacity at current timestep
        :type f1: float
        :returns: corrected pressure
        :rtype: float
        """
        return self.p * np.exp(self.dbeta * (f0 + f1) / 2)

    def run(self, maxiter=100, dbeta=0.001, computer=CPU(num_procs=4),
            write=False, user="", p_final=1.01325):
        """Run Gibbs-Duhem integration.

        :param maxiter: max number of allowed interations
        :type maxiter: int
        :param dbeta: beta-step for each iteration
        :type dbeta: float
        """

        # declare global parameters
        self.dbeta = dbeta
        self.computer = computer
        self.user = user

        # declare empty list to be filled up
        p_list = []
        T_list = []
        rho1_list = []
        rho2_list = []
        V_vap = []
        H_vap = []

        # run equilibration simulations
        self._run_init(self.rho1, self.n1, 1)
        self._run_init(self.rho2, self.n2, 2)

        if write:
            # write first line of output file
            f =  open(write, 'w')
            f.write("p T dv dh \n")

        converged = False
        iter = 0
        while iter < maxiter and not converged:
            v1, h1, rho1 = self._run_npt(self.restart1, 1)
            v2, h2, rho2 = self._run_npt(self.restart2, 2)
            Dv = v2 - v1
            Dh = h2 - h1

            print(f"Volume difference is {Dv} Å³/mol")
            print(f"Enthalpy difference is {Dh} kcal/mol")

            p_list.append(self.p)
            T_list.append(1/self.beta)
            rho1_list.append(rho1)
            rho2_list.append(rho2)
            V_vap.append(Dv)
            H_vap.append(Dh)

            if write:
                f.write(f"{self.p} {1/self.beta} {Dv} {Dh} \n")

            # from integrators import PredictorCorrector
            # integrator = PredictorCorrector(self.dbeta, ClapeyronEquation(Dh, Dv))
            # self.beta, lnp = integrator.find_next(self.beta, np.log(self.p))
            # self.p = np.exp(lnp)

            f0 = self._compute_f(self.p, Dv, Dh)
            p_pred = self._predict_p(f0)
            print(f"p_pred: {p_pred} atm")
            self.beta += self.dbeta
            print(f"Temp: {1/self.beta} K")
            f1 = self._compute_f(p_pred, Dv, Dh)
            if self.p > p_final:
                converged = True
            self.p = self._correct_p(f0, f1)
            iter += 1

        if write:
            f.close()
        return p_list, T_list, V_vap, H_vap


class Read:
    def __init__(self, filename):
        if hasattr(filename, "read"):
            output = filename
        else:
            output = open(filename, 'r')
        self.read_file(output)

    def read_file(self, fileobj):
        """Read Gibbs-Duhem output file
        """
        string = fileobj.readline()  # string should start with the kws
        self.keywords = string.split()
        contents = fileobj.read()
        self.contents = pd.read_table(StringIO(string + contents), sep=r'\s+')

    def find(self, entry_name):
        return np.asarray(self.contents[entry_name])

    def get_keywords(self):
        """Return list of available data columns in the log file."""
        print(", ".join(self.keywords))


if __name__ == "__main__":
    rho1 = 0.97
    rho2 = 0.00022
    T_init = 370
    p_init = 0.36
    p_final = 1.01325

    gibbsduhem = GibbsDuhem(T_init, p_init)
    gibbsduhem.set_box1(rho1, 6, 6, 6)
    gibbsduhem.set_box2(rho2, 6, 6, 6)
    p, T, V, H = gibbsduhem.run(computer=CPU(num_procs=36, lmp_exec="lmp_mpi"), dbeta=1e-5, write="output.dat")
