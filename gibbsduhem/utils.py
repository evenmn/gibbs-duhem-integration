import subprocess
import numpy as np


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
