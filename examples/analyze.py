from gibbsduhem.analyze import Read
import matplotlib.pyplot as plt

analyze = Read("Blk.prp")
density = analyze.find("DENSITY")
nummol = analyze.find("TOT_MOL")

plt.plot(density)
plt.xlabel("Step")
plt.ylabel("Density [g/cm^3]")
plt.show()
