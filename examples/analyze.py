from gibbsduhem import Read
import matplotlib.pyplot as plt

filenames = ["gibbs-duhem_7/output.dat", "gibbs-duhem_9/output.dat", "gibbs-duhem_10/output.dat"]
labels = ["dT=1.0", "dT=0.5", "dT=0.1"]

for filename, label in zip(filenames, labels):
    output = Read(filename)
    temp = output.find("T")
    press = output.find("p")
    rhoA = output.find("rhoA")
    rhoB = output.find("rhoB")

    plt.plot(temp, press, label=label)

p_gemc = [0.32, 0.37, 0.47, 0.56, 0.75, 0.80, 0.98]
T_gemc = [370, 375, 380, 385, 390, 395, 400]

p_vega = [0.316, 0.689]
T_vega = [370, 390]

# plt.plot(T_gemc, p_gemc, 'or', label="GEMC")
# plt.plot(T_vega, p_vega, 'ob', label="Vega,Abascal(2006)")
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure [kPa]")
plt.grid()
plt.legend(loc='best')
plt.show()

stop

plt.plot(temp, rhoB)
plt.xlabel("Temperature [K]")
plt.ylabel("Density [g/cm^3]")
plt.grid()
plt.show()
