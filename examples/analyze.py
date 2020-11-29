from gibbsduhem import Read
import matplotlib.pyplot as plt

output = Read("output.dat")
temp = output.find("T")
press = output.find("p")

plt.plot(temp, press)
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure [kPa]")
plt.grid()
plt.show()
