import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


NUM_PARTICLES  = 10000
# 1) Загрузим данные и отбросим нестационарную часть
data_eq = pd.read_csv("state.csv")


# 2) Параметры
kB = 1.380649e-23  # Дж/К
m1 = 39.948e-3/6.022e23  # кг, аргон
m2 = 83.798e-3/6.022e23  # кг, криптон

# 3) Температура из полной энергии
data_eq["T_fromE"] = (data_eq["E1"] + data_eq["E2"]) / ( (NUM_PARTICLES)*kB )

# 4) Левые и правые части equipartition:
data_eq["eq1_x"] = 0.5 * m1 * data_eq["vx2_1"]
data_eq["eq1_y"] = 0.5 * m1 * data_eq["vy2_1"]
data_eq["eq2_x"] = 0.5 * m2 * data_eq["vx2_2"]
data_eq["eq2_y"] = 0.5 * m2 * data_eq["vy2_2"]
data_eq["kbT_div2"] = 0.5 * kB * data_eq["temperature"]

# 5) График: сравнение eq1_x, eq1_y и kB*T/2
plt.figure(figsize=(10,6))
plt.plot(data_eq["time"], data_eq["eq1_x"] - 25e-23, label=f"v_x^2")
plt.plot(data_eq["time"], data_eq["eq1_y"] - 25e-23, label=f"v_y^2")
plt.plot(data_eq["time"], data_eq["kbT_div2"], '--', label=f"k_BT")
# plt.plot(data_eq["time"], (data_eq["eq1_x"] + data_eq["eq1_y"]), label=f"sum")
plt.xlabel("time"); plt.ylabel("Energy per dof (J)")
plt.title("Equipartition check, species 1")
plt.legend(); plt.grid()
plt.savefig('6.png')

# Аналогично для species 2
plt.figure(figsize=(10,6))
plt.plot(data_eq["time"], data_eq["eq2_x"] + 50e-23, label=f"v_x^2")
plt.plot(data_eq["time"], data_eq["eq2_y"] + 50e-23, label=f"v_y^2")
plt.plot(data_eq["time"], data_eq["kbT_div2"], '--', label=f"k_BT")
# plt.plot(data_eq["time"], (data_eq["eq2_x"] + data_eq["eq2_y"])/2, label=f"sum")
plt.xlabel("time"); plt.ylabel("Energy per dof (J)")
plt.title("Equipartition check, species 2")
plt.legend(); plt.grid(); 
plt.savefig('7.png')
