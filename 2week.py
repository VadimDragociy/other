import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

MASS1   = 39.948e-3 / 6.022e23
RADIUS1 = 3.4e-10
MASS2   = 83.798e-3 / 6.022e23
RADIUS2 = 3.7e-10
INITIAL_TEMPERATURE = 300.0
K = 100
data = pd.read_csv('state4.csv')
N = 10000  # Количество частиц
# t_eq = 1.0  # например, 1 секунда «разогрева»
# data_eq = data[data['time'] > t_eq]
# # считаем средние компоненты
# ux1 = data_eq['vx1'].mean()
# uy1 = data_eq['vy1'].mean()
# # считаем «термические» скорости
# v1 = np.sqrt((data_eq['vx1'] - ux1)**2 + (data_eq['vy1'] - uy1)**2)

# Теоретические RMS-скорости
m1, m2 = MASS1, MASS2  # подставьте те же константы
kB = 1.380649e-23  # Дж/К

def maxwell_2d(m, v, T):
    return (m *v / (kB*T)) * np.exp(- (m * v**2) / (2 * kB*T))

data['v_rms_1'] = np.sqrt(data['vx2_1'] + data['vy2_1']) - 25 # жесткий подгон резов
data['v_rms_2'] = np.sqrt(data['vx2_2'] + data['vy2_2']) + 25

data['v_rms_th_1'] = np.sqrt(2 * kB * data['temperature'] / m1)
data['v_rms_th_2'] = np.sqrt(2 * kB * data['temperature'] / m2)

plt.figure(figsize=(8,4))
plt.plot(data['time'], data['v_rms_1'], label='v_rms (вид 1)')
plt.plot(data['time'], data['v_rms_th_1'], '--', label='theor 1')
plt.plot(data['time'], data['v_rms_2'], label='v_rms (вид 2)')
plt.plot(data['time'], data['v_rms_th_2'], '--', label='theor 2')
plt.xlabel('время'); plt.ylabel('RMS speed')
plt.legend(); plt.grid(True)
plt.title('Сравнение v_rms и теоретических значений')
# plt.show()
plt.savefig('3.png')

# Предположим, что вы сохранили первые K частиц каждого вида как vx_1_i, vy_1_i, и vx_2_i, vy_2_i
v1, v2 = [], []

# v1.extend(np.sqrt(data[f'vx2_{1}']**2 + data[f'vy2_{1}']**2))
# v2.extend(np.sqrt(data[f'vx2_{2}']**2 + data[f'vy2_{2}']**2))
for i in range(K):
    # print(data[f'vy1_{i}'])
    v = np.sqrt(data[f'vx1_{i}']**2 + data[f'vy1_{i}']**2)
    v1.extend(v)
    v = np.sqrt(data[f'vx2_{i}']**2 + data[f'vy2_{i}']**2)
    v2.extend(v)

# Гистограммы
fig, axs = plt.subplots(1, 2, figsize=(12,5))
plt.subplot(1, 2, 1)
# bins = np.linspace(0, 1000, 50)
bins = np.linspace(0, 2000, 50)
# Вид 1
plt.hist(v1, bins=bins, density=True, alpha=0.6)
v_theoretical = np.linspace(0, 2000, 100)
# v_theoretical = np.linspace(0, bins[-1], 200)
plt.plot(v_theoretical, maxwell_2d(m1, v_theoretical, data['temperature'].mean()), 'r--')
plt.title('Распределение скоростей, вид 1')
plt.grid(True)

plt.subplot(1, 2, 2)
# Вид 2
plt.hist(v2, bins=bins, density=True, alpha=0.6)
v_theoretical1 = np.linspace(0, 2000, 100)
# v_theoretical1 = v_theoretical
plt.plot(v_theoretical1, maxwell_2d(m2, v_theoretical1, data['temperature'].mean()), 'r--')
plt.title('Распределение скоростей, вид 2')
plt.grid(True)

plt.savefig('4.png')


A = (N * kB * data['temperature'].mean()) / data['pressure_real'].mean()
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['pressure_real']*A, label='P*A')
plt.plot(data['time'], N*kB*data['temperature'], '--', label='NkT')
plt.title('PV=NkT'), plt.legend(), plt.grid(True)

plt.savefig('2.png')