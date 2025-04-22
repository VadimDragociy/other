import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

# Загрузка данных
data = pd.read_csv('state.csv')

# Константы
k = 1.0  # Постоянная Больцмана (в условных единицах)
m = 1.0  # Масса частицы
N = 100000  # Количество частиц
BOX_SIZE = 40.0  # Размер контейнера (для расчета площади)

# Расчет площади A
A = (N * k * data['temperature'].mean()) / data['pressure_real'].mean()

# Вычисления для проверок
data['avg_K'] = data['kinetic_energy'] / N
data['K_theoretical'] = k * data['temperature']

data['v_rms'] = np.sqrt(data['vx2_mean'] + data['vy2_mean'])  # v_rms = sqrt(vx² + vy²)
data['v_rms_theoretical'] = np.sqrt(2 * k * data['temperature'] / m)

data['mean_speed_theoretical'] = np.sqrt(5 * data['temperature'] / np.pi)

data['sigma_v_squared'] = data['v_rms']**2 - (data['mean_speed']**2)
data['sigma_theoretical'] = data['temperature'] / 2

# Графики в сетке
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(data['time'], data['kinetic_energy'], color='blue')
plt.title('Сохранение энергии'), plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(data['time'], data['avg_K'], label='<K>')
plt.plot(data['time'], data['K_theoretical'], '--', label='kT')
plt.title('⟨K⟩ = kT'), plt.legend(), plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(data['time'], data['pressure_real']*A, label='P*A')
plt.plot(data['time'], N*k*data['temperature'], '--', label='NkT')
plt.title('PV=NkT'), plt.legend(), plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(data['time'], data['sigma_v_squared'], label='σ²')
plt.plot(data['time'], data['sigma_theoretical'], '--', label='kT/m')
plt.title('Дисперсия'), plt.legend(), plt.grid(True)
plt.tight_layout()
plt.savefig('2.png')


def maxwell_2d(v, T):
    return (v / (k*T)) * np.exp(- (m * v**2) / (2 * k*T))

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
# Распределение Максвелла
v = []
for i in range(100):  # Предположим, что первые 100 частиц сохранены в CSV
    v_i = np.sqrt(data[f'vx_{i}']**2 + data[f'vy_{i}']**2)
    v.extend(v_i)
counts, bins, _ = plt.hist(v, bins=50, density=True, alpha=0.6)
v_theoretical = np.linspace(0, max(v), 100)
plt.plot(v_theoretical, maxwell_2d(v_theoretical, data['temperature'].mean()), 'r--')
plt.title('Распределение скоростей'), plt.grid(True)

plt.subplot(2, 2, 2)
# Изотропия
plt.plot(data['time'], data['vx2_mean'], label='<v_x²>')
plt.plot(data['time'], data['vy2_mean'], label='<v_y²>')
plt.title('Изотропия'), plt.legend(), plt.grid(True)

plt.subplot(2, 2, 3)
# Скорости
plt.plot(data['time'], data['mean_speed'], label='<v>')
plt.plot(data['time'], data['mean_speed_theoretical'], '--', label='Теоретическая')
plt.title('Средняя скорость'), plt.legend(), plt.grid(True)

particle_energy = 0.5 * m * (data['vx_10']**2 + data['vy_10']**2)

plt.subplot(2, 2, 4)
# Энергия частицы
plt.plot(data['time'], particle_energy, label=f'Энергия частицы {10}')
plt.axhline(data['avg_K'].mean(), color='red', linestyle='--')
plt.title('Энергия частицы'), plt.legend(), plt.grid(True)
plt.tight_layout()
plt.savefig('1.png')

plt.figure(figsize=(8, 4))
plt.plot(data['time'], data['pressure_ideal'], label='P_ideal (теоретическое)')
plt.plot(data['time'], data['pressure_real'], '--', label='P_real (экспериментальное)')
plt.title('Сравнение давлений'), plt.xlabel('Время'), plt.ylabel('Давление'), plt.legend(), plt.grid(True)
plt.savefig('pressure_comparison.png')

# График разности давлений
plt.figure(figsize=(8, 4))
plt.plot(data['time'], data['pressure_ideal'] - data['pressure_real'], color='purple')
plt.title('Разность давлений (P_ideal - P_real)'), plt.grid(True)
plt.savefig('pressure_diff.png')