import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN INICIAL DE PARÁMETROS
# ==========================================
fs = 1000          # Frecuencia de muestreo (Hz)
t = np.linspace(-1, 1, 2000, endpoint=False) # Eje de tiempo (-1s a 1s)
N = len(t)         # Número total de muestras
freq = np.fft.fftfreq(N, 1/fs) # Eje de frecuencias en Hz
freq_shifted = np.fft.fftshift(freq) # Frecuencias centradas en 0 Hz

# ==========================================
# 2. GENERACIÓN DE SEÑALES EN EL TIEMPO
# ==========================================
# Señal 1: Senoidal pura de 10 Hz
f1 = 10
x_seno = np.sin(2 * np.pi * f1 * t)

# Señal 2: Pulso Rectangular (ancho de 0.2 segundos)
x_pulso = np.where(np.abs(t) <= 0.1, 1.0, 0.0)

# ==========================================
# 3. CÁLCULO DE LA TRANSFORMADA DE FOURIER (FFT)
# ==========================================
# Transformada para la señal senoidal
X_seno_fft = np.fft.fftshift(np.fft.fft(x_seno)) / N
mag_seno = np.abs(X_seno_fft)
fase_seno = np.angle(X_seno_fft)

# Transformada para el pulso rectangular
X_pulso_fft = np.fft.fftshift(np.fft.fft(x_pulso)) / N
mag_pulso = np.abs(X_pulso_fft)
fase_pulso = np.angle(X_pulso_fft)

# ==========================================
# 4. DEMOSTRACIÓN DE LA PROPIEDAD DE LINEALIDAD
# x_suma = a*x1 + b*x2  =>  FFT(x_suma) = a*FFT(x1) + b*FFT(x2)
# ==========================================
a, b = 2.0, 0.5
x_combinada = a * x_seno + b * x_pulso
X_combinada_fft = np.fft.fftshift(np.fft.fft(x_combinada)) / N
mag_combinada = np.abs(X_combinada_fft)

# ==========================================
# 5. VISUALIZACIÓN Y GRAFICACIÓN DE RESULTADOS
# ==========================================
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Análisis de Señales mediante la Transformada de Fourier', fontsize=14, fontweight='bold')

# --- Graficar Señal Senoidal ---
axs[0, 0].plot(t, x_seno, color='navy')
axs[0, 0].set_title('Señal Senoidal (10 Hz) - Dominio del Tiempo')
axs[0, 0].set_xlabel('Tiempo [s]'); axs[0, 0].set_ylabel('Amplitud'); axs[0, 0].grid(True)

axs[0, 1].plot(freq_shifted, mag_seno, color='crimson')
axs[0, 1].set_title('Espectro de Magnitud (Senoidal) - Dominio Frecuencia')
axs[0, 1].set_xlim([-30, 30]); axs[0, 1].set_xlabel('Frecuencia [Hz]'); axs[0, 1].grid(True)

# --- Graficar Pulso Rectangular ---
axs[1, 0].plot(t, x_pulso, color='darkgreen')
axs[1, 0].set_title('Pulso Rectangular - Dominio del Tiempo')
axs[1, 0].set_xlabel('Tiempo [s]'); axs[1, 0].set_ylabel('Amplitud'); axs[1, 0].grid(True)

axs[1, 1].plot(freq_shifted, mag_pulso, color='darkorange')
axs[1, 1].set_title('Espectro de Magnitud (Sinc) - Dominio Frecuencia')
axs[1, 1].set_xlim([-50, 50]); axs[1, 1].set_xlabel('Frecuencia [Hz]'); axs[1, 1].grid(True)

# --- Graficar Linealidad (Señal Combinada) ---
axs[2, 0].plot(t, x_combinada, color='purple')
axs[2, 0].set_title('Señal Combinada (2*Seno + 0.5*Pulso) - Tiempo')
axs[2, 0].set_xlabel('Tiempo [s]'); axs[2, 0].set_ylabel('Amplitud'); axs[2, 0].grid(True)

axs[2, 1].plot(freq_shifted, mag_combinada, color='magenta')
axs[2, 1].set_title('Espectro Combinado (Verificación de Linealidad)')
axs[2, 1].set_xlim([-50, 50]); axs[2, 1].set_xlabel('Frecuencia [Hz]'); axs[2, 1].grid(True)

plt.tight_layout()
plt.show()