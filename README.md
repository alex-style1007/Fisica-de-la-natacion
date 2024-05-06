# Importamos las librerias necesarias

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
Declaramos las variables necesarias

```python
n_nadadores=100

masas = np.random.randint(50, 131, size=n_nadadores)
porcentaje_muscular = np.random.randint(60, 86, size=n_nadadores)

A = 0.5  # Área de la sección transversal del nadador en m^2
rho = 1000  # Densidad del agua en kg/m^3
g = 9.81  # Aceleración gravitacional en m/s^2
v0 = 0  # Velocidad inicial en m/s
t0 = 0  # Tiempo inicial en s
tf = 20  # Tiempo final en s
dt = 0.01  # Incremento de tiempo en s
Cd = 0.8  # Coeficiente de arrastre del nadador

# Dimensiones de la piscina olímpica
longitud_piscina = 50  # Longitud en metros
ancho_piscina = 25  # Ancho en metros
```

### DECLARACION DE FUNCIÓNES NECESARIAS PARA EL ANALISIS

```python
# Función para calcular la eficiencia del nadador en función del % de masa muscular
def calcular_eficiencia(porcentaje_muscular):
    # Supongamos una relación lineal entre el % de masa muscular y la eficiencia,
    # donde 100% de masa muscular corresponde a una eficiencia de 1 y 0% de masa muscular a una eficiencia de 0.5
    eficiencia = 0.5 + 0.5 * (porcentaje_muscular / 100)
    return eficiencia

# Coeficientes de eficiencia para cada nadador
eficiencia = {masas[i]: calcular_eficiencia(porcentaje_muscular[i]) for i in range(len(masas))}

# Función para calcular la resistencia del agua en los brazos
def resistencia_agua(angulo_rotacion, velocidad, masa):
    # Devuelve una energía nula en todo momento
    return 0
# Función para simular el movimiento de natación y calcular el trabajo y la energía cinética de los brazos
def swim(masa):
    t = np.arange(t0, tf, dt)
    v = np.zeros_like(t)
    v[0] = v0
    work = 0
    arm_rotation = np.linspace(0, 2*np.pi, len(t))  # Ángulo de rotación de los brazos
    arm_kinetic_energy = np.zeros_like(t)  # Inicializar arreglo para la energía cinética de los brazos
    for i in range(1, len(t)):
        if v[i-1] < 0.01:  # Verificar si la velocidad es muy pequeña
            F_drag = 0  # Si es así, establecer la fuerza de arrastre en cero
        else:
            F_drag = 0.5 * rho * A * Cd * v[i-1]**2

        F_buoy = rho * g * masa * np.exp(-t[i]/10)  # Modelo de decaimiento exponencial para la fuerza de empuje
        F_total = F_buoy - F_drag
        a = F_total / masa
        v[i] = v[i-1] + a * dt

        # Calcular la energía cinética de los brazos en este paso
        arm_kinetic_energy[i] = 0.5 * masa * v[i]**2

        # Comprobamos si el nadador ha llegado al final de la piscina y lo "teletransportamos" al inicio
        if (i*dt) % longitud_piscina == 0:
            v[i] = v0

        work += F_drag * v[i-1] * dt

    # Calcular la velocidad máxima
    vmax = np.max(v)

    return t, v, work, arm_rotation, arm_kinetic_energy, vmax
```
# Simulación del movimiento de natación y cálculo del trabajo y la energía cinética de los brazos para cada masa

```python
resultados_trabajo = []
resultados_energia_brazos = []
resultados_velocidad_maxima = []
for masa in masas:
    t, v, work, arm_rotation, arm_kinetic_energy, vmax = swim(masa)
    resultados_trabajo.append([masa, work])
    resultados_energia_brazos.append([masa, np.sum(arm_kinetic_energy)])
    resultados_velocidad_maxima.append([masa, vmax])
```
### Mostrar los resultados como un DataFrame

```python
df_trabajo = pd.DataFrame(resultados_trabajo, columns=['Masa (kg)', 'Trabajo (J)'])
df_energia_brazos = pd.DataFrame(resultados_energia_brazos, columns=['Masa (kg)', 'Energía de los brazos (J)'])
df_velocidad_maxima = pd.DataFrame(resultados_velocidad_maxima, columns=['Masa (kg)', 'Velocidad Máxima (m/s)'])
```
## Redondear los resultados a 4 cifras significativas

```python
df_trabajo['Trabajo (J)'] = df_trabajo['Trabajo (J)'].round(4)
df_energia_brazos['Energía de los brazos (J)'] = df_energia_brazos['Energía de los brazos (J)'].round(4)

# Combinar los DataFrames
df_resultados_nadador = pd.merge(df_trabajo, df_energia_brazos, on='Masa (kg)')
df_resultados_nadador = pd.merge(df_resultados_nadador, df_velocidad_maxima, on='Masa (kg)')

# Redondear la velocidad máxima a 4 cifras significativas
df_resultados_nadador['Velocidad Máxima (m/s)'] = df_resultados_nadador['Velocidad Máxima (m/s)'].round(4)

print("Resultados - Trabajo, Energía de los brazos y Velocidad Máxima:")
print(df_resultados_nadador)

#Guardar los datos en un excel
df_resultados_nadador.to_excel("Resultados_nadador.xlsx", index=True)
```

### Graficar los datos

```python
# Graficar los resultados del trabajo realizado
plt.figure(figsize=(10, 6))
plt.scatter(df_resultados_nadador['Masa (kg)'], df_resultados_nadador['Trabajo (J)'], label='Trabajo realizado')
plt.plot(df_resultados_nadador['Masa (kg)'], np.poly1d(np.polyfit(df_resultados_nadador['Masa (kg)'], df_resultados_nadador['Trabajo (J)'], 1))(df_resultados_nadador['Masa (kg)']), color='red', label='Línea de tendencia')
plt.xlabel('Masa (kg)')
plt.ylabel('Trabajo (J)')
plt.title('Trabajo realizado en función de la masa del nadador con línea de tendencia')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los resultados del trabajo realizado
plt.figure(figsize=(10, 6))
plt.scatter(df_resultados_nadador['Masa (kg)'], df_resultados_nadador['Energía de los brazos (J)'], label='Energia en los brazos')
plt.plot(df_resultados_nadador['Masa (kg)'], np.poly1d(np.polyfit(df_resultados_nadador['Masa (kg)'], df_resultados_nadador['Energía de los brazos (J)'], 1))(df_resultados_nadador['Masa (kg)']), color='red', label='Línea de tendencia')
plt.xlabel('Masa (kg)')
plt.ylabel('Energía de los brazos (J)')
plt.title('Energia realizada en función de la masa del nadador con línea de tendencia')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los resultados del trabajo realizado
plt.figure(figsize=(10, 6))
plt.scatter(df_resultados_nadador['Masa (kg)'], df_resultados_nadador['Velocidad Máxima (m/s)'], label='Velocidad Máxima (m/s)')
plt.plot(df_resultados_nadador['Masa (kg)'], np.poly1d(np.polyfit(df_resultados_nadador['Masa (kg)'], df_resultados_nadador['Velocidad Máxima (m/s)'], 1))(df_resultados_nadador['Masa (kg)']), color='red', label='Línea de tendencia')
plt.xlabel('Masa (kg)')
plt.ylabel('Velocidad Máxima (m/s)')
plt.title('Velocidad alcanzada en función de la masa del nadador con línea de tendencia')
plt.legend()
plt.grid(True)
plt.show()
```

#### Cuerpo libre

```python
def diagrama_cuerpo_libre():
    plt.figure(figsize=(8, 6))
    plt.title('Diagrama de Cuerpo Libre del Nadador')
    plt.xlim(-1, 1)
    plt.ylim(0, 2)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(0, 1, color='black', label='Nadador (Partícula)')

    # Fuerza de flotación (flecha hacia arriba)
    plt.arrow(0, 1.2, 0, 0.3, head_width=0.1, head_length=0.1, fc='b', ec='b', label='Fuerza de Flotación (Ff)')

    # Fuerza de arrastre (flecha hacia abajo)
    plt.arrow(0, 0.8, 0, -0.3, head_width=0.1, head_length=0.1, fc='r', ec='r', label='Fuerza de Arrastre (Fa)')

    # Peso del nadador (flecha hacia abajo)
    plt.arrow(0, 1, 0, -0.5, head_width=0.1, head_length=0.1, fc='g', ec='g', label='Peso (P)')

    # Fuerza de gravedad (flecha hacia abajo)
    plt.arrow(0, 0.8, 0, -0.3, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linestyle='dotted', label='Fuerza de Gravedad (Fg)')

    plt.legend()
    plt.text(0.5, 1, "Las fuerzas de resistencia del agua y la fuerza de empuje\nse desprecian debido a su influencia relativamente pequeña\nen comparación con otras fuerzas en este modelo.", fontsize=10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    plt.show()


diagrama_cuerpo_libre()
```
# **Analisis de las piernas**

1. Definir funciones y calcular velocidad y energia 

```python
def calcular_energia_piernas(masa, velocidad):
    # Calcula la energía cinética de las piernas como una partícula
    energia_piernas = 0.5 * masa * velocidad**2
    return energia_piernas
def swim_p(masa):
    t = np.arange(t0, tf, dt)
    v = np.zeros_like(t)
    v[0] = v0
    leg_kinetic_energy = np.zeros_like(t)  # Inicializar arreglo para la energía cinética de las piernas
    for i in range(1, len(t)):
        if v[i-1] < 0.01:  # Verificar si la velocidad es muy pequeña
            F_drag = 0  # Si es así, establecer la fuerza de arrastre en cero
        else:
            F_drag = 0.5 * rho * A * Cd * v[i-1]**2

        F_buoy = rho * g * masa * np.exp(-t[i]/10)  # Modelo de decaimiento exponencial para la fuerza de empuje
        F_total = F_buoy - F_drag
        a = F_total / masa
        v[i] = v[i-1] + a * dt

        # Calcular la energía cinética de las piernas (partícula) en este paso
        leg_kinetic_energy[i] = calcular_energia_piernas(masa, v[i])  # Utiliza la función para calcular la energía de las piernas

        # Comprobamos si el nadador ha llegado al final de la piscina y lo "teletransportamos" al inicio
        if (i*dt) % longitud_piscina == 0:
            v[i] = v0

    return v, leg_kinetic_energy

# Calcular la energía de las piernas y la velocidad para cada masa
resultados_energia_piernas = []
resultados_velocidad = []
for masa in masas:
    velocidad, energia_piernas = swim_p(masa)
    resultados_energia_piernas.append([masa, np.sum(energia_piernas)])
    resultados_velocidad.append([masa, np.max(velocidad)])
```

## Crear un dataframe

```python
# Crear DataFrame con los resultados de la energía de las piernas y la velocidad
df_energia_piernas = pd.DataFrame(resultados_energia_piernas, columns=['Masa (kg)', 'Energía de las piernas (J)'])
df_velocidad = pd.DataFrame(resultados_velocidad, columns=['Masa (kg)', 'Velocidad máxima (m/s)'])

# Combinar DataFrames
df_resultados = pd.merge(df_energia_piernas, df_velocidad, on='Masa (kg)')

# Mostrar DataFrame
print(df_resultados)
df_resultados.to_excel("Piernas como una particula.xlsx",index=False)
```
## Gráficos

```python
# Graficar los resultados del trabajo realizado
plt.figure(figsize=(10, 6))
plt.scatter(df_resultados['Masa (kg)'], df_resultados['Energía de las piernas (J)'], label='Energia realizada en las piernas')
plt.plot(df_resultados['Masa (kg)'], np.poly1d(np.polyfit(df_resultados['Masa (kg)'], df_resultados['Energía de las piernas (J)'], 1))(df_resultados['Masa (kg)']), color='g', label='Línea de tendencia')
plt.xlabel('Masa (kg)')
plt.ylabel('Energía de las piernas (J)')
plt.title('Energia acomulada en función de la masa del nadador con línea de tendencia')
plt.legend()
plt.grid(True)
plt.show()

# Función para graficar el diagrama de cuerpo libre
def graficar_diagrama_cuerpo_libre():
    # Configuración del diagrama
    plt.figure(figsize=(6, 6))
    plt.title('Diagrama de Cuerpo Libre de las Piernas del Nadador')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')

    # Dibujar el cuerpo del nadador (partícula)
    plt.scatter(0, 0, color='black', label='Nadador (Partícula)')

    # Dibujar la fuerza de flotación (hacia arriba)
    plt.arrow(0, 0, 0, 0.5, head_width=0.1, head_length=0.1, fc='b', ec='b', label='Fuerza de Flotación')

    # Dibujar la fuerza de gravedad (hacia abajo)
    plt.arrow(0, 0, 0, -0.5, head_width=0.1, head_length=0.1, fc='g', ec='g', label='Peso')

    # Texto explicativo
    plt.text(0.5, 0.2, "Fuerzas despreciadas:\n- Resistencia del agua\n- Fuerza muscular de las piernas", fontsize=10)

    # Mostrar leyenda
    plt.legend()

    # Mostrar el diagrama
    plt.show()

# Llamar a la función para graficar el diagrama de cuerpo libre
graficar_diagrama_cuerpo_libre()
```

# Calcular la energia potencial

```python
def calcular_energia_potencial(masa):
    # Energía potencial gravitatoria
    energia_potencial = masa * g * longitud_piscina / 2  # Suponiendo que la piscina está nivelada horizontalmente
    return energia_potencial

# Calcular la energía potencial gravitatoria para cada masa
resultados_energia_potencial = []
for masa in masas:
    energia_potencial = calcular_energia_potencial(masa)
    resultados_energia_potencial.append([masa, energia_potencial])

# Crear DataFrame con los resultados de la energía potencial gravitatoria
df_energia_potencial = pd.DataFrame(resultados_energia_potencial, columns=['Masa (kg)', 'Energía Potencial Gravitacional (J)'])

# Mostrar DataFrame
print(df_energia_potencial)
```
1. Graficar los resultados del trabajo realizado

```python
# Graficar los resultados del trabajo realizado
plt.figure(figsize=(10, 6))
plt.scatter(df_energia_potencial['Masa (kg)'], df_energia_potencial['Energía Potencial Gravitacional (J)'], label='Energia gravitacional en las piernas')
plt.plot(df_energia_potencial['Masa (kg)'], np.poly1d(np.polyfit(df_energia_potencial['Masa (kg)'], df_energia_potencial['Energía Potencial Gravitacional (J)'], 1))(df_energia_potencial['Masa (kg)']), color='g', label='Línea de tendencia')
plt.xlabel('Masa (kg)')
plt.ylabel('Energía Potencial Gravitacional (J)')
plt.title('Energia Potencial Gravitacional en función de la masa del nadador con línea de tendencia')
plt.legend()
plt.grid(True)
plt.show()
#Guardar el dataframe
df_energia_potencial.to_excel("energia gravitacional.xlsx",index=False)
```
