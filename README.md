# QuoreMindHP v1.0.0 - Framework de Alta Precisión
[![License: Apache2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/licenses/Apache2.0)
![Project Status](https://img.shields.io/badge/Project%20Status-In%20Progress-blue)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)




![IMG-20250411-WA0000](https://github.com/user-attachments/assets/d82e4a7e-e01a-4170-bec7-248e487e1abb)

**Un framework en Python para lógica bayesiana, análisis estadístico y modelado de Ruido Probabilístico de Referencia (PRN) utilizando aritmética de precisión arbitraria.**

Basado en el concepto original de QuoreMind v1.0.0 por Jacobo Tlacaelel Mina Rodríguez.
Adaptado con alta precisión usando la biblioteca `mpmath`.

---

## Características Principales

* **Lógica Bayesiana de Alta Precisión:** Realiza cálculos de probabilidad posterior, condicional y conjunta utilizando `mpmath` para máxima exactitud, especialmente útil con probabilidades muy pequeñas o cercanas a 1.
* **Análisis Estadístico Robusto:**
    * Cálculo de la **Distancia de Mahalanobis** con alta precisión, incluyendo la inversión de la matriz de covarianza, lo que aumenta la estabilidad numérica para datos mal condicionados.
    * Cálculo de la **Entropía de Shannon** y **Cosenos Direccionales** con precisión arbitraria.
* **Modelo PRN de Alta Precisión:** Implementación de la clase `PRN_HP` donde el factor de influencia y las operaciones de combinación se manejan con `mpmath`.
* **Configuración Flexible:** Permite ajustar la precisión decimal global deseada para todos los cálculos a través de `mpmath.mp.dps`.
* **Utilidades de Alta Precisión:** Incluye funciones para calcular constantes matemáticas (como 'e') con precisión configurable.
* **Decoradores:** Incluye decoradores para medir tiempos de ejecución y validar rangos de entrada para los tipos `mpmath`.

## Motivación

La aritmética de punto flotante estándar (float de 64 bits) puede sufrir de errores de redondeo que se acumulan en cálculos complejos o iterativos. Esto puede ser problemático en áreas como:

* Simulaciones científicas sensibles.
* Análisis estadístico con datos mal condicionados (e.g., matrices de covarianza casi singulares en Mahalanobis).
* Lógica bayesiana con probabilidades extremadamente pequeñas.

`QuoreMindHP` extiende el framework `QuoreMind` original incorporando la biblioteca `mpmath` para realizar estos cálculos críticos con una precisión arbitraria definida por el usuario, mejorando la fiabilidad y robustez numérica.

## Dependencias

* **Python:** 3.7+ recomendado
* **mpmath:** Para la aritmética de alta precisión (`pip install mpmath`)
* **NumPy:** Utilizado internamente y para comparación (`pip install numpy`)
* **SciPy:** (Opcional, para comparación de Mahalanobis) (`pip install scipy`)
* **Scikit-learn:** (Opcional, para cálculo de covarianza estándar para comparación) (`pip install scikit-learn`)

## Instalación

1.  Clona o descarga este repositorio/script.
2.  Instala las dependencias requeridas:
    ```bash
    pip install mpmath numpy scipy scikit-learn
    ```

## Configuración de la Precisión

La precisión de los cálculos se controla globalmente estableciendo los dígitos decimales deseados (`dps`) para `mpmath`. Puedes hacerlo al inicio de tu script:

```python
import mpmath

# Establece la precisión deseada (e.g., 100 dígitos decimales)
PRECISION_GLOBAL = 100 
mpmath.mp.dps = PRECISION_GLOBAL 

# ... el resto de tu código usando QuoreMindHP ...
```

El framework QuoreMindHP usará esta precisión para todas sus operaciones internas.

## Uso

### Ejemplo Básico

```python
import mpmath
from quoremind_hp import BayesLogicHP, StatisticalAnalysisHP, PRN_HP, BayesLogicConfigHP # Asume que guardaste el script como quoremind_hp.py

# 1. Configurar precisión
mpmath.mp.dps = 50 

# 2. Instanciar componentes
config_bayes = BayesLogicConfigHP(action_threshold="0.6") # Configuración personalizada
bayes_logic = BayesLogicHP(config_bayes)
stats = StatisticalAnalysisHP()
prn = PRN_HP(influence="0.75", algorithm_type="custom_algo")

# 3. Preparar datos (usar strings para máxima precisión inicial si es necesario)
entropy = stats.shannon_entropy([1, 2, 2, 3, 1, 4]) 
coherence = mpmath.mpf("0.85")
previous_action = 1

# 4. Ejecutar lógica bayesiana
decision_info = bayes_logic.calculate_probabilities_and_select_action(
    entropy, coherence, prn.influence, previous_action
)

print("--- Decisión Bayesiana HP ---")
print(f"Acción a tomar: {decision_info['action_to_take']}")
print(f"Probabilidad Condicional Acción: {mpmath.nstr(decision_info['conditional_action_given_b'], n=20)}")

# 5. Calcular distancia de Mahalanobis HP
data_set = [['1.0', '1.1'], ['1.01', '1.12'], ['0.99', '1.08'], ['3.0', '3.1']]
point = ['1.5', '1.6']
mahal_dist = stats.compute_mahalanobis_distance_hp(data_set, point)
print(f"\nDistancia de Mahalanobis HP: {mpmath.nstr(mahal_dist, n=mpmath.mp.dps)}") 
```

### Ejecutar Demostraciones

El script principal (quoremind_hp.py si lo guardaste así) incluye funciones de demostración que puedes ejecutar directamente para ver ejemplos de cada componente en acción:

```bash
python quoremind_hp.py 
```

Esto ejecutará `run_bayes_logic_hp_example()`, `run_statistical_analysis_hp_example()`, `run_prn_hp_example()` y `run_e_calculation_example()`.

### Calcular Constantes (ej. 'e')

Puedes usar la función incluida para calcular 'e' u otras constantes/funciones soportadas por mpmath con la precisión que necesites:

```python
from quoremind_hp import calculate_e_mpmath
import mpmath

# Calcular 'e' con 200 dígitos decimales usando Taylor
e_val = calculate_e_mpmath(method='taylor', iterations=200, precision_dps=200)
print(mpmath.nstr(e_val, n=200))

# Obtener Pi con la precisión global actual
print(mpmath.pi) 
```

## Core Components

* **BayesLogicConfigHP**: Dataclass para configurar los umbrales y epsilon de BayesLogicHP usando tipos mpmath.mpf.
* **BayesLogicHP**: Implementa las fórmulas de Bayes y lógica de decisión relacionada usando aritmética mpmath.
* **StatisticalAnalysisHP**: Provee métodos para calcular entropía, cosenos y, crucialmente, la distancia de Mahalanobis con alta precisión (incluyendo cálculo de media, covarianza e inversa de covarianza con mpmath).
* **PRN_HP**: Modela el Ruido Probabilístico de Referencia, manejando el factor de influencia (influence) como mpmath.mpf.
* **calculate_e_mpmath**: Función de ejemplo/utilidad para calcular la constante 'e' con alta precisión.

## Potenciales Aplicaciones

Este framework es útil en escenarios donde la precisión numérica es crítica:

* **Simulaciones científicas**: Física, astronomía, sistemas caóticos.
* **Estadística y Machine Learning**: Análisis de datos con posibles problemas de condicionamiento numérico, evaluación de modelos, lógica bayesiana robusta.
* **Finanzas Cuantitativas**: Modelos complejos, cálculo de riesgos.
* **Geometría Computacional**: Cálculos precisos de intersecciones o distancias.
* **Investigación y Educación**: Exploración de constantes, demostración de limitaciones numéricas, verificación de algoritmos.
* **Integración con tu proyecto cuadrante-coremind**: Especialmente donde la lógica bayesiana, la distancia de Mahalanobis o las simulaciones (potencialmente cuánticas) requieran mayor robustez numérica.

## Licencia

Este proyecto se distribuye bajo la Licencia Apache 2.0 Consulta el archivo LICENSE para más detalles.

## Autor

* **Concepto Original (QuoreMind)**
* **Adaptación a Alta Precisión (QuoreMindHP)**:[ Jacobo Tlacaelel Mina Rodríguez. "jako" ]

## Etiquetas

`python` `bayes` `estadística` `matemáticas` `alta-precisión` `mpmath` `mahalanobis` `bayesian-logic` `shannon-entropy` `numerical-robustness` `computational-science` `machinelearning`
