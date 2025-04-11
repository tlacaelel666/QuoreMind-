"""
QuoreMindHP v1.0.0 - Framework de Alta Precisión

Este marco de trabajo en Python proporciona herramientas para implementar 
lógica bayesiana, realizar análisis estadísticos (incluyendo distancia de Mahalanobis 
de alta precisión) y modelar el "Ruido Probabilístico de Referencia" (PRN), 
utilizando aritmética de precisión arbitraria con la biblioteca mpmath.

Fecha: 11-04-2025  (Adapta la fecha si es necesario)
Autor: Jacobo Tlacaelel Mina Rodríguez (adaptado por IA)
Proyecto: QuoreMindHP v1.0.0 (Basado en QuoreMind v1.0.0)
"""

import numpy as np
# Quitamos TensorFlow y TFP por ahora para centrarnos en mpmath y numpy/scipy
# Si necesitas TFP específicamente, la integración sería más compleja
# import tensorflow as tf 
# import tensorflow_probability as tfp 
import mpmath
from typing import Tuple, List, Dict, Union, Any, Optional, Callable, TypeVar
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis # Para comparación
from sklearn.covariance import EmpiricalCovariance # Para obtener covarianza inicial
import functools
import time
from dataclasses import dataclass

# --- Configuración Global de Precisión ---
# Establece los dígitos decimales de precisión para mpmath
PRECISION_DPS = 50  # Por ejemplo, 50 dígitos decimales
mpmath.mp.dps = PRECISION_DPS 

# Definimos un tipo para los números de alta precisión de mpmath
MP_Float = type(mpmath.mpf(1.0)) 
MP_Complex = type(mpmath.mpc(1.0 + 1.0j))

# --- Decoradores Adaptados ---

FuncType = TypeVar('FuncType', bound=Callable[..., Any])

def timer_decorator(func: FuncType) -> FuncType:
    """Decorador que mide el tiempo de ejecución de una función."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # Usar time.perf_counter para mejor precisión
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Función {func.__name__:<30} ejecutada en {end_time - start_time:.6f} segundos")
        return result
    return wrapper

def validate_mp_input_decorator(min_val: Union[float, str] = "0.0", 
                                max_val: Union[float, str] = "1.0") -> Callable[[FuncType], FuncType]:
    """
    Decorador que valida que los argumentos numéricos (mpmath.mpf) estén en un rango.
    Se usan strings para inicializar los límites con precisión.
    """
    min_mpf = mpmath.mpf(min_val)
    max_mpf = mpmath.mpf(max_val)
    
    def decorator(func: FuncType) -> FuncType:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validar args posicionales (ignorando 'self' si es método)
            start_index = 1 if args and hasattr(args[0], func.__name__) else 0
            for i, arg in enumerate(args[start_index:], start_index):
                if isinstance(arg, MP_Float) and not (min_mpf <= arg <= max_mpf):
                    raise ValueError(
                        f"Argumento posicional {i} ({arg}) debe estar entre {min_mpf} y {max_mpf}."
                    )
            
            # Validar kwargs
            for name, arg in kwargs.items():
                 if isinstance(arg, MP_Float) and not (min_mpf <= arg <= max_mpf):
                    raise ValueError(
                        f"Argumento '{name}' ({arg}) debe estar entre {min_mpf} y {max_mpf}."
                    )
            return func(*args, **kwargs)
        return decorator
    return decorator

# --- Clases Principales Adaptadas ---

@dataclass
class BayesLogicConfigHP:
    """Configuración para BayesLogicHP usando mpmath."""
    epsilon: MP_Float = mpmath.mpf("1e-15") # Epsilon más pequeño gracias a alta precisión
    high_entropy_threshold: MP_Float = mpmath.mpf("0.8")
    high_coherence_threshold: MP_Float = mpmath.mpf("0.6")
    action_threshold: MP_Float = mpmath.mpf("0.5")

    def __post_init__(self):
        # Asegurar que todos los umbrales sean mpf
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if not isinstance(value, MP_Float):
                setattr(self, field_name, mpmath.mpf(str(value)))


class BayesLogicHP:
    """
    Clase para lógica Bayesiana usando aritmética de alta precisión (mpmath).
    """
    def __init__(self, config: Optional[BayesLogicConfigHP] = None):
        self.config = config or BayesLogicConfigHP()

    @validate_mp_input_decorator("0.0", "1.0")
    def calculate_posterior_probability(self, prior_a: MP_Float, prior_b: MP_Float, 
                                        conditional_b_given_a: MP_Float) -> MP_Float:
        """Calcula P(A|B) = (P(B|A) * P(A)) / P(B) con alta precisión."""
        prior_b_safe = mpmath.fmax(prior_b, self.config.epsilon)
        # Usar mpmath para todas las operaciones
        posterior = mpmath.fdiv(mpmath.fmul(conditional_b_given_a, prior_a), prior_b_safe)
        return mpmath.fmax(0, mpmath.fmin(1, posterior)) # Asegurar resultado en [0, 1]

    @validate_mp_input_decorator("0.0", "1.0")
    def calculate_conditional_probability(self, joint_probability: MP_Float, 
                                          prior: MP_Float) -> MP_Float:
        """Calcula P(X|Y) = P(X, Y) / P(Y) con alta precisión."""
        prior_safe = mpmath.fmax(prior, self.config.epsilon)
        conditional = mpmath.fdiv(joint_probability, prior_safe)
        return mpmath.fmax(0, mpmath.fmin(1, conditional))

    @validate_mp_input_decorator(min_val="0.0", max_val="1.0")
    def calculate_high_entropy_prior(self, entropy: MP_Float) -> MP_Float:
        """Deriva P(A) basado en entropía alta."""
        return mpmath.mpf("0.3") if entropy > self.config.high_entropy_threshold else mpmath.mpf("0.1")

    @validate_mp_input_decorator(min_val="0.0", max_val="1.0")
    def calculate_high_coherence_prior(self, coherence: MP_Float) -> MP_Float:
        """Deriva P(B) basado en coherencia alta."""
        return mpmath.mpf("0.6") if coherence > self.config.high_coherence_threshold else mpmath.mpf("0.2")

    @validate_mp_input_decorator(min_val="0.0", max_val="1.0")
    def calculate_joint_probability(self, coherence: MP_Float, action: int, 
                                     prn_influence: MP_Float) -> MP_Float:
        """Calcula P(A, B) basado en coherencia, acción e influencia PRN."""
        # Usamos mpmath para los cálculos internos
        if coherence > self.config.high_coherence_threshold:
            if action == 1:
                return prn_influence * mpmath.mpf("0.8") + (mpmath.mpf(1) - prn_influence) * mpmath.mpf("0.2")
            else:
                return prn_influence * mpmath.mpf("0.1") + (mpmath.mpf(1) - prn_influence) * mpmath.mpf("0.7")
        else:
            return mpmath.mpf("0.3")

    @timer_decorator
    @validate_mp_input_decorator(min_val="0.0", max_val="1.0") # Valida entropy, coherence, prn_influence
    def calculate_probabilities_and_select_action(
        self, entropy: MP_Float, coherence: MP_Float, prn_influence: MP_Float, action: int
    ) -> Dict[str, Union[int, MP_Float]]:
        """Integra los cálculos bayesianos de alta precisión."""
        
        # Asegurarse que las entradas son mpf (podrían venir como float)
        entropy = mpmath.mpf(str(entropy))
        coherence = mpmath.mpf(str(coherence))
        prn_influence = mpmath.mpf(str(prn_influence))
        
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        # P(B|A) ajustada por entropía
        one = mpmath.mpf(1)
        cond_b_a = (prn_influence * mpmath.mpf("0.7") + (one - prn_influence) * mpmath.mpf("0.3")
                    if entropy > self.config.high_entropy_threshold 
                    else mpmath.mpf("0.2"))

        # P(A|B) usando Bayes
        posterior_a_given_b = self.calculate_posterior_probability(
            high_entropy_prior, high_coherence_prior, cond_b_a
        )

        # P(A, B)
        joint_probability_ab = self.calculate_joint_probability(
            coherence, action, prn_influence
        )

        # P(Action|B)
        conditional_action_given_b = self.calculate_conditional_probability(
            joint_probability_ab, high_coherence_prior
        )

        # Decisión final
        action_to_take = 1 if conditional_action_given_b > self.config.action_threshold else 0

        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }


class StatisticalAnalysisHP:
    """Análisis estadístico usando mpmath donde sea apropiado."""

    @staticmethod
    def shannon_entropy(data: List[Any]) -> MP_Float:
        """Calcula la entropía de Shannon con alta precisión."""
        if not data:
            return mpmath.mpf(0)
            
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        
        entropy = mpmath.mpf(0)
        for p in probabilities:
            if p > 0:
                p_mpf = mpmath.mpf(str(p)) # Convertir a mpf
                entropy -= p_mpf * mpmath.log(p_mpf, 2) # Usar mpmath.log con base 2

        return entropy

    @staticmethod
    def calculate_cosines(entropy: MP_Float, prn_object: MP_Float) -> Tuple[MP_Float, MP_Float, MP_Float]:
        """Calcula cosenos direccionales con alta precisión."""
        one = mpmath.mpf(1)
        epsilon_mpf = mpmath.mpf("1e-50") # Epsilon muy pequeño
        
        # Asegurar que no sean exactamente cero para evitar problemas en magnitud si ambos son cero
        entropy = mpmath.fmax(entropy, epsilon_mpf)
        prn_object = mpmath.fmax(prn_object, epsilon_mpf)

        magnitude = mpmath.sqrt(entropy**2 + prn_object**2 + one)
        
        cos_x = mpmath.fdiv(entropy, magnitude)
        cos_y = mpmath.fdiv(prn_object, magnitude)
        cos_z = mpmath.fdiv(one, magnitude)
        return cos_x, cos_y, cos_z

    @staticmethod
    @timer_decorator
    def compute_mahalanobis_distance_hp(data: List[List[Union[float, str]]], 
                                        point: List[Union[float, str]]) -> MP_Float:
        """
        Calcula la distancia de Mahalanobis con alta precisión usando mpmath.

        Args:
            data: Conjunto de datos (lista de listas de números o strings representándolos).
            point: Punto para el cual se calcula la distancia.
        
        Returns:
            Distancia de Mahalanobis como mpmath.mpf.
        """
        # Convertir datos a matriz mpmath
        try:
            data_mp = mpmath.matrix(data)
            point_mp = mpmath.matrix(point)
        except Exception as e:
            raise ValueError(f"Error al convertir datos a mpmath.matrix: {e}")
            
        if data_mp.cols != point_mp.cols or point_mp.rows != 1:
             raise ValueError("Dimensiones incompatibles entre los datos y el punto.")

        n_rows = data_mp.rows
        n_cols = data_mp.cols

        if n_rows <= n_cols:
            print("ADVERTENCIA: Número de muestras menor o igual al número de dimensiones.")
            print("             La matriz de covarianza puede ser singular.")
            # Considerar usar regularización o más datos si es posible
            
        # 1. Calcular vector de medias (alta precisión)
        mean_vector = mpmath.matrix(1, n_cols)
        for j in range(n_cols):
            col_sum = mpmath.fsum(data_mp[:, j])
            mean_vector[0, j] = mpmath.fdiv(col_sum, n_rows)
            
        # 2. Calcular matriz de covarianza (alta precisión)
        #    Cov(X, Y) = E[(X - E[X]) * (Y - E[Y])]
        cov_matrix = mpmath.zeros(n_cols)
        for i in range(n_cols):
            for j in range(i, n_cols): # Calcular solo la triangular superior
                sum_prod = mpmath.mpf(0)
                for k in range(n_rows):
                    diff_i = data_mp[k, i] - mean_vector[0, i]
                    diff_j = data_mp[k, j] - mean_vector[0, j]
                    sum_prod += diff_i * diff_j
                # Usar n o n-1 para la covarianza muestral/poblacional? 
                # Scipy/numpy usan n-1 por defecto (estimador insesgado)
                # Usaremos n-1 aquí también para consistencia, si n > 1
                denom = mpmath.mpf(n_rows - 1) if n_rows > 1 else mpmath.mpf(1)
                cov_ij = mpmath.fdiv(sum_prod, denom)
                cov_matrix[i, j] = cov_ij
                if i != j:
                    cov_matrix[j, i] = cov_ij # Simétrica

        print("Matriz de Covarianza (mpmath):")
        mpmath.pprint(cov_matrix)

        # 3. Calcular inversa de la matriz de covarianza (alta precisión)
        try:
            # Usar mpmath.inverse()
            inv_cov_matrix = mpmath.inverse(cov_matrix)
            print("Inversa de la Matriz de Covarianza (mpmath):")
            mpmath.pprint(inv_cov_matrix)
        except mpmath.LUError: # Ocurre si es singular
            print("ADVERTENCIA: Matriz de covarianza singular (detectado por mpmath).")
            # mpmath no tiene una pseudo-inversa directa como numpy.
            # Podría intentarse regularización (añadir pequeña identidad)
            # O calcularla manualmente si fuera necesario (complejo).
            # Por ahora, retornamos infinito o NaN para indicar fallo.
            # return mpmath.inf 
            # Alternativa: Usar numpy pinv sobre versión float y convertir de vuelta?
            # Esto perdería precisión. Mejor indicamos el error.
            print("             Retornando Infinito como indicador de error.")
            return mpmath.inf
            
        # 4. Calcular diferencia del punto a la media
        diff_point = point_mp - mean_vector

        # 5. Calcular distancia Mahalanobis^2: (x-mu)^T * Sigma^-1 * (x-mu)
        #    En mpmath: diff * inv_cov * diff.T (si diff es fila)
        term1 = diff_point * inv_cov_matrix 
        mahalanobis_sq = term1 * diff_point.T 
        
        # El resultado de la multiplicación de matrices es una matriz 1x1
        mahalanobis_sq_val = mahalanobis_sq[0,0]

        # La distancia es la raíz cuadrada
        if mahalanobis_sq_val < 0:
             # Puede ocurrir por errores numéricos mínimos si es casi cero
             print(f"ADVERTENCIA: Mahalanobis^2 resultó negativo ({mahalanobis_sq_val}). Ajustando a cero.")
             mahalanobis_sq_val = mpmath.mpf(0)
             
        distance = mpmath.sqrt(mahalanobis_sq_val)
        return distance
        
    @staticmethod
    @timer_decorator
    def compute_mahalanobis_distance_numpy(data: List[List[float]], point: List[float]) -> float:
        """Calcula Mahalanobis usando numpy/scipy (precisión estándar) para comparación."""
        data_np = np.array(data, dtype=np.float64)
        point_np = np.array(point, dtype=np.float64)
        
        # Covarianza con sklearn (similar a scipy)
        cov_estimator = EmpiricalCovariance().fit(data_np)
        cov_matrix = cov_estimator.covariance_
        
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print("ADVERTENCIA (numpy): Matriz singular, usando pseudo-inversa.")
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            
        mean_vector = np.mean(data_np, axis=0)
        
        # Usar scipy.spatial.distance.mahalanobis
        distance = scipy_mahalanobis(point_np, mean_vector, inv_cov_matrix)
        return distance


class PRN_HP:
    """PRN usando mpmath para la influencia."""
    def __init__(self, influence: Union[MP_Float, float, str], 
                 algorithm_type: Optional[str] = None, **parameters):
        # Convertir influencia a mpf asegurando precisión
        self.influence = mpmath.mpf(str(influence)) 
        if not (mpmath.mpf(0) <= self.influence <= mpmath.mpf(1)):
            raise ValueError(f"La influencia debe estar entre 0 y 1. Valor recibido: {self.influence}")

        self.algorithm_type = algorithm_type
        self.parameters = parameters # Estos podrían ser mpf también si es necesario

    def adjust_influence(self, adjustment: Union[MP_Float, float, str]) -> None:
        """Ajusta influencia con alta precisión."""
        adj_mpf = mpmath.mpf(str(adjustment))
        new_influence = self.influence + adj_mpf
        # Truncar al rango [0, 1] usando mpmath
        new_influence = mpmath.fmax(0, mpmath.fmin(1, new_influence))
        self.influence = new_influence

    def combine_with(self, other_prn: 'PRN_HP', 
                     weight: Union[MP_Float, float, str] = "0.5") -> 'PRN_HP':
        """Combina PRNs con alta precisión."""
        weight_mpf = mpmath.mpf(str(weight))
        if not (mpmath.mpf(0) <= weight_mpf <= mpmath.mpf(1)):
             raise ValueError(f"El peso debe estar entre 0 y 1. Valor recibido: {weight_mpf}")
             
        one = mpmath.mpf(1)
        combined_influence = (self.influence * weight_mpf + 
                              other_prn.influence * (one - weight_mpf))

        combined_params = {**self.parameters, **other_prn.parameters}
        algorithm = self.algorithm_type if weight_mpf >= mpmath.mpf("0.5") else other_prn.algorithm_type
        return PRN_HP(combined_influence, algorithm, **combined_params)

    def __str__(self) -> str:
        # Usar nstr para formato con precisión controlada
        influence_str = mpmath.nstr(self.influence, n=10) 
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo_str = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"PRN_HP(influence={influence_str}{algo_str}{', ' + params_str if params_str else ''})"

# Podrías crear ComplexPRN_HP de forma similar usando mpmath.mpc

# --- Función para Calcular 'e' con Alta Precisión ---
@timer_decorator
def calculate_e_mpmath(method: str = 'taylor', 
                       iterations: int = 100, 
                       precision_dps: Optional[int] = None) -> MP_Float:
    """
    Calcula 'e' con alta precisión usando mpmath.

    Args:
        method: 'taylor' o 'limit' o 'const'.
        iterations: Número de iteraciones para Taylor/Límite.
        precision_dps: Precisión decimal temporal para este cálculo.

    Returns:
        Valor de 'e' como mpmath.mpf.
    """
    original_dps = mpmath.mp.dps
    if precision_dps is not None:
        mpmath.mp.dps = precision_dps
        
    e_result = mpmath.mpf(0)
    one = mpmath.mpf(1)

    if method == 'taylor':
        e_result = one
        fact = one
        for i in range(1, iterations + 1):
            fact *= i
            term = mpmath.fdiv(one, fact)
            if mpmath.fabs(term) < mpmath.mpf(f'1e-{(precision_dps or original_dps) + 2}'): # Criterio de parada
                 print(f"    Convergencia de Taylor alcanzada en iteración {i}")
                 break
            e_result += term
            
    elif method == 'limit':
        # (1 + 1/n)^n para n grande. Usaremos 'iterations' como n.
        n = mpmath.mpf(iterations) 
        if n <= 0: n = mpmath.mpf('1e6') # Valor por defecto si iterations no es útil
        base = one + mpmath.fdiv(one, n)
        e_result = mpmath.power(base, n)
        
    elif method == 'const':
        e_result = mpmath.exp(one) # Usar la constante interna de mpmath
        
    else:
        raise ValueError("Método inválido. Usar 'taylor', 'limit' o 'const'.")

    # Restaurar precisión original
    mpmath.mp.dps = original_dps
    return e_result


# --- Funciones de Demostración Adaptadas ---

def run_bayes_logic_hp_example():
    print("\n" + "="*15 + " Ejemplo BayesLogicHP " + "="*15)
    # Usar mpmath.mpf para las entradas
    entropy_value = StatisticalAnalysisHP.shannon_entropy(['a', 'b', 'b', 'c', 'a', 'b', 'd'])
    coherence_value = mpmath.mpf("0.7")
    prn_influence = mpmath.mpf("0.8")
    action_input = 1

    # Configuración personalizada opcional
    config_hp = BayesLogicConfigHP(
        epsilon="1e-30", 
        high_entropy_threshold="0.75", 
        action_threshold="0.51"
    )
    bayes_hp = BayesLogicHP(config_hp)
    
    decision = bayes_hp.calculate_probabilities_and_select_action(
        entropy_value, coherence_value, prn_influence, action_input
    )

    print("--- Resultados Decisión Bayesiana (Alta Precisión) ---")
    for key, value in decision.items():
        if isinstance(value, MP_Float):
            # Mostrar con buena precisión usando nstr
            print(f"  {key:<28}: {mpmath.nstr(value, n=PRECISION_DPS)}") 
        else:
            print(f"  {key:<28}: {value}")

def run_statistical_analysis_hp_example():
    print("\n" + "="*15 + " Ejemplo StatisticalAnalysisHP " + "="*15)
    stats_hp = StatisticalAnalysisHP()

    # 1. Entropía
    data_entropy = [1, 1, 2, 3, 3, 3, 4, 4, 5]
    entropy_hp = stats_hp.shannon_entropy(data_entropy)
    print(f"--- Entropía de Shannon (Alta Precisión) ---")
    print(f"Datos: {data_entropy}")
    print(f"Entropía: {mpmath.nstr(entropy_hp, n=PRECISION_DPS)}")

    # 2. Cosenos
    entropy_val = mpmath.mpf("0.9")
    prn_val = mpmath.mpf("0.2")
    cos_x, cos_y, cos_z = stats_hp.calculate_cosines(entropy_val, prn_val)
    print("\n--- Cosenos Direccionales (Alta Precisión) ---")
    print(f"cos_x: {mpmath.nstr(cos_x, n=15)}")
    print(f"cos_y: {mpmath.nstr(cos_y, n=15)}")
    print(f"cos_z: {mpmath.nstr(cos_z, n=15)}")

    # 3. Distancia de Mahalanobis (Alta Precisión vs NumPy)
    # Datos donde la matriz de covarianza podría ser sensible
    data_mahalanobis = [
        ['1.0', '2.000000000000001'],
        ['1.000000000000001', '2.0'],
        ['3.0', '4.0'],
        ['3.0', '4.000000000000002'],
        ['5.0', '6.0']
    ]
    point_mahalanobis = ['2.0', '3.0']

    print("\n--- Distancia de Mahalanobis (Alta Precisión vs NumPy) ---")
    print(f"Datos (representados como strings para precisión): {data_mahalanobis}")
    print(f"Punto: {point_mahalanobis}")

    # Convertir a float para numpy (puede perder precisión aquí)
    data_mahalanobis_float = [[float(x) for x in row] for row in data_mahalanobis]
    point_mahalanobis_float = [float(x) for x in point_mahalanobis]

    distance_hp = stats_hp.compute_mahalanobis_distance_hp(data_mahalanobis, point_mahalanobis)
    distance_np = stats_hp.compute_mahalanobis_distance_numpy(data_mahalanobis_float, point_mahalanobis_float)

    print(f"\nDistancia Mahalanobis (Alta Precisión): {mpmath.nstr(distance_hp, n=PRECISION_DPS)}")
    print(f"Distancia Mahalanobis (NumPy float64):  {distance_np:.15f}")
    diff = mpmath.fabs(distance_hp - mpmath.mpf(str(distance_np)))
    print(f"Diferencia Absoluta:                   {mpmath.nstr(diff, n=PRECISION_DPS)}")

def run_prn_hp_example():
    print("\n" + "="*15 + " Ejemplo PRN_HP " + "="*15)
    prn1 = PRN_HP(influence="0.65", algorithm_type="Kalman", state_dim=4)
    prn2 = PRN_HP(influence="0.8", algorithm_type="Particle", num_particles=5000)
    
    print(f"PRN 1: {prn1}")
    print(f"PRN 2: {prn2}")
    
    prn1.adjust_influence("-0.1")
    print(f"PRN 1 (ajustado): {prn1}")
    
    combined_prn = prn1.combine_with(prn2, weight="0.7")
    print(f"PRN Combinado: {combined_prn}")

def run_e_calculation_example():
    print("\n" + "="*15 + " Ejemplo Cálculo de 'e' (Alta Precisión) " + "="*15)
    
    prec_e = 100 # Calcular 'e' con 100 dígitos decimales
    print(f"Calculando 'e' con {prec_e} dígitos de precisión:")
    
    e_taylor = calculate_e_mpmath(method='taylor', iterations=100, precision_dps=prec_e)
    print(f"  e (Taylor, 100 iter): {mpmath.nstr(e_taylor, n=prec_e)}")

    # Para el límite, 'iterations' es 'n'. Necesita un n grande.
    e_limit = calculate_e_mpmath(method='limit', iterations=100000, precision_dps=prec_e) 
    print(f"  e (Límite, n=100k):   {mpmath.nstr(e_limit, n=prec_e)}")
    
    e_const = calculate_e_mpmath(method='const', precision_dps=prec_e)
    print(f"  e (mpmath.exp(1)):    {mpmath.nstr(e_const, n=prec_e)}")

def main():
    """Función principal que ejecuta todas las demostraciones de alta precisión."""
    print("===== DEMOSTRACIÓN QuoreMindHP (Alta Precisión) =====")
    print(f"Precisión global de mpmath establecida a: {mpmath.mp.dps} dígitos decimales")

    run_bayes_logic_hp_example()
    run_statistical_analysis_hp_example()
    run_prn_hp_example()
    run_e_calculation_example()

    print("\n" + "="*40)
    print("===== FIN DE LA DEMOSTRACIÓN =====")

if __name__ == "__main__":
    # Establecer una precisión por defecto al iniciar
    global_precision = 50 # Puedes ajustar esto
    mpmath.mp.dps = global_precision
    PRECISION_DPS = global_precision # Asegurar que la constante global se alinee
    
    main()
