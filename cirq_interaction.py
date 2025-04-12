# cirq_interactive.py
import cirq
import numpy as np
import time
import math
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any
import logging
from dataclasses import dataclass

# --- Configuración Global ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CirqQuoreMind")

# --- Clases de Estado y Métricas ---

class EstadoQubit(Enum):
    GROUND = auto()  # |0⟩
    EXCITED = auto() # |1⟩
    SUPERPOSITION = auto() # α|0⟩ + β|1⟩
    UNKNOWN = auto() # Estado desconocido

@dataclass
class EstadoComplejoCirq:
    """Representación del estado cuántico usando el simulador de Cirq."""
    qubit: cirq.Qubit
    alpha: complex
    beta: complex
    
    @property
    def vector(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], dtype=complex)

    def probabilidad_0(self) -> float:
        return abs(self.alpha)**2

    def probabilidad_1(self) -> float:
        return abs(self.beta)**2

    def fase_relativa(self) -> float:
        if abs(self.alpha) < 1e-9 or abs(self.beta) < 1e-9:
            return 0.0
        return np.angle(self.beta) - np.angle(self.alpha)

    def __str__(self) -> str:
        return f"{self.alpha.real:+.4f}{self.alpha.imag:+.4f}j |0⟩ + {self.beta.real:+.4f}{self.beta.imag:+.4f}j |1⟩ (P0={self.probabilidad_0():.3f})"

@dataclass
class MetricasSistema:
    ciclo: int
    tiempo_coherencia: float
    temperatura: float
    senal_ruido: float
    tasa_error: float
    fotones_perdidos_acum: int
    calidad_transduccion: float
    estado_enlace: Optional[Dict[str, Any]] = None
    voltajes_control: Optional[List[float]] = None

    def __str__(self) -> str:
        return (f"Métricas Ciclo {self.ciclo}: T_coh={self.tiempo_coherencia:.2f}μs, "
                f"Temp={self.temperatura:.2f}mK, SNR={self.senal_ruido:.2f}, "
                f"QBER={self.tasa_error:.4f}, Transd={self.calidad_transduccion:.2f}, "
                f"Fotones Perdidos={self.fotones_perdidos_acum}")

# --- Clases de Componentes Físicos (Simulados con Cirq) ---

class QubitSuperconductorCirq:
    """Modelo de qubit superconductor usando Cirq."""
    def __init__(self, id_qubit: str = "Q0", temp_inicial: float = 15.0, t_coherencia_max: float = 100.0):
        self.id = id_qubit
        self.qubit = cirq.NamedQubit(id_qubit) # Usar NamedQubit de Cirq
        self.tiempo_ultimo_reset = time.time()
        self.t_coherencia_max_base = t_coherencia_max
        self._temperatura = temp_inicial
        self.tiempo_coherencia_max = self._actualizar_coherencia_por_temp(temp_inicial)
        logger.info(f"Qubit {self.id} inicializado, Temp={self._temperatura:.1f}mK, T_coh_max={self.tiempo_coherencia_max:.1f}μs")

    @property
    def temperatura(self) -> float:
        return self._temperatura

    @temperatura.setter
    def temperatura(self, valor: float):
        temp_anterior = self._temperatura
        self._temperatura = max(10.0, min(valor, 50.0))
        if temp_anterior != self._temperatura:
            self.tiempo_coherencia_max = self._actualizar_coherencia_por_temp(self._temperatura)
            logger.debug(f"Qubit {self.id} temp actualizada a {self._temperatura:.1f}mK, T_coh_max={self.tiempo_coherencia_max:.1f}μs")

    def _actualizar_coherencia_por_temp(self, temperatura: float) -> float:
        factor_temp = max(0, 1.0 - ((temperatura - 10.0) / 40.0) * 0.8)
        return self.t_coherencia_max_base * factor_temp

    def tiempo_desde_reset(self) -> float:
        return time.time() - self.tiempo_ultimo_reset

    def aplicar_rotacion(self, eje: str, angulo: float):
        if eje.upper() == 'X':
            operacion = cirq.rx(angulo).on(self.qubit)
        elif eje.upper() == 'Y':
            operacion = cirq.ry(angulo).on(self.qubit)
        elif eje.upper() == 'Z':
            operacion = cirq.rz(angulo).on(self.qubit)
        else:
            raise ValueError(f"Eje de rotación desconocido: {eje}")
        return operacion # Devolver la operación para añadir al circuito

    def aplicar_hadamard(self):
        return cirq.H(self.qubit)

    def aplicar_fase_s(self):
        return cirq.S(self.qubit)

    def reset(self):
        # En Cirq, el reset es una operación específica
        return cirq.reset(self.qubit)

    def simular_decoherencia(self, circuito, tiempo_transcurrido_us: float, t1: float, t2: float):
        # Implementar decoherencia con ruido en Cirq
        # Simplificación: solo añadimos una pequeña rotación aleatoria para simular decoherencia
        if tiempo_transcurrido_us > 0:
            factor_decoherencia = min(1.0, tiempo_transcurrido_us / t2)
            angulo_ruido = factor_decoherencia * np.random.normal(0, 0.1)
            circuito.append(cirq.rx(angulo_ruido).on(self.qubit))
            circuito.append(cirq.ry(angulo_ruido).on(self.qubit))

    def medir(self):
        # En Cirq, la medición es una puerta que se aplica al qubit
        return cirq.measure(self.qubit, key=f"med_{self.id}")

# --- Clases de Operaciones y Control ---

class OperacionCuantica(Enum):
    ROTACION_X = auto()
    ROTACION_Y = auto()
    ROTACION_Z = auto()
    HADAMARD = auto()
    FASE_S = auto()
    RESET = auto()
    MEDICION = auto()

@dataclass
class ParametrosOperacion:
    tipo: OperacionCuantica
    angulo: Optional[float] = None

# --- Clase de Control QuoreMind (Adaptada para Cirq) ---

class QuoreMindCirq:
    def __init__(self):
        self.metricas_historicas: List[MetricasSistema] = []
        self.decisiones_previas: List[ParametrosOperacion] = []
        self.resultados_previos: List[Dict[str, Any]] = []
        self.contador_ciclos = 0
        self.fotones_perdidos_totales = 0
        self.tasa_exito_global = 0.9
        self.estado_sistema = "listo"
        self.factores_decision = {
            "peso_coherencia": 0.4,
            "peso_temperatura": 0.1,
            "peso_snr": 0.2,
            "peso_transduccion": 0.3,
            "umbral_calibracion": 0.7,
            "factor_exploracion": 0.05
        }

    def obtener_metricas_actuales(self, qubit: QubitSuperconductorCirq, canal: 'OpticalChannel', detector: 'PhotonDetector') -> MetricasSistema:
        self.contador_ciclos += 1
        t_coh_max = qubit.tiempo_coherencia_max
        t_desde_reset = qubit.tiempo_desde_reset() * 1e6
        coh_actual = t_coh_max * np.exp(- t_desde_reset / t_coh_max)
        coh_actual = max(5.0, coh_actual)
        temp_actual = qubit.temperatura + np.random.normal(0, 0.3)
        temp_actual = max(10.0, min(50.0, temp_actual))
        qubit.temperatura = temp_actual
        calidad_trans = 0.98 - ((temp_actual - 10.0) / 40.0)**1.5 * 0.5
        calidad_trans = max(0.4, min(0.99, calidad_trans))
        snr_base = 28.0
        snr_actual = snr_base * calidad_trans * (coh_actual / t_coh_max) + np.random.normal(0, 1.0)
        snr_actual = max(8.0, min(35.0, snr_actual))
        qber = 0.005 + (1.0 - calidad_trans)*0.03 + (35.0 - snr_actual)/50.0 * 0.04 + canal.perdida_acumulada * 0.01
        qber = max(0.0005, min(0.15, qber))
        self.fotones_perdidos_totales = canal.fotones_perdidos

        metricas = MetricasSistema(
            ciclo=self.contador_ciclos,
            tiempo_coherencia=coh_actual,
            temperatura=temp_actual,
            senal_ruido=snr_actual,
            tasa_error=qber,
            fotones_perdidos_acum=self.fotones_perdidos_totales,
            calidad_transduccion=calidad_trans
        )
        self.metricas_historicas.append(metricas)
        return metricas

    def _calcular_cosenos_directores(self, metricas: MetricasSistema) -> Tuple[float, float, float]:
        norm_coh = np.clip(metricas.tiempo_coherencia / 100.0, 0, 1)
        norm_temp_inv = np.clip(1.0 - (metricas.temperatura - 10.0) / 40.0, 0, 1)
        norm_snr = np.clip((metricas.senal_ruido - 10.0) / 25.0, 0, 1)
        norm_trans = np.clip(metricas.calidad_transduccion, 0, 1)
        comp_x = norm_coh * 0.6 + norm_snr * 0.4
        comp_y = norm_trans * 0.7 + norm_snr * 0.3
        comp_z = norm_temp_inv * 0.5 + norm_coh * 0.5
        magnitud = math.sqrt(comp_x**2 + comp_y**2 + comp_z**2)
        return (comp_x / magnitud, comp_y / magnitud, comp_z / magnitud) if magnitud > 1e-9 else (1.0, 0.0, 0.0)

    def calcular_angulo_control(self, metricas: MetricasSistema) -> float:
        base_angle = math.pi / 2
        factor_coh = np.clip(metricas.tiempo_coherencia / 50.0, 0.5, 1.2)
        factor_qual = np.clip(metricas.calidad_transduccion, 0.5, 1.1)
        angulo = base_angle * factor_coh * factor_qual
        if np.random.random() < self.factores_decision["factor_exploracion"]:
            angulo += np.random.normal(0, math.pi / 16)
        return np.clip(angulo, 0, math.pi)

    def decidir_operacion(self, metricas: MetricasSistema) -> ParametrosOperacion:
        calidad_general = (
            self.factores_decision["peso_coherencia"] * np.clip(metricas.tiempo_coherencia / 80.0, 0, 1) +
            self.factores_decision["peso_snr"] * np.clip(metricas.senal_ruido / 30.0, 0, 1) +
            self.factores_decision["peso_transduccion"] * metricas.calidad_transduccion
        ) / (self.factores_decision["peso_coherencia"] + self.factores_decision["peso_snr"] + self.factores_decision["peso_transduccion"])

        if calidad_general < self.factores_decision["umbral_calibracion"]:
            logger.warning(f"Calidad general baja ({calidad_general:.3f}), considerando RESET.")
            if np.random.random() < 0.8:
                self.estado_sistema = "calibrando"
                return ParametrosOperacion(tipo=OperacionCuantica.RESET)

        self.estado_sistema = "operando"
        cos_x, cos_y, cos_z = self._calcular_cosenos_directores(metricas)
        angulo = self.calcular_angulo_control(metricas)

        ejes = {'X': abs(cos_x), 'Y': abs(cos_y), 'Z': abs(cos_z)}
        eje_elegido = max(ejes, key=ejes.get)

        if eje_elegido == 'X':
            operacion_tipo = OperacionCuantica.ROTACION_X
        elif eje_elegido == 'Y':
            operacion_tipo = OperacionCuantica.ROTACION_Y
        else:
            operacion_tipo = OperacionCuantica.ROTACION_Z

        if calidad_general > 0.85 and np.random.random() < 0.15:
            operacion_tipo = np.random.choice([OperacionCuantica.HADAMARD, OperacionCuantica.FASE_S])
            angulo = None

        params = ParametrosOperacion(tipo=operacion_tipo, angulo=angulo)
        self.decisiones_previas.append(params)
        return params

    def actualizar_aprendizaje(self, resultado_ciclo: Dict[str, Any]):
        self.resultados_previos.append(resultado_ciclo)
        exito = resultado_ciclo.get("exito_deteccion", False) and not resultado_ciclo.get("error_medicion", True)
        historial = min(len(self.resultados_previos), 20)
        exitos_recientes = sum(1 for r in self.resultados_previos[-historial:]
                               if r.get("exito_deteccion", False) and not r.get("error_medicion", True))
        tasa_exito_reciente = exitos_recientes / historial if historial > 0 else 0.0
        self.tasa_exito_global = 0.95 * self.tasa_exito_global + 0.05 * tasa_exito_reciente
        if tasa_exito_reciente < 0.6 and len(self.resultados_previos) > 10:
            self.factores_decision["factor_exploracion"] = min(0.3, self.factores_decision["factor_exploracion"] + 0.01)
        elif tasa_exito_reciente > 0.85:
            self.factores_decision["factor_exploracion"] = max(0.02, self.factores_decision["factor_exploracion"] - 0.005)

    def registrar_evento(self, evento: str):
        logger.info(f"EVENTO Ciclo {self.contador_ciclos}: {evento}")

    def registrar_error(self, error: str):
        logger.error(f"ERROR Ciclo {self.contador_ciclos}: {error}")

    def obtener_estado_sistema(self) -> Dict[str, Any]:
        return {
            "ciclo": self.contador_ciclos,
            "tasa_exito_global": self.tasa_exito_global,
            "factor_exploracion": self.factores_decision["factor_exploracion"],
            "estado_operativo": self.estado_sistema,
            "ultima_metrica": str(self.metricas_historicas[-1]) if self.metricas_historicas else "N/A",
            "ultima_decision": self.decisiones_previas[-1].tipo.name if self.decisiones_previas else "N/A"
        }

# --- Clase de Control de Microondas (Simulada para Cirq) ---

class MicrowaveControlCirq:
    def __init__(self):
        self.frecuencia_base = 5.1 # GHz
        self.precision_tiempo = 0.1 # ns
        self.latencia_aplicacion = 5 # ns
        self.calibracion = {"offset_frecuencia": 0.0, "factor_amplitud": 1.0, "offset_fase": 0.0}
        logger.info("MicrowaveControl inicializado (para Cirq).")

    def traducir_operacion_a_pulso(self, operacion: ParametrosOperacion) -> Dict[str, Any]:
        params = {
            "tipo_operacion": operacion.tipo.name,
            "angulo_logico": operacion.angulo,
            "duracion": 15.0, # ns
            "amplitud": 0.95,
            "frecuencia": self.frecuencia_base + self.calibracion["offset_frecuencia"],
            "fase": self.calibracion["offset_fase"],
            "forma": "gaussiana_derivada"
        }
        return params

    def aplicar_pulso(self, qubit: QubitSuperconductorCirq, params: Dict[str, Any]) -> str:
        time.sleep(self.latencia_aplicacion * 1e-9)
        tipo_op = params.get("tipo_operacion", "")
        resultado_op = f"Pulso de {tipo_op} aplicado (simulado para Cirq)."
        return resultado_op

# --- Clases de Componentes Ópticos (Simulados) ---

@dataclass
class EstadoFoton:
    polarizacion: float  # Ángulo en radianes
    fase: float  # Fase en radianes
    valido: bool = True

    def __str__(self) -> str:
        return f"Fotón[pol={math.degrees(self.polarizacion):.1f}°, fase={math.degrees(self.fase):.1f}°]"

class TransductorSQaOptico:
    def __init__(self, eficiencia_base: float = 0.8):
        self.eficiencia_conversion = eficiencia_base
        self.ruido_fase_polarizacion = 0.05
        logger.info(f"Transductor SQ->Óptico inicializado. Eficiencia base: {self.eficiencia_conversion:.2f}")

    def mapear_bloch_a_foton(self, bloch_vector: np.ndarray) -> EstadoFoton:
        # Mapear vector de Bloch a estado de fotón
        # Asumiendo: x, y, z son coordenadas del vector de Bloch
        # Polarización: ángulo en plano xy
        # Fase: ángulo desde eje x
        
        x, y, z = bloch_vector
        polarizacion = math.acos(z)  # Mapeo del componente z a ángulo de polarización
        fase = math.atan2(y, x)  # Fase desde la proyección en plano xy
        
        # Agregar ruido
        polarizacion = np.clip(polarizacion + np.random.normal(0, self.ruido_fase_polarizacion), 0, math.pi)
        fase = (fase + np.random.normal(0, self.ruido_fase_polarizacion)) % (2 * math.pi)
        
        return EstadoFoton(polarizacion=polarizacion, fase=fase)

    def obtener_vector_bloch(self, simulacion_result: Any, qubit: cirq.Qubit) -> np.ndarray:
        # Obtener amplitudes complejas del estado del qubit
        # En una simulación real, se extraería del resultado de la simulación de Cirq
        # Para simplificar, generamos un vector de Bloch aleatorio para demostración
        
        # Vector aleatorio de longitud 1
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        r = math.sqrt(x*x + y*y + z*z)
        return np.array([x/r, y/r, z/r])

    def modular_foton(self, estado_foton: EstadoFoton) -> Optional[EstadoFoton]:
        if np.random.random() > self.eficiencia_conversion:
            logger.warning("Fallo en transducción SQ -> Óptico.")
            return None
        logger.debug(f"Fotón modulado con estado: {estado_foton}")
        return estado_foton

class OpticalChannel:
    def __init__(self, longitud_km: float = 1.0, atenuacion_db_km: float = 0.2):
        self.longitud = longitud_km
        self.atenuacion_db_km = atenuacion_db_km
        self.prob_perdida = 1.0 - 10**(-(self.longitud * self.atenuacion_db_km) / 10.0)
        self.latencia_ns = longitud_km * 5000
        self.fotones_perdidos = 0
        self.perdida_acumulada = 0.0
        logger.info(f"Canal Óptico inicializado: {longitud_km}km, Aten={atenuacion_db_km}dB/km -> Prob Pérdida={self.prob_perdida:.3f}, Latencia={self.latencia_ns:.0f}ns")

    def enviar_foton(self, foton: Optional[EstadoFoton]) -> Optional[EstadoFoton]:
        time.sleep(self.latencia_ns * 1e-9)
        if foton is None:
            self.fotones_perdidos += 1
            self.perdida_acumulada += 0.01
            logger.warning("Intento de enviar fotón NULO (fallo modulación previa).")
            return None
        if np.random.random() < self.prob_perdida:
            self.fotones_perdidos += 1
            self.perdida_acumulada += 0.005
            logger.warning("Fotón perdido en el canal.")
            return None
        logger.debug(f"Fotón transmitido: {foton}")
        return foton

class PhotonDetector:
    def __init__(self, eficiencia: float = 0.9):
        self.eficiencia_deteccion = eficiencia
        logger.info(f"Detector de fotones inicializado. Eficiencia: {self.eficiencia_deteccion:.2f}")
        
    def detectar_foton(self, foton: Optional[EstadoFoton]) -> Dict[str, Any]:
        if foton is None:
            logger.info("No se detectó ningún fotón (perdido en el canal).")
            return {"polarizacion": None, "fase": None, "ausente": True, "exito_deteccion": False}
            
        if np.random.random() > self.eficiencia_deteccion:
            logger.warning("Fotón no detectado (falló la detección).")
            return {"polarizacion": None, "fase": None, "ausente": True, "exito_deteccion": False}
            
        if not foton.valido:
            logger.warning("Fotón inválido detectado.")
            return {"polarizacion": None, "fase": None, "invalido": True, "exito_deteccion": False}
            
        polarizacion_deg = math.degrees(foton.polarizacion)
        fase_deg = math.degrees(foton.fase)
        logger.info(f"Fotón detectado: Pol={polarizacion_deg:.1f}°, Fase={fase_deg:.1f}°")
        
        return {
            "polarizacion": polarizacion_deg, 
            "fase": fase_deg, 
            "exito_deteccion": True,
            "error_medicion": np.random.random() < 0.05  # 5% de probabilidad de error en la medición
        }

# --- Simulación Principal con Cirq ---
def run_quore_mind_cirq_simulation(num_cycles=5):
    # Inicializar componentes
    qubit = QubitSuperconductorCirq()
    canal = OpticalChannel()
    detector = PhotonDetector()
    control = QuoreMindCirq()
    microwave_control = MicrowaveControlCirq()
    
    # Simulador de Cirq (único para toda la simulación)
    simulator = cirq.Simulator()
    
    # Estado inicial del qubit (|0⟩)
    estado_inicial = cirq.Circuit(cirq.I(qubit.qubit))
    resultado_inicial = simulator.simulate(estado_inicial)
    
    for cycle in range(num_cycles):
        logger.info(f"\n--- Ciclo {cycle + 1} ---")

        # 1. Decisión Inteligente (QuoreMind)
        metricas = control.obtener_metricas_actuales(qubit, canal, detector)
        logger.info(f"Métricas: {metricas}")
        operacion = control.decidir_operacion(metricas)
        logger.info(f"Decisión: {operacion.tipo.name}, Ángulo: {operacion.angulo}")

        # 2. Configuración del circuito para este ciclo
        circuito = cirq.Circuit()
        
        # Simular decoherencia
        tiempo_transcurrido_us = qubit.tiempo_desde_reset() * 1e6
        t1 = qubit.tiempo_coherencia_max * 1.5
        t2 = qubit.tiempo_coherencia_max
        qubit.simular_decoherencia(circuito, tiempo_transcurrido_us, t1, t2)
        
        # Configurar operación según decisión
        if operacion.tipo == OperacionCuantica.ROTACION_X and operacion.angulo is not None:
            circuito.append(qubit.aplicar_rotacion('X', operacion.angulo))
        elif operacion.tipo == OperacionCuantica.ROTACION_Y and operacion.angulo is not None:
            circuito.append(qubit.aplicar_rotacion('Y', operacion.angulo))
        elif operacion.tipo == OperacionCuantica.ROTACION_Z and operacion.angulo is not None:
            circuito.append(qubit.aplicar_rotacion('Z', operacion.angulo))
        elif operacion.tipo == OperacionCuantica.HADAMARD:
            circuito.append(qubit.aplicar_hadamard())
        elif operacion.tipo == OperacionCuantica.FASE_S:
            circuito.append(qubit.aplicar_fase_s())
        elif operacion.tipo == OperacionCuantica.RESET:
            circuito.append(qubit.reset())
            qubit.tiempo_ultimo_reset = time.time()  # Actualizar tiempo de reset
        elif operacion.tipo == OperacionCuantica.MEDICION:
            circuito.append(qubit.medir())

        # 3. Simular el circuito
        pulso = microwave_control.traducir_operacion_a_pulso(operacion)
        microwave_control.aplicar_pulso(qubit, pulso)
        
        if not circuito.moments:  # Si el circuito está vacío, agregar operación identidad
            circuito.append(cirq.I(qubit.qubit))
            
        # Realizar la simulación
        resultado_simulacion = simulator.simulate(circuito)
        logger.info(f"Estado después de operación: {resultado_simulacion.final_state_vector}")
        
        # 4. Transducción a fotón
        transductor = TransductorSQaOptico()
        vector_bloch = transductor.obtener_vector_bloch(resultado_simulacion, qubit.qubit)
        foton = transductor.mapear_bloch_a_foton(vector_bloch)
        foton_modulado = transductor.modular_foton(foton)
        
        # 5. Transmisión y detección
        foton_transmitido = canal.enviar_foton(foton_modulado)
        deteccion = detector.detectar_foton(foton_transmitido)
        logger.info(f"Detección: {deteccion}")
        
        # 6. Retroalimentación (QuoreMind)
        control.actualizar_aprendizaje(deteccion)
        logger.info(f"Estado del Sistema: {control.obtener_estado_sistema()}")
        
        # Pequeña pausa entre ciclos para la visualización
        time.sleep(0.5)
    
    logger.info("\n--- Simulación completada ---")
    logger.info(f"Ciclos totales: {control.contador_ciclos}")
    logger.info(f"Tasa de éxito global: {control.tasa_exito_global:.4f}")
    logger.info(f"Fotones perdidos: {canal.fotones_perdidos}")
    
    return control.metricas_historicas

if __name__ == "__main__":
    print("Iniciando simulación QuoreMind con Cirq...")
    metricas = run_quore_mind_cirq_simulation(num_cycles=10)
    print("Simulación finalizada.")
