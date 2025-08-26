import logging
import numpy as np
import sympy as sp
from sympy import symbols, I, sqrt, exp, pi, simplify, expand, Matrix
from sympy.physics.quantum import Dagger, qapply
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict
import cirq
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings

# --- Configuración Global ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CirqQuoreMind")

# Suprimir advertencias de SymPy para mejor visualización
warnings.filterwarnings("ignore", category=UserWarning, module="sympy")

# --- Enumeraciones y Constantes ---

class EstadoQubit(Enum):
    """Estados básicos de un qubit con representaciones matemáticas exactas."""
    GROUND = auto()           # |0⟩
    EXCITED = auto()          # |1⟩ 
    SUPERPOSITION = auto()    # α|0⟩ + β|1⟩
    ENTANGLED = auto()        # Estados entrelazados
    MIXED = auto()            # Estados mixtos (matriz densidad)
    UNKNOWN = auto()          # Estado desconocido

class BaseEstados:
    """Constantes para estados cuánticos fundamentales usando SymPy."""
    # Símbolos simbólicos
    alpha, beta, gamma, delta = symbols('alpha beta gamma delta', complex=True)
    theta, phi = symbols('theta phi', real=True)
    t = symbols('t', real=True, positive=True)
    
    # Estados base computacionales
    ZERO = Matrix([1, 0])
    ONE = Matrix([0, 1])
    PLUS = (Matrix([1, 0]) + Matrix([0, 1])) / sqrt(2)
    MINUS = (Matrix([1, 0]) - Matrix([0, 1])) / sqrt(2)
    
    # Matrices de Pauli simbólicas
    PAULI_X = Matrix([[0, 1], [1, 0]])
    PAULI_Y = Matrix([[0, -I], [I, 0]])
    PAULI_Z = Matrix([[1, 0], [0, -1]])
    IDENTITY = Matrix([[1, 0], [0, 1]])

# --- Clase Principal de Estado Cuántico ---

@dataclass
class EstadoComplejoCirq:
    """Representación avanzada del estado cuántico con matemática simbólica y numérica."""
    qubit: cirq.Qubit
    alpha: Union[complex, sp.Expr] = field(default=1.0)
    beta: Union[complex, sp.Expr] = field(default=0.0)
    symbolic: bool = field(default=False)
    normalize_on_init: bool = field(default=True)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Inicialización automática y normalización."""
        if self.symbolic:
            self.alpha = sp.sympify(self.alpha)
            self.beta = sp.sympify(self.beta)
            if self.normalize_on_init:
                self._normalize_symbolic()
        else:
            self.alpha = complex(self.alpha)
            self.beta = complex(self.beta)
            if self.normalize_on_init:
                self._normalize_numeric()

    def _normalize_numeric(self):
        """Normalización numérica del estado."""
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 1e-15:
            self.alpha /= norm
            self.beta /= norm

    def _normalize_symbolic(self):
        """Normalización simbólica del estado."""
        norm_squared = self.alpha * sp.conjugate(self.alpha) + self.beta * sp.conjugate(self.beta)
        norm = sp.sqrt(norm_squared)
        if norm != 0:
            self.alpha = simplify(self.alpha / norm)
            self.beta = simplify(self.beta / norm)

    @property
    def vector(self) -> Union[np.ndarray, Matrix]:
        """Vector de estado como numpy array o matriz simbólica."""
        if self.symbolic:
            return Matrix([self.alpha, self.beta])
        return np.array([self.alpha, self.beta], dtype=complex)

    @property
    def density_matrix(self) -> Union[np.ndarray, Matrix]:
        """Matriz densidad del estado puro."""
        if self.symbolic:
            psi = self.vector
            return psi * Dagger(psi)
        else:
            psi = self.vector.reshape(-1, 1)
            return psi @ psi.conj().T

    def probabilidad_0(self) -> Union[float, sp.Expr]:
        """Probabilidad de medir |0⟩."""
        if self.symbolic:
            return simplify(self.alpha * sp.conjugate(self.alpha))
        return abs(self.alpha)**2

    def probabilidad_1(self) -> Union[float, sp.Expr]:
        """Probabilidad de medir |1⟩."""
        if self.symbolic:
            return simplify(self.beta * sp.conjugate(self.beta))
        return abs(self.beta)**2

    def fase_relativa(self) -> Union[float, sp.Expr]:
        """Fase relativa entre los coeficientes."""
        if self.symbolic:
            if self.alpha == 0 or self.beta == 0:
                return 0
            return sp.arg(self.beta) - sp.arg(self.alpha)
        else:
            if abs(self.alpha) < 1e-9 or abs(self.beta) < 1e-9:
                return 0.0
            return np.angle(self.beta) - np.angle(self.alpha)

    def fase_global(self) -> Union[float, sp.Expr]:
        """Fase global del estado."""
        if self.symbolic:
            return sp.arg(self.alpha) if self.alpha != 0 else sp.arg(self.beta)
        else:
            return np.angle(self.alpha) if abs(self.alpha) > 1e-9 else np.angle(self.beta)

    def pureza(self) -> Union[float, sp.Expr]:
        """Pureza del estado (1 para estados puros)."""
        rho = self.density_matrix
        if self.symbolic:
            return simplify(sp.trace(rho * rho))
        else:
            return np.real(np.trace(rho @ rho))

    def entropia_von_neumann(self) -> Union[float, sp.Expr]:
        """Entropía de von Neumann del estado."""
        if self.symbolic:
            # Para estados puros simbólicos
            return 0
        else:
            rho = self.density_matrix
            eigenvals = np.linalg.eigvals(rho)
            eigenvals = eigenvals[eigenvals > 1e-15]  # Evitar log(0)
            return -np.sum(eigenvals * np.log2(eigenvals))

    def fidelidad(self, otro_estado: 'EstadoComplejoCirq') -> Union[float, sp.Expr]:
        """Fidelidad entre dos estados cuánticos."""
        if self.symbolic and otro_estado.symbolic:
            psi1 = self.vector
            psi2 = otro_estado.vector
            overlap = simplify(Dagger(psi1) * psi2)
            return simplify(overlap * sp.conjugate(overlap))
        else:
            # Convertir a numérico si es necesario
            v1 = self.to_numeric().vector
            v2 = otro_estado.to_numeric().vector
            overlap = np.vdot(v1, v2)
            return abs(overlap)**2

    def distancia_traza(self, otro_estado: 'EstadoComplejoCirq') -> Union[float, sp.Expr]:
        """Distancia de traza entre dos estados."""
        if self.symbolic and otro_estado.symbolic:
            rho1 = self.density_matrix
            rho2 = otro_estado.density_matrix
            diff = rho1 - rho2
            # Para estados puros, la distancia de traza es 2*sqrt(1-fidelidad)
            fid = self.fidelidad(otro_estado)
            return 2 * sp.sqrt(1 - fid)
        else:
            rho1 = self.to_numeric().density_matrix
            rho2 = otro_estado.to_numeric().density_matrix
            diff = rho1 - rho2
            eigenvals = np.linalg.eigvals(diff @ diff.conj().T)
            return np.sqrt(np.sum(np.real(eigenvals)))

    def aplicar_operador(self, operador: Union[np.ndarray, Matrix]) -> 'EstadoComplejoCirq':
        """Aplica un operador unitario al estado."""
        if self.symbolic:
            if isinstance(operador, np.ndarray):
                operador = Matrix(operador)
            nuevo_vector = operador * self.vector
            return EstadoComplejoCirq(
                qubit=self.qubit,
                alpha=nuevo_vector[0],
                beta=nuevo_vector[1],
                symbolic=True,
                normalize_on_init=True
            )
        else:
            if isinstance(operador, Matrix):
                operador = np.array(operador.evalf(), dtype=complex)
            nuevo_vector = operador @ self.vector
            return EstadoComplejoCirq(
                qubit=self.qubit,
                alpha=nuevo_vector[0],
                beta=nuevo_vector[1],
                symbolic=False,
                normalize_on_init=True
            )

    def evolucion_temporal(self, hamiltoniano: Union[np.ndarray, Matrix], 
                          tiempo: Union[float, sp.Expr]) -> 'EstadoComplejoCirq':
        """Evolución temporal bajo un Hamiltoniano dado."""
        if self.symbolic:
            if isinstance(hamiltoniano, np.ndarray):
                hamiltoniano = Matrix(hamiltoniano)
            # Operador de evolución U = exp(-i*H*t)
            U = sp.exp(-I * hamiltoniano * tiempo)
            return self.aplicar_operador(U)
        else:
            if isinstance(hamiltoniano, Matrix):
                hamiltoniano = np.array(hamiltoniano.evalf(), dtype=complex)
            U = expm(-1j * hamiltoniano * float(tiempo))
            return self.aplicar_operador(U)

    def to_numeric(self) -> 'EstadoComplejoCirq':
        """Convierte estado simbólico a numérico."""
        if not self.symbolic:
            return self
        
        alpha_num = complex(self.alpha.evalf())
        beta_num = complex(self.beta.evalf())
        
        return EstadoComplejoCirq(
            qubit=self.qubit,
            alpha=alpha_num,
            beta=beta_num,
            symbolic=False,
            normalize_on_init=True,
            metadata=self.metadata.copy()
        )

    def to_symbolic(self) -> 'EstadoComplejoCirq':
        """Convierte estado numérico a simbólico."""
        if self.symbolic:
            return self
        
        return EstadoComplejoCirq(
            qubit=self.qubit,
            alpha=sp.sympify(self.alpha),
            beta=sp.sympify(self.beta),
            symbolic=True,
            normalize_on_init=True,
            metadata=self.metadata.copy()
        )

    def clasificar_estado(self) -> EstadoQubit:
        """Clasifica el tipo de estado cuántico."""
        if self.symbolic:
            p0 = self.probabilidad_0()
            p1 = self.probabilidad_1()
            
            if p0 == 1:
                return EstadoQubit.GROUND
            elif p1 == 1:
                return EstadoQubit.EXCITED
            else:
                return EstadoQubit.SUPERPOSITION
        else:
            p0 = self.probabilidad_0()
            p1 = self.probabilidad_1()
            
            if abs(p0 - 1) < 1e-10:
                return EstadoQubit.GROUND
            elif abs(p1 - 1) < 1e-10:
                return EstadoQubit.EXCITED
            else:
                return EstadoQubit.SUPERPOSITION

    def representacion_bloch(self) -> Tuple[Union[float, sp.Expr], Union[float, sp.Expr], Union[float, sp.Expr]]:
        """Coordenadas en la esfera de Bloch (x, y, z)."""
        if self.symbolic:
            # Coordenadas simbólicas de Bloch
            x = self.alpha * sp.conjugate(self.beta) + sp.conjugate(self.alpha) * self.beta
            y = I * (sp.conjugate(self.alpha) * self.beta - self.alpha * sp.conjugate(self.beta))
            z = self.alpha * sp.conjugate(self.alpha) - self.beta * sp.conjugate(self.beta)
            return simplify(x), simplify(y), simplify(z)
        else:
            x = 2 * np.real(np.conj(self.alpha) * self.beta)
            y = 2 * np.imag(np.conj(self.alpha) * self.beta)
            z = abs(self.alpha)**2 - abs(self.beta)**2
            return x, y, z

    def visualizar_bloch(self):
        """Visualiza el estado en la esfera de Bloch."""
        if self.symbolic:
            print("Visualización requiere conversión a valores numéricos")
            estado_num = self.to_numeric()
            return estado_num.visualizar_bloch()
        
        x, y, z = self.representacion_bloch()
        
        # Crear gráfico 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dibujar esfera de Bloch
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.1, color='lightblue')
        
        # Dibujar ejes
        ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
        
        # Dibujar vector de estado
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
        
        # Etiquetas
        ax.text(1.1, 0, 0, 'X', fontsize=12)
        ax.text(0, 1.1, 0, 'Y', fontsize=12)
        ax.text(0, 0, 1.1, 'Z', fontsize=12)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Estado en Esfera de Bloch\n{str(self)}')
        
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Representación en cadena del estado."""
        if self.symbolic:
            return f"({self.alpha})|0⟩ + ({self.beta})|1⟩"
        else:
            alpha_str = f"{self.alpha.real:+.4f}{self.alpha.imag:+.4f}j"
            beta_str = f"{self.beta.real:+.4f}{self.beta.imag:+.4f}j"
            p0 = self.probabilidad_0()
            return f"{alpha_str}|0⟩ + {beta_str}|1⟩ (P₀={p0:.3f})"

    def __repr__(self) -> str:
        return f"EstadoComplejoCirq(α={self.alpha}, β={self.beta}, symbolic={self.symbolic})"

# --- Funciones de Utilidad ---

def crear_estado_bell(tipo: str = "phi_plus") -> Tuple[EstadoComplejoCirq, EstadoComplejoCirq]:
    """Crea estados de Bell entrelazados."""
    q1, q2 = cirq.LineQubit.range(2)
    
    estados_bell = {
        "phi_plus": (1/np.sqrt(2), 0, 0, 1/np.sqrt(2)),    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        "phi_minus": (1/np.sqrt(2), 0, 0, -1/np.sqrt(2)),   # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
        "psi_plus": (0, 1/np.sqrt(2), 1/np.sqrt(2), 0),     # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
        "psi_minus": (0, 1/np.sqrt(2), -1/np.sqrt(2), 0),   # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    }
    
    if tipo not in estados_bell:
        raise ValueError(f"Tipo debe ser uno de: {list(estados_bell.keys())}")
    
    coeff = estados_bell[tipo]
    
    # Para estados de Bell, necesitamos representación de dos qubits
    # Aquí simplificamos mostrando los coeficientes principales
    estado1 = EstadoComplejoCirq(q1, alpha=coeff[0], beta=coeff[1])
    estado2 = EstadoComplejoCirq(q2, alpha=coeff[2], beta=coeff[3])
    
    return estado1, estado2

def optimizar_estado(objetivo_vector: np.ndarray, 
                    estado_inicial: EstadoComplejoCirq,
                    operadores_disponibles: List[np.ndarray]) -> EstadoComplejoCirq:
    """Optimiza una secuencia de operadores para alcanzar un estado objetivo."""
    
    def costo(parametros):
        estado_actual = estado_inicial.to_numeric()
        for i, param in enumerate(parametros):
            if i < len(operadores_disponibles):
                U = expm(-1j * param * operadores_disponibles[i])
                estado_actual = estado_actual.aplicar_operador(U)
        
        fidelidad = abs(np.vdot(estado_actual.vector, objetivo_vector))**2
        return 1 - fidelidad  # Minimizar (1 - fidelidad)
    
    # Optimización
    resultado = minimize(costo, x0=np.random.random(len(operadores_disponibles)), 
                        method='BFGS')
    
    # Aplicar resultado óptimo
    estado_optimizado = estado_inicial.to_numeric()
    for i, param in enumerate(resultado.x):
        if i < len(operadores_disponibles):
            U = expm(-1j * param * operadores_disponibles[i])
            estado_optimizado = estado_optimizado.aplicar_operador(U)
    
    return estado_optimizado

# --- Nueva Clase para Matriz de Densidad ---
@dataclass
class MatrizDensidadCirq:
    """Representación avanzada de una matriz de densidad de un qubit."""
    qubit: cirq.Qubit
    rho: Union[np.ndarray, Matrix] = field(default_factory=lambda: np.eye(2, dtype=complex) / 2)
    symbolic: bool = False
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Inicialización automática."""
        if self.symbolic:
            if isinstance(self.rho, np.ndarray):
                self.rho = Matrix(self.rho)
            self._validate_symbolic()
        else:
            if isinstance(self.rho, Matrix):
                self.rho = np.array(self.rho.evalf(), dtype=complex)
            self._validate_numeric()
        
        # Asegurarse de que la matriz es de 2x2
        if self.rho.shape != (2, 2):
            raise ValueError("La matriz de densidad debe ser de 2x2.")

    def _validate_numeric(self):
        """Valida propiedades de la matriz de densidad numérica."""
        # Normalizar traza
        trace = np.trace(self.rho)
        if abs(trace - 1.0) > 1e-10:
            logger.warning(f"Normalizando matriz de densidad (traza = {trace:.6f})")
            self.rho = self.rho / trace
        
        # Verificar hermiticidad
        if not np.allclose(self.rho, self.rho.conj().T, rtol=1e-12):
            logger.warning("Matriz de densidad no es hermítica")
        
        # Verificar semidefinida positiva
        eigenvals = np.linalg.eigvals(self.rho)
        if np.any(eigenvals < -1e-12):
            logger.warning("Matriz de densidad tiene eigenvalores negativos")

    def _validate_symbolic(self):
        """Valida propiedades de la matriz de densidad simbólica."""
        # Para validación simbólica, verificamos solo la estructura básica
        trace = sp.trace(self.rho)
        if trace != 1 and trace != sp.sympify(1):
            self.rho = self.rho / trace

    @classmethod
    def from_estado_puro(cls, estado: EstadoComplejoCirq) -> 'MatrizDensidadCirq':
        """Crea matriz de densidad desde un estado puro."""
        rho = estado.density_matrix
        return cls(estado.qubit, rho, symbolic=estado.symbolic)

    @classmethod
    def mezcla_estadistica(cls, estados: List[EstadoComplejoCirq], 
                          probabilidades: List[Union[float, sp.Expr]],
                          qubit: cirq.Qubit) -> 'MatrizDensidadCirq':
        """Crea una mezcla estadística de estados puros."""
        if len(estados) != len(probabilidades):
            raise ValueError("Número de estados y probabilidades debe coincidir")
        
        # Determinar si es simbólico
        symbolic = any(e.symbolic for e in estados) or any(isinstance(p, sp.Expr) for p in probabilidades)
        
        if symbolic:
            rho_total = Matrix([[0, 0], [0, 0]])
            for estado, prob in zip(estados, probabilidades):
                rho_i = estado.to_symbolic().density_matrix
                rho_total += prob * rho_i
        else:
            rho_total = np.zeros((2, 2), dtype=complex)
            for estado, prob in zip(estados, probabilidades):
                rho_i = estado.to_numeric().density_matrix
                rho_total += float(prob) * rho_i
        
        return cls(qubit, rho_total, symbolic=symbolic)

    def evolucion_lindblad(self, hamiltoniano: Union[np.ndarray, Matrix], 
                          lindblad_ops: List[Tuple[Union[np.ndarray, Matrix], float]], 
                          tiempo: float, pasos: int = 100) -> 'MatrizDensidadCirq':
        """
        Evoluciona la matriz de densidad usando la ecuación de Lindblad.
        
        La ecuación de Lindblad describe la evolución de sistemas abiertos:
        drho/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k† - 1/2{L_k†L_k, rho})
        
        Args:
            hamiltoniano: Operador Hamiltoniano del sistema.
            lindblad_ops: Lista de tuplas (Operador L_k, Tasa gamma_k).
            tiempo: Tiempo total de la evolución.
            pasos: Número de pasos de tiempo para la integración.
        """
        if self.symbolic:
            return self._evolucion_lindblad_simbolica(hamiltoniano, lindblad_ops, tiempo)
        
        # Asegurarse de que el Hamiltoniano es un array de numpy
        if isinstance(hamiltoniano, Matrix):
            hamiltoniano = np.array(hamiltoniano.evalf(), dtype=complex)

        dt = tiempo / pasos
        rho_t = self.rho.copy()

        for _ in range(pasos):
            # Término unitario (Hamiltoniano): -i[H, rho]
            H_comm = -1j * (hamiltoniano @ rho_t - rho_t @ hamiltoniano)

            # Término de decoherencia (Lindblad)
            L_term = np.zeros_like(rho_t, dtype=complex)
            for L_op, gamma in lindblad_ops:
                if isinstance(L_op, Matrix):
                    L_op = np.array(L_op.evalf(), dtype=complex)
                L_dag = L_op.conj().T
                
                # Término de Lindblad: gamma * (L rho L† - 1/2{L†L, rho})
                dissipator = (L_op @ rho_t @ L_dag - 
                            0.5 * (L_dag @ L_op @ rho_t + rho_t @ L_dag @ L_op))
                L_term += gamma * dissipator
            
            # Actualización de la matriz de densidad
            rho_t = rho_t + dt * (H_comm + L_term)
            
            # Normalización para conservar la traza
            rho_t = rho_t / np.trace(rho_t)

        return MatrizDensidadCirq(self.qubit, rho_t, symbolic=False)

    def _evolucion_lindblad_simbolica(self, hamiltoniano: Matrix, 
                                    lindblad_ops: List[Tuple[Matrix, sp.Expr]], 
                                    tiempo: sp.Expr) -> 'MatrizDensidadCirq':
        """Versión simbólica de la evolución de Lindblad (aproximación primer orden)."""
        if isinstance(hamiltoniano, np.ndarray):
            hamiltoniano = Matrix(hamiltoniano)
        
        # Término Hamiltoniano
        H_comm = -I * (hamiltoniano * self.rho - self.rho * hamiltoniano)
        
        # Términos de Lindblad
        L_term = Matrix([[0, 0], [0, 0]])
        for L_op, gamma in lindblad_ops:
            if isinstance(L_op, np.ndarray):
                L_op = Matrix(L_op)
            L_dag = Dagger(L_op)
            dissipator = L_op * self.rho * L_dag - sp.Rational(1, 2) * (L_dag * L_op * self.rho + self.rho * L_dag * L_op)
            L_term += gamma * dissipator
        
        # Evolución de primer orden: rho(t) ≈ rho(0) + t * drho/dt
        rho_evolved = self.rho + tiempo * (H_comm + L_term)
        rho_evolved = simplify(rho_evolved)
        
        return MatrizDensidadCirq(self.qubit, rho_evolved, symbolic=True)

    def evolucion_unitaria(self, operador_U: Union[np.ndarray, Matrix]) -> 'MatrizDensidadCirq':
        """Evolución unitaria: rho' = U rho U†"""
        if self.symbolic:
            if isinstance(operador_U, np.ndarray):
                operador_U = Matrix(operador_U)
            U_dag = Dagger(operador_U)
            rho_new = operador_U * self.rho * U_dag
            return MatrizDensidadCirq(self.qubit, simplify(rho_new), symbolic=True)
        else:
            if isinstance(operador_U, Matrix):
                operador_U = np.array(operador_U.evalf(), dtype=complex)
            U_dag = operador_U.conj().T
            rho_new = operador_U @ self.rho @ U_dag
            return MatrizDensidadCirq(self.qubit, rho_new, symbolic=False)

    def pureza(self) -> Union[float, sp.Expr]:
        """Calcula la pureza del estado mixto: Tr(rho²)"""
        if self.symbolic:
            return simplify(sp.trace(self.rho * self.rho))
        else:
            return np.real(np.trace(self.rho @ self.rho))

    def entropia_von_neumann(self) -> Union[float, sp.Expr]:
        """Calcula la entropía de Von Neumann: S = -Tr(rho log rho)"""
        if self.symbolic:
            # Para matriz simbólica, usar fórmula para qubit
            rho00 = self.rho[0, 0]
            rho11 = self.rho[1, 1]
            # S = -p0*log2(p0) - p1*log2(p1) donde p0=rho00, p1=rho11
            return -rho00 * sp.log(rho00, 2) - rho11 * sp.log(rho11, 2)
        else:
            eigenvals = np.linalg.eigvals(self.rho)
            eigenvals = eigenvals[eigenvals > 1e-15]  # Evitar log(0)
            return -np.sum(eigenvals * np.log2(eigenvals))

    def concurrencia(self) -> Union[float, sp.Expr]:
        """Medida de entrelazamiento para estados de dos qubits (extendible)."""
        if self.symbolic:
            # Fórmula simplificada para un qubit (siempre 0)
            return 0
        else:
            # Para un solo qubit, la concurrencia es siempre 0
            return 0.0

    def fidelidad_con_puro(self, estado_puro: EstadoComplejoCirq) -> Union[float, sp.Expr]:
        """Fidelidad entre matriz de densidad y estado puro: F = <psi|rho|psi>"""
        if self.symbolic and estado_puro.symbolic:
            psi = estado_puro.vector
            return simplify(Dagger(psi) * self.rho * psi)
        else:
            rho_num = self.to_numeric().rho
            psi_num = estado_puro.to_numeric().vector
            return np.real(psi_num.conj().T @ rho_num @ psi_num)

    def distancia_bures(self, otra_matriz: 'MatrizDensidadCirq') -> Union[float, sp.Expr]:
        """Distancia de Bures entre dos matrices de densidad."""
        if self.symbolic and otra_matriz.symbolic:
            # Fórmula simbólica simplificada
            fidelidad = self._fidelidad_matriz(otra_matriz)
            return sp.sqrt(2 * (1 - sp.sqrt(fidelidad)))
        else:
            # Implementación numérica
            rho1 = self.to_numeric().rho
            rho2 = otra_matriz.to_numeric().rho
            
            # F = Tr(sqrt(sqrt(rho1) @ rho2 @ sqrt(rho1)))²
            sqrt_rho1 = sp.linalg.sqrtm(rho1)
            M = sqrt_rho1 @ rho2 @ sqrt_rho1
            sqrt_M = sp.linalg.sqrtm(M)
            fidelidad = np.real(np.trace(sqrt_M))**2
            
            return np.sqrt(2 * (1 - np.sqrt(fidelidad)))

    def _fidelidad_matriz(self, otra_matriz: 'MatrizDensidadCirq') -> Union[float, sp.Expr]:
        """Fidelidad entre dos matrices de densidad."""
        if self.symbolic and otra_matriz.symbolic:
            # Implementación simbólica simplificada
            return simplify(sp.trace(self.rho * otra_matriz.rho))
        else:
            rho1 = self.to_numeric().rho
            rho2 = otra_matriz.to_numeric().rho
            sqrt_rho1 = sp.linalg.sqrtm(rho1)
            M = sqrt_rho1 @ rho2 @ sqrt_rho1
            sqrt_M = sp.linalg.sqrtm(M)
            return np.real(np.trace(sqrt_M))**2

    def poblaciones(self) -> Tuple[Union[float, sp.Expr], Union[float, sp.Expr]]:
        """Devuelve las poblaciones de los estados |0⟩ y |1⟩."""
        if self.symbolic:
            return self.rho[0, 0], self.rho[1, 1]
        else:
            return np.real(self.rho[0, 0]), np.real(self.rho[1, 1])

    def coherencias(self) -> Tuple[Union[complex, sp.Expr], Union[complex, sp.Expr]]:
        """Devuelve los elementos no diagonales (coherencias cuánticas)."""
        if self.symbolic:
            return self.rho[0, 1], self.rho[1, 0]
        else:
            return self.rho[0, 1], self.rho[1, 0]

    def to_numeric(self) -> 'MatrizDensidadCirq':
        """Convierte matriz simbólica a numérica."""
        if not self.symbolic:
            return self
        
        rho_num = np.array(self.rho.evalf(), dtype=complex)
        return MatrizDensidadCirq(self.qubit, rho_num, symbolic=False, metadata=self.metadata.copy())

    def to_symbolic(self) -> 'MatrizDensidadCirq':
        """Convierte matriz numérica a simbólica."""
        if self.symbolic:
            return self
        
        rho_sym = Matrix(self.rho)
        return MatrizDensidadCirq(self.qubit, rho_sym, symbolic=True, metadata=self.metadata.copy())

    def extraer_estado_puro(self) -> Optional[EstadoComplejoCirq]:
        """Extrae estado puro si la pureza es 1 (dentro de tolerancia)."""
        pureza = self.pureza()
        
        if self.symbolic:
            # Para caso simbólico, asumir que es puro si se puede extraer
            eigenvals, eigenvecs = self.rho.diagonalize()
            # Encontrar el eigenvalor más grande (debería ser 1 para estados puros)
            max_idx = 0  # Simplificación para caso simbólico
            alpha = eigenvecs[0, max_idx]
            beta = eigenvecs[1, max_idx]
            return EstadoComplejoCirq(self.qubit, alpha, beta, symbolic=True)
        else:
            if abs(pureza - 1.0) < 1e-10:
                eigenvals, eigenvecs = np.linalg.eig(self.rho)
                max_idx = np.argmax(eigenvals)
                eigenvec = eigenvecs[:, max_idx]
                return EstadoComplejoCirq(self.qubit, eigenvec[0], eigenvec[1], symbolic=False)
            return None

    def visualizar_matriz(self):
        """Visualiza la matriz de densidad como heatmap."""
        if self.symbolic:
            print("Visualización requiere conversión a valores numéricos")
            return self.to_numeric().visualizar_matriz()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Parte real
        im1 = ax1.imshow(np.real(self.rho), cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('Parte Real')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['|0⟩', '|1⟩'])
        ax1.set_yticklabels(['⟨0|', '⟨1|'])
        plt.colorbar(im1, ax=ax1)
        
        # Parte imaginaria
        im2 = ax2.imshow(np.imag(self.rho), cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title('Parte Imaginaria')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['|0⟩', '|1⟩'])
        ax2.set_yticklabels(['⟨0|', '⟨1|'])
        plt.colorbar(im2, ax=ax2)
        
        # Magnitud
        im3 = ax3.imshow(np.abs(self.rho), cmap='viridis', vmin=0, vmax=1)
        ax3.set_title('Magnitud')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['|0⟩', '|1⟩'])
        ax3.set_yticklabels(['⟨0|', '⟨1|'])
        plt.colorbar(im3, ax=ax3)
        
        # Información del estado
        ax4.axis('off')
        p0, p1 = self.poblaciones()
        c01, c10 = self.coherencias()
        pureza = self.pureza()
        entropia = self.entropia_von_neumann()
        
        info_text = f"""
        Estado Mixto
        
        Poblaciones:
        P(|0⟩) = {p0:.4f}
        P(|1⟩) = {p1:.4f}
        
        Coherencias:
        ⟨0|ρ|1⟩ = {c01:.4f}
        ⟨1|ρ|0⟩ = {c10:.4f}
        
        Métricas:
        Pureza = {pureza:.4f}
        Entropía = {entropia:.4f}
        """
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Representación en cadena de la matriz de densidad."""
        if self.symbolic:
            return f"Matriz de Densidad (Simbólica):\n{self.rho}\nPureza: {self.pureza()}"
        else:
            pureza = self.pureza()
            entropia = self.entropia_von_neumann()
            p0, p1 = self.poblaciones()
            return (f"Matriz de Densidad:\n{self.rho}\n"
                   f"Poblaciones: P₀={p0:.4f}, P₁={p1:.4f}\n"
                   f"Pureza: {pureza:.4f}, Entropía: {entropia:.4f}")

    def __repr__(self) -> str:
        return f"MatrizDensidadCirq(qubit={self.qubit}, symbolic={self.symbolic})"

# --- Funciones de Utilidad para Estados Mixtos ---

def crear_estado_termico(temperatura: float, energia_gap: float, qubit: cirq.Qubit) -> MatrizDensidadCirq:
    """Crea un estado térmico en equilibrio."""
    beta = 1.0 / temperatura  # Asumiendo kB = 1
    Z = 1 + np.exp(-beta * energia_gap)  # Función de partición
    
    p0 = 1.0 / Z  # Probabilidad del estado |0⟩
    p1 = np.exp(-beta * energia_gap) / Z  # Probabilidad del estado |1⟩
    
    rho_thermal = np.array([[p0, 0], [0, p1]], dtype=complex)
    return MatrizDensidadCirq(qubit, rho_thermal, symbolic=False)

def simular_decoherencia_T1(estado_inicial: EstadoComplejoCirq, T1: float, tiempo: float) -> MatrizDensidadCirq:
    """Simula decoherencia por decaimiento de amplitud (T1)."""
    # Operador de Lindblad para decaimiento: sigma_minus = |0⟩⟨1|
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 1.0 / T1
    
    # Crear matriz de densidad inicial
    rho_inicial = MatrizDensidadCirq.from_estado_puro(estado_inicial)
    
    # Hamiltoniano trivial (sin evolución unitaria)
    H = np.zeros((2, 2), dtype=complex)
    
    # Evolución con Lindblad
    return rho_inicial.evolucion_lindblad(H, [(sigma_minus, gamma)], tiempo)

def simular_decoherencia_T2(estado_inicial: EstadoComplejoCirq, T2: float, tiempo: float) -> MatrizDensidadCirq:
    """Simula decoherencia de fase pura (T2)."""
    # Operador de Lindblad para dephasing: sigma_z
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    gamma_phi = 1.0 / T2
    
    rho_inicial = MatrizDensidadCirq.from_estado_puro(estado_inicial)
    H = np.zeros((2, 2), dtype=complex)
    
    return rho_inicial.evolucion_lindblad(H, [(sigma_z, gamma_phi)], tiempo)

# --- Ejemplo de Uso ---

if __name__ == "__main__":
    # Crear qubit
    q = cirq.LineQubit(0)
    
    # Estado simbólico
    print("=== Estados Simbólicos ===")
    estado_simbolico = EstadoComplejoCirq(
        q, 
        alpha=BaseEstados.alpha,
        beta=BaseEstados.beta,
        symbolic=True
    )
    print(f"Estado simbólico: {estado_simbolico}")
    print(f"Probabilidades: P₀ = {estado_simbolico.probabilidad_0()}, P₁ = {estado_simbolico.probabilidad_1()}")
    
    # Estado numérico específico
    print("\n=== Estados Numéricos ===")
    estado_hadamard = EstadoComplejoCirq(q, alpha=1/np.sqrt(2), beta=1/np.sqrt(2))
    print(f"Estado |+⟩: {estado_hadamard}")
    print(f"Clasificación: {estado_hadamard.clasificar_estado()}")
    print(f"Coordenadas Bloch: {estado_hadamard.representacion_bloch()}")
    
    # Aplicar operador Pauli-X
    print("\n=== Aplicación de Operadores ===")
    estado_x = estado_hadamard.aplicar_operador(BaseEstados.PAULI_X.evalf())
    print(f"Después de Pauli-X: {estado_x}")
    
    # Fidelidad entre estados
    print(f"Fidelidad |+⟩ con X|+⟩: {estado_hadamard.fidelidad(estado_x):.6f}")
    
    # Estados de Bell
    print("\n=== Estados de Bell ===")
    bell1, bell2 = crear_estado_bell("phi_plus")
    print(f"Estado Bell Φ⁺: {bell1}, {bell2}")

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
