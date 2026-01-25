import brian2 as b2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeuronConfig:
    """Configuration for neuron parameters"""
    n_neurons: int = 10
    threshold: float = 1.0
    tau_ms: float = 10.0
    refractory_ms: float = 5.0
    method: str = 'euler'

@dataclass
class SimulationResults:
    """Structured results from simulation"""
    spike_rates: Dict[str, float]
    biological_score: float
    spike_counts: Dict[str, np.ndarray]
    memory_updated: bool

class BrainSimulator:
    """
    Simulates a Spiking Neural Network (SNN) using Brian2 with improved architecture.
    Uses Leaky Integrate-and-Fire (LIF) model for biological plausibility.
    """
    
    def __init__(
        self, 
        duration_ms: float = 50.0,
        neuron_config: Optional[NeuronConfig] = None,
        enable_logging: bool = False
    ):
        self.duration = duration_ms * b2.ms
        self.duration_ms = duration_ms
        self.config = neuron_config or NeuronConfig()
        self.enable_logging = enable_logging
        
        # Optimize Brian2 compilation
        b2.prefs.codegen.target = 'numpy'
        # b2.prefs.codegen.cpp.compiler = 'unix'  # or 'msvc' on Windows
        
        # Define brain regions with biological plausibility
        self.regions = {
            'attention': {'tau': 8, 'sensitivity': 1.0},
            'emotion': {'tau': 15, 'sensitivity': 1.2},
            'reward': {'tau': 12, 'sensitivity': 1.3},
            'pain': {'tau': 10, 'sensitivity': 1.1},
            'consistency': {'tau': 20, 'sensitivity': 0.9},
            'authority': {'tau': 18, 'sensitivity': 1.0},
            'fear': {'tau': 10, 'sensitivity': 1.4},
            'social': {'tau': 16, 'sensitivity': 1.1},
            'disgust': {'tau': 12, 'sensitivity': 1.3},
            'trust': {'tau': 18, 'sensitivity': 1.0},
            'uncertainty': {'tau': 14, 'sensitivity': 1.2},
            'moral': {'tau': 20, 'sensitivity': 0.95}
        }
        
        # Short-Term Memory state
        self.memory_state = {}
        self.priming_factor = 0.3
        self.decay_factor = 0.5
        
        # Statistics tracking
        self.last_simulation_stats = {}
        
    def simulate_signals(self, features: Dict[str, float]) -> SimulationResults:
        """
        Runs a spiking simulation based on input features using the Izhikevich model.
        
        Args:
            features: Dict with keys 'attention', 'emotion', 'reward', 'pain', etc.
        
        Returns:
            SimulationResults with spike rates and biological score
        """
        # Validate inputs
        features = self._validate_features(features)
        
        # Izhikevich Model Equations
        # v: membrane potential, u: recovery variable
        # Standard parameters for Regular Spiking (RS) neurons
        eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1 (unless refractory)
        du/dt = a*(b*v - u) / ms : 1 (unless refractory)
        I : 1
        a : 1
        b : 1
        c : 1
        d : 1
        '''
        
        neuron_groups = {}
        monitors = {}
        net = b2.Network()
        
        # Default Izhikevich parameters for Regular Spiking (RS)
        # These can be tuned per region if more complexity is needed
        base_a, base_b, base_c, base_d = 0.02, 0.2, -65, 8
        
        # Create neuron groups with region-specific parameters
        for region, params in self.regions.items():
            group = b2.NeuronGroup(
                self.config.n_neurons,
                eqs,
                threshold='v > 30',
                reset='v = c; u += d',
                refractory=self.config.refractory_ms * b2.ms,
                method=self.config.method
            )
            
            # Initialize state variables
            group.v = -65  # Resting potential
            group.u = base_b * group.v
            
            group.a = base_a
            group.b = base_b
            group.c = base_c
            group.d = base_d
            
            # Compute input current with validation
            feature_val = np.clip(features.get(region, 0.5), 0, 1)
            memory_priming = self.memory_state.get(region, 0.0) * self.priming_factor
            sensitivity = params['sensitivity']
            
            # Izhikevich current I: spikes usually start around I=10
            # Range: [5, 25] scaled by sensitivity and priming
            group.I = 5.0 + (feature_val * 15.0 * sensitivity) + (memory_priming * 10.0)
            
            neuron_groups[region] = group
            spike_monitor = b2.SpikeMonitor(group)
            monitors[region] = spike_monitor
            net.add(group, spike_monitor)
        
        # Run simulation
        try:
            net.run(self.duration)
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
        
        # Extract and process results
        spike_rates = {}
        spike_counts = {}
        for region in self.regions.keys():
            counts = monitors[region].count
            spike_counts[region] = counts
            # Normalize to [0, 1] range based on expected max firing rate
            # In Izhikevich, with I=20, freq is ~60-80Hz
            # 50ms duration -> ~3-4 spikes per neuron
            avg_spikes = np.mean(counts)
            expected_max = (self.duration_ms / 1000.0) * 100.0 # 100Hz max
            spike_rates[region] = min(1.0, avg_spikes / (expected_max if expected_max > 0 else 1))
        
        # Update memory
        self._update_memory(spike_rates)
        
        # Calculate biological plausibility score
        biological_score = self._compute_biological_score(spike_rates)
        
        self.last_simulation_stats = {
            'spike_rates': spike_rates,
            'biological_score': biological_score,
            'memory_state': self.memory_state.copy()
        }
        
        if self.enable_logging:
            logger.info(f"Simulation (Izhikevich) complete. Score: {biological_score:.3f}")
        
        return SimulationResults(
            spike_rates=spike_rates,
            biological_score=biological_score,
            spike_counts=spike_counts,
            memory_updated=True
        )
    
    def _validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize input features"""
        validated = {}
        for region in self.regions.keys():
            val = features.get(region, 0.5)
            # Clip to valid range
            validated[region] = np.clip(float(val), 0.0, 1.0)
        return validated
    
    def _update_memory(self, spike_rates: Dict[str, float]) -> None:
        """Update short-term memory with exponential decay"""
        for region, spike_score in spike_rates.items():
            old_mem = self.memory_state.get(region, 0.0)
            # Exponential moving average
            self.memory_state[region] = np.clip(
                spike_score + (old_mem * self.decay_factor),
                0.0, 1.0
            )
    
    def _compute_biological_score(self, spike_rates: Dict[str, float]) -> float:
        """
        Compute biological plausibility score with neuroscientific weighting.
        """
        # Approach/Engagement weights
        reward_weight = 1.5
        emotion_weight = 1.2
        trust_weight = 1.3
        moral_weight = 1.1
        social_weight = 0.9
        
        # Avoidance/Threat weights (negative)
        pain_weight = -2.5
        fear_weight = -2.0
        disgust_weight = -1.8
        uncertainty_weight = -1.2
        
        # Modulators
        consistency_weight = 0.8
        authority_weight = 0.7
        
        # Calculate approach signals
        approach = (
            spike_rates.get('reward', 0.0) * reward_weight +
            spike_rates.get('emotion', 0.0) * emotion_weight +
            spike_rates.get('trust', 0.0) * trust_weight +
            spike_rates.get('moral', 0.0) * moral_weight +
            spike_rates.get('social', 0.0) * social_weight
        )
        
        # Calculate avoidance signals
        avoidance = (
            spike_rates.get('pain', 0.0) * pain_weight +
            spike_rates.get('fear', 0.0) * fear_weight +
            spike_rates.get('disgust', 0.0) * disgust_weight +
            spike_rates.get('uncertainty', 0.0) * uncertainty_weight
        )
        
        # Modulators (multiplicative)
        attention = np.clip(0.5 + spike_rates.get('attention', 0.5), 0.5, 1.5)
        consistency = np.clip(0.7 + (spike_rates.get('consistency', 0.5) * consistency_weight), 0.5, 1.3)
        authority = np.clip(0.8 + (spike_rates.get('authority', 0.5) * authority_weight), 0.6, 1.2)
        
        # Combine: approach + avoidance with modulation
        # Attention amplifies overall signal; consistency & authority boost credibility
        score = (approach + avoidance) * attention * consistency * authority
        
        return np.clip(score, -1.0, 3.0)
    
    def evaluate_keyword_snn(
        self, 
        text: str, 
        keyword: str, 
        llm_signals: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Refines LLM-based signals using spiking simulation.
        
        Args:
            text: Input text
            keyword: Keyword to evaluate
            llm_signals: LLM-generated neural signals
        
        Returns:
            Tuple of (biological_score, detailed_results)
        """
        # Extract features from nested LLM signal structure
        features = self._extract_features_from_llm(llm_signals)
        
        # Run simulation
        results = self.simulate_signals(features)
        
        # Return score with additional metadata
        return results.biological_score, {
            'spike_rates': results.spike_rates,
            'memory_state': self.memory_state.copy(),
            'text': text,
            'keyword': keyword
        }
    
    def _extract_features_from_llm(self, llm_signals: Dict[str, Any]) -> Dict[str, float]:
        """
        Safely extract features from nested LLM signal structure.
        """
        def safe_get(d, *keys, default=None):
            for key in keys:
                if isinstance(d, dict):
                    d = d.get(key, {})
                else:
                    return default
            if isinstance(d, (int, float)):
                return float(d)
            if isinstance(d, str):
                try:
                    return float(d)
                except ValueError:
                    return default
            return default
        
        # Helper to get the first available signal from a list of paths
        def get_signal(paths, default=0.5):
            for path in paths:
                val = safe_get(llm_signals, *path)
                if val is not None:
                    return val
            return default

        return {
            # Core Systems
            'attention': get_signal([['attention'], ['dlpfc', 'executive']]),
            'emotion': get_signal([['emotion'], ['amygdala', 'salience']]),
            'reward': get_signal([['reward', 'dopamine'], ['reward'], ['ventral_striatum', 'motivation']]),
            'pain': get_signal([['pain'], ['insula', 'pain'], ['acc', 'distress']]),
            'consistency': get_signal([['consistency'], ['hippocampus', 'consistency'], ['hippocampus', 'coherence']]),
            'authority': get_signal([['authority'], ['dlpfc', 'authority'], ['dlpfc', 'social_hierarchy']]),
            
            # Advanced Emotional Systems
            'fear': get_signal([['fear'], ['amygdala', 'threat'], ['amygdala', 'fear']]),
            
            # Social Cognition
            'social': get_signal([['social'], ['mmpfc', 'mentalizing'], ['tpj', 'theory_of_mind']]),
            
            # Disgust System
            'disgust': get_signal([['disgust'], ['insula', 'disgust'], ['insula', 'core_disgust']]),
            
            # Trust & Affiliation
            'trust': get_signal([['trust'], ['mmpfc', 'trust'], ['ventral_striatum', 'affiliation']]),
            
            # Uncertainty & Prediction Error
            'uncertainty': get_signal([['uncertainty'], ['insula', 'uncertainty'], ['anterior_insula', 'prediction_error']]),
            
            # Moral Reasoning
            'moral': get_signal([['moral'], ['mmpfc', 'moral'], ['pcc', 'self_reference']])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve last simulation statistics"""
        return self.last_simulation_stats.copy()
    
    def clear_memory(self) -> None:
        """Reset the biological memory state"""
        self.memory_state = {}
        logger.info("🧠 [BrainSimulator] Memory cleared.")
    
    def reset(self) -> None:
        """Full reset of simulator state"""
        self.clear_memory()
        self.last_simulation_stats = {}
        logger.info("🧠 [BrainSimulator] Full reset complete.")

