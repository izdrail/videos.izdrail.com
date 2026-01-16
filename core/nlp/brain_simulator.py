import brian2 as b2
import numpy as np
from typing import Dict, Any, List, Optional

class BrainSimulator:
    """
    Simulates a Spiking Neural Network (SNN) using Brian 2 to evaluate biological signals.
    Uses a Leaky Integrate-and-Fire (LIF) model.
    """
    
    def __init__(self, duration_ms: float = 50.0):
        self.duration = duration_ms * b2.ms
        # Global preference for Brian 2 speed
        b2.prefs.codegen.target = 'numpy'
        
        # Short-Term Memory (STM) state
        self.memory_state = {} 
        self.priming_factor = 0.3  # How much previous state influences current (I)
        self.decay_factor = 0.5    # How much memory decays if not reinforced
        
    def simulate_signals(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Runs a spiking simulation based on input features.
        
        Features:
          - attention (0.0 - 1.0)
          - emotion (0.0 - 1.0)
          - reward (0.0 - 1.0)
          - pain (0.0 - 1.0)
          
        Returns:
          Normalized spike rates for each region.
        """
        # 1. Define Model Equations
        # v: membrane potential, I: input current, tau: time constant
        eqs = '''
        dv/dt = (I - v) / tau : 1
        I : 1
        tau : second
        '''
        
        # 2. Setup Neuron Groups (one for each "brain region")
        regions = ['attention', 'emotion', 'reward', 'pain', 'consistency', 'authority']
        n_neurons_per_group = 10
        
        # Create a single group with subgroups for efficiency if needed, 
        # but for small scale, separate groups are clearer.
        neuron_groups = {}
        monitors = {}
        
        net = b2.Network()
        
        for region in regions:
            group = b2.NeuronGroup(
                n_neurons_per_group, 
                model=eqs, 
                threshold='v > 1.0', 
                reset='v = 0', 
                refractory=5*b2.ms,
                method='exact'
            )
            
            # Initial conditions
            group.v = 0
            group.tau = 10*b2.ms
            
            # Map input features to current I
            # Base current + scaled feature influence
            feature_val = features.get(region, 0.5)
            
            # STM Influence: Add priming from previous state
            memory_priming = self.memory_state.get(region, 0.0) * self.priming_factor
            
            # Threshold is 1.0, tau is 10ms. 
            # I > 1.0 will cause spiking.
            group.I = 0.5 + (feature_val * 1.5) + memory_priming # Range [0.5, 2.0+]
            
            neuron_groups[region] = group
            monitors[region] = b2.SpikeMonitor(group)
            net.add(group, monitors[region])
            
        # 3. Running simulation
        net.run(self.duration)
        
        # 4. Extract results and Update Memory
        results = {}
        for region in regions:
            spike_counts = monitors[region].count
            avg_rate = np.mean(spike_counts) / (self.duration / b2.ms)
            spike_score = min(1.0, avg_rate / 0.15)
            
            results[region] = spike_score
            
            # Update Memory: Current score + Decayed old memory
            old_mem = self.memory_state.get(region, 0.0)
            self.memory_state[region] = min(1.0, spike_score + (old_mem * self.decay_factor))
            
        return results

    def clear_memory(self):
        """Resets the biological memory state"""
        print("🧠 [BrainSimulator] Memory cleared.")
        self.memory_state = {}

    def evaluate_keyword_snn(self, text: str, keyword: str, llm_signals: Dict[str, Any]) -> float:
        """
        Refines LLM-based signals using a spiking simulation.
        """
        # Flatten LLM signals for the simulator
        features = {
            'attention': llm_signals.get('attention', 0.5),
            'emotion': llm_signals.get('amygdala', {}).get('salience', 0.5),
            'reward': llm_signals.get('reward', {}).get('dopamine', 0.5),
            'pain': llm_signals.get('insula', {}).get('pain', 0.0),
            'consistency': llm_signals.get('hippocampus', {}).get('consistency', 0.5),
            'authority': llm_signals.get('dlpfc', {}).get('authority', 0.5)
        }
        
        simulation_results = self.simulate_signals(features)
        
        # Calculate biological score based on spike densities
        # High Reward/Emotion spike rates increase score, Pain rates decrease it.
        reward_spike = simulation_results.get('reward', 0.0)
        emotion_spike = simulation_results.get('emotion', 0.0)
        pain_spike = simulation_results.get('pain', 0.0)
        attention_multiplier = 0.5 + simulation_results.get('attention', 0.5) # 0.5 to 1.5
        
        score = (reward_spike * 1.5 + emotion_spike * 1.2) - (pain_spike * 2.5)
        return score * attention_multiplier
