"""Basic sanity check for NNNC reference implementation"""
import numpy as np
from reference_impl.NNNC import NNNCCore
from reference_impl.neutral_environment_space import NeutralEnvironmentSpace
from reference_impl.systemic_algorithms_impl import InterAlgorithmCommunicationProtocol

def test_basic_operation():
    """Test basic instantiation and one cognitive cycle"""
    print("Creating NNNC Core...")
    nnnc = NNNCCore()
    print("Creating NES...")
    nes = NeutralEnvironmentSpace()
    print("Creating IACP...")
    iacp = InterAlgorithmCommunicationProtocol()
    
    print("\nStarting test cycle...")
    # Get random input from NES
    input_data = nes.get_sensory_data()
    print(f"Input shape: {input_data.shape}")
    
    # Run NNNC cognitive cycle
    print("Running NNNC cognitive cycle...")
    decision = nnnc.perceive_and_decide(input_data)
    print(f"Decision confidence: {decision['confidence']:.3f}")
    print(f"System capacity: {decision['capacity']:.3f}")
    
    # Run algorithm cycle through IACP
    print("\nRunning IACP algorithm cycle...")
    cycle_result = iacp.execute_cycle(
        input_data,
        nnnc.facn.subconscious,
        nnnc.cogflux.dimensions
    )
    
    print(f"Cycle complete. Final output shape: {cycle_result['final_output'].shape}")
    print(f"Critical thinking status: {cycle_result['critical_thinking']['status']}")
    
    if cycle_result.get('mutation'):
        print(f"Mutation occurred: {cycle_result['mutation']['type']}")
    
    if cycle_result.get('bias_formed'):
        print(f"New bias formed: {cycle_result['bias_formed']['type']}")
    
    print("\nGetting system state...")
    state = nnnc.get_system_state()
    print(f"Total dimensions: {len(state['capacity_dimensions'])}")
    print(f"Global efficiency: {state['global_efficiency']:.3f}")
    print(f"Existence time: {state['existence_time']}")
    
    print("\nAll basic operations completed successfully!")
    assert True

if __name__ == "__main__":
    test_basic_operation()