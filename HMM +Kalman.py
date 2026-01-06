import numpy as np

class SwitchingKalmanFilter:
    def __init__(self, transition_matrix, kf_params):
        """
        transition_matrix: HMM state transition probabilities (NxN)
        kf_params: Dictionary containing A, C, Q, R for each discrete state
        """
        self.pi = transition_matrix
        self.kf_params = kf_params
        self.num_states = len(transition_matrix)
        
    def predict_update(self, x_prev, P_prev, z_curr, measurement):
        """
        Standard Kalman Update step mapped to a specific HMM state (z_curr)
        """
        # Get parameters for the current regime
        A = self.kf_params[z_curr]['A']
        C = self.kf_params[z_curr]['C']
        Q = self.kf_params[z_curr]['Q']
        R = self.kf_params[z_curr]['R']
        I = np.eye(A.shape[0])

        # --- PREDICT ---
        x_pred = A @ x_prev
        P_pred = A @ P_prev @ A.T + Q

        # --- UPDATE ---
        y = measurement - (C @ x_pred) # Innovation
        S = C @ P_pred @ C.T + R      # Innovation Covariance
        K = P_pred @ C.T @ np.linalg.inv(S) # Kalman Gain

        x_curr = x_pred + K @ y
        P_curr = (I - K @ C) @ P_pred

        return x_curr, P_curr

# --- Example Usage ---

# 1. Define HMM Transition (0: Stationary, 1: Moving)
# High probability of staying in the same state
trans_mat = np.array([[0.95, 0.05], 
                      [0.10, 0.90]])

# 2. Define Kalman Parameters for each regime
params = {
    0: { # Regime 0: Stationary (Small noise, Identity transition)
        'A': np.array([[1.0]]),
        'C': np.array([[1.0]]),
        'Q': np.array([[0.01]]),
        'R': np.array([[0.1]])
    },
    1: { # Regime 1: Moving (Large noise, Constant velocity-like)
        'A': np.array([[1.1]]),
        'C': np.array([[1.0]]),
        'Q': np.array([[0.5]]),
        'R': np.array([[0.1]])
    }
}

# 3. Simulation
skf = SwitchingKalmanFilter(trans_mat, params)

# Initial conditions
x = np.array([[0.0]])
P = np.array([[1.0]])
current_z = 0 # Start stationary

# Simulated measurements
measurements = [0.1, 0.2, 0.5, 1.2, 2.5, 4.0, 6.2]
estimated_states = []

print(f"{'Step':<6} | {'Mode':<12} | {'Estimate':<10}")
print("-" * 35)

for i, m in enumerate(measurements):
    # In a real SLDS, you'd estimate Z. Here we simulate a mode switch at step 3.
    if i == 3: current_z = 1 
    
    x, P = skf.predict_update(x, P, current_z, np.array([[m]]))
    estimated_states.append(x[0,0])
    
    mode_str = "Stationary" if current_z == 0 else "Moving"
    print(f"{i:<6} | {mode_str:<12} | {x[0,0]:.4f}")
