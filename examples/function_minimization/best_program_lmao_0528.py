# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Simulated Annealing algorithm that balances exploration and exploitation.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)

    # Parameters for simulated annealing
    temperature = 100.0
    cooling_factor = 0.995
    momentum = 0.5
    exploration_rate = 0.3
    min_temperature = 0.1

    # Track the best solution found so far
    best_global = (best_x, best_y, best_value)

    for _ in range(iterations):
        # Decide whether to do local or global move
        if np.random.rand() < exploration_rate:
            # Global exploration: uniform random in the entire space
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[0], bounds[1])
        else:
            # Local exploration: perturbation around the current best point
            x = best_x + np.random.uniform(-0.5, 0.5)
            y = best_y + np.random.uniform(-0.5, 0.5)

        # Apply momentum to help escape local minima
        x = momentum * best_x + (1 - momentum) * x
        y = momentum * best_y + (1 - momentum) * y

        # Clamp to bounds
        x = max(bounds[0], min(x, bounds[1]))
        y = max(bounds[0], min(y, bounds[1]))

        value = evaluate_function(x, y)

        # Update the best solution if we found a better one
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
            best_global = (x, y, value)

        # Cool down the temperature
        temperature *= cooling_factor
        if temperature < min_temperature:
            temperature = min_temperature

    return best_global


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")
