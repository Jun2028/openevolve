import numpy as np

def search_algorithm(iterations=1000, bounds=(-5,5)):
    low, high = bounds
    pop_size = 50
    
    # Initialize population
    population = np.random.uniform([low]*2, [high]*2, (pop_size, 2))
    
    # Track best solution found so far (global best)
    best_value = float('inf')
    best_solution = None
    
    # Define crossover and mutation functions
    def crossover(parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(2):
            if np.random.rand() < 0.7:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2
    
    def mutate(individual):
        new_ind = individual.copy()
        for i in range(2):
            if np.random.rand() < 0.2:
                new_ind[i] += np.random.uniform(-2.0, 2.0)
                new_ind[i] = np.clip(new_ind[i], low, high)
        return new_ind
    
    # Parameters for simulated annealing with restarts
    T_initial = 10.0
    cooling_rate = 0.95
    min_temp = 0.01
    restart_interval = 100
    
    # Track iterations without improvement
    ni = 0
    
    # Track temperature for simulated annealing
    sa_temp = T_initial
    
    for i in range(iterations):
        # Evaluate fitness for all individuals
        fitness = np.zeros(pop_size)
        for j in range(pop_size):
            x, y = population[j]
            fitness[j] = evaluate_function(x, y)
        
        # Find best solution in current population
        best_idx = np.argmin(fitness)
        current_best = population[best_idx]
        current_fitness = fitness[best_idx]
        
        # Update global best if found
        if current_fitness < best_value:
            best_value = current_fitness
            best_solution = current_best.copy()
            ni = 0
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            tournament_size = 3
            candidates = np.random.choice(pop_size, tournament_size, replace=False)
            winner_idx = candidates[np.argmin(fitness[candidates])]
            parent = population[winner_idx].copy()
            
            # Apply crossover with probability 0.8
            if np.random.rand() < 0.8:
                second_parent_idx = np.random.randint(0, pop_size)
                second_parent = population[second_parent_idx].copy()
                child1, child2 = crossover(parent, second_parent)
                child1 = mutate(child1)
                child2 = mutate(child2)
                # Choose the better of the two children
                f1 = evaluate_function(*child1)
                f2 = evaluate_function(*child2)
                new_population.append(child1 if f1 < f2 else child2)
            else:
                # Just mutate the selected parent
                new_population.append(mutate(parent))
        
        # Update population for next iteration
        population = np.array(new_population)
        
        # Dynamic local search with simulated annealing around best solution
        if best_solution is not None:
            # Perform local search with simulated annealing
            T = sa_temp
            current_x, current_y = best_solution
            current_val = best_value
            best_x, best_y = current_x, current_y
            best_val = current_val
            
            # Perform local search steps
            for _ in range(20):  # Perform 20 local search steps
                # Perturb the solution
                new_x = current_x + np.random.uniform(-0.5, 0.5)
                new_y = current_y + np.random.uniform(-0.5, 0.5)
                new_x = np.clip(new_x, low, high)
                new_y = np.clip(new_y, low, high)
                
                # Evaluate the perturbed solution
                new_val = evaluate_function(new_x, new_y)
                
                # Simulated annealing acceptance criterion
                if new_val < current_val or np.random.rand() < np.exp(-(new_val - current_val) / T):
                    current_val = new_val
                    current_x, current_y = new_x, new_y
                
                if current_val < best_val:
                    best_val = current_val
                    best_x, best_y = current_x, current_y
                
                T *= cooling_rate
                if T < min_temp:
                    T = min_temp
            
            # Update global best if local search found better solution
            if best_val < best_value:
                best_value = best_val
                best_solution = np.array([best_x, best_y])
            
            # Update simulated annealing temperature
            sa_temp = T
        
        # Dynamic restart strategy based on improvement rate
        if best_value < -1.4 or np.random.rand() < 0.1:
            # Restart with larger exploration
            x_restart = np.random.uniform(low, high)
            y_restart = np.random.uniform(low, high)
            restart_fitness = evaluate_function(x_restart, y_restart)
            
            if restart_fitness < best_value:
                best_value = restart_fitness
                best_solution = np.array([x_restart, y_restart])
                ni = 0
        
        # Track iterations without improvement
        ni += 1
    
    return best_solution[0], best_solution[1], best_value

def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2)/20

def run_search():
    x, y, value = search_algorithm()
    return x, y, value

if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")