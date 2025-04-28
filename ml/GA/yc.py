import numpy as np
import random
import pandas as pd
import joblib
import random
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor
from matplotlib.collections import LineCollection  # Add this import
# --- Genetic Algorithm Optimizer Class ---
class GeneticAlgorithmOptimizer:
    def __init__(self, model_paths, feature_names_path, scaler_path, imputer_path,
                 population_size, generations, crossover_rate, initial_mutation_rate,
                 continuous_features=None, categorical_features=None,
                 target_temperatures=None, target_activation_energies=None,
                 previous_best_individual=None):
        self.models = [joblib.load(model_path) for model_path in model_paths]
        self.feature_names = joblib.load(feature_names_path)
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = initial_mutation_rate
        self.initial_mutation_rate = initial_mutation_rate
        self.continuous_features = continuous_features or {}
        self.categorical_features = categorical_features or {}
        self.feature_name_to_index = {name: i for i, name in enumerate(self.feature_names)}
        self.target_temperatures = target_temperatures
        self.target_activation_energies = target_activation_energies
        self.previous_best_individual = previous_best_individual

        for feature, values in self.categorical_features.items():
            if not all(isinstance(v, (int, str)) for v in values):
                raise ValueError(f"Categorical feature '{feature}' values must be integers or strings.")

    def _encode_individual(self, individual_dict):
        individual = []
        for feature_name in self.feature_names:
            individual.append(individual_dict.get(feature_name, 0))
        return individual

    def _generate_initial_population(self):
        population = []
        if self.previous_best_individual is not None and len(population) < self.population_size:
            population.append(self.previous_best_individual.copy())

        random_count = int(0.5 * self.population_size)  # 50% 随机个体
        for _ in range(random_count):
            individual_dict = {}
            for feature, (min_val, max_val) in self.continuous_features.items():
                individual_dict[feature] = random.uniform(min_val, max_val)
            for feature, values in self.categorical_features.items():
                individual_dict[feature] = random.choice(values)
            individual = self._encode_individual(individual_dict)
            population.append(individual)

        while len(population) < self.population_size:
            individual_dict = {}
            for feature, (min_val, max_val) in self.continuous_features.items():
                individual_dict[feature] = random.uniform(min_val, max_val)
            for feature, values in self.categorical_features.items():
                individual_dict[feature] = random.choice(values)
            individual = self._encode_individual(individual_dict)
            population.append(individual)
        return np.array(population)

    def _batch_fitness(self, population):
        population_df = pd.DataFrame(population, columns=self.feature_names)
        population_imputed = self.imputer.transform(population_df)
        population_scaled = self.scaler.transform(population_imputed)
        predictions = np.expm1(self.models[0].predict(population_scaled))

        onset_temp_predictions = predictions[:, 0]
        activation_energy_predictions = predictions[:, 1]

        temp_penalty = np.zeros(len(population))
        energy_penalty = np.zeros(len(population))

        if self.target_temperatures:
            lower_temp_target, upper_temp_target = self.target_temperatures
            temp_penalty = np.where(onset_temp_predictions > upper_temp_target,
                                    onset_temp_predictions - upper_temp_target,
                                    np.where(onset_temp_predictions < lower_temp_target,
                                             lower_temp_target - onset_temp_predictions, 0))
            temp_reward = -np.square(onset_temp_predictions - (lower_temp_target + upper_temp_target) / 2)

        if self.target_activation_energies:
            lower_energy_target, upper_energy_target = self.target_activation_energies
            energy_penalty = np.where(activation_energy_predictions > upper_energy_target,
                                      activation_energy_predictions - upper_energy_target,
                                      np.where(activation_energy_predictions < lower_energy_target,
                                               lower_energy_target - activation_energy_predictions, 0))
            energy_reward = -np.square(activation_energy_predictions - (lower_energy_target + upper_energy_target) / 2)

        combined_penalty = temp_penalty + energy_penalty
        combined_reward = 0.01 * (temp_reward + energy_reward)

        feature_penalty = np.zeros(len(population))
        for feature, (min_val, max_val) in self.continuous_features.items():
            idx = self.feature_name_to_index.get(feature)
            if idx is not None:
                center = (min_val + max_val) / 2
                normalized_deviation = np.abs(population[:, idx] - center) / (max_val - min_val)
                feature_penalty += normalized_deviation ** 2

        total_penalty = combined_penalty + 0.5 * feature_penalty
        total_fitness = -total_penalty + combined_reward

        logging.info(f"Temp Reward: {temp_reward.mean():.4f}, Energy Reward: {energy_reward.mean():.4f}")
        logging.info(f"Combined Penalty: {combined_penalty.mean():.4f}, Feature Penalty: {feature_penalty.mean():.4f}")
        logging.info(f"Total Fitness: {total_fitness.mean():.4f}")

        return total_fitness

    def _select(self, population, fitnesses):
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-6
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            probabilities = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probabilities = fitnesses / total_fitness
        selected_indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
        return [population[i] for i in selected_indices]

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return child1, child2

    def _mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                feature_name = self.feature_names[i]
                if feature_name in self.continuous_features:
                    min_val, max_val = self.continuous_features[feature_name]
                    mutated_individual[i] = random.uniform(min_val, max_val)
                elif feature_name in self.categorical_features:
                    values = self.categorical_features[feature_name]
                    mutated_individual[i] = random.choice(values)
        return mutated_individual

    def optimize(self):
        population = self._generate_initial_population()
        best_individual = None
        best_fitness = -float('inf')
        feature_ranges_over_generations = {feature: [] for feature in self.feature_names}
        temps_per_gen = []
        energies_per_gen = []
        previous_best_fitness = -float('inf')

        for generation in range(self.generations):
            fitnesses = self._batch_fitness(population)

            population_df = pd.DataFrame(population, columns=self.feature_names)
            population_imputed = self.imputer.transform(population_df)
            population_scaled = self.scaler.transform(population_imputed)
            predictions = np.expm1(self.models[0].predict(population_scaled))
            temps_per_gen.append(predictions[:, 0])
            energies_per_gen.append(predictions[:, 1])

            # 记录每代的平均温度和平均活化能
            logging.info(f"Generation {generation + 1}: Mean Onset Temperature = {np.mean(predictions[:, 0]):.4f}")
            logging.info(f"Generation {generation + 1}: Mean Activation Energy = {np.mean(predictions[:, 1]):.4f}")

            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_individual = population[best_idx].copy()

            for feature_index, feature_name in enumerate(self.feature_names):
                feature_values = population[:, feature_index]
                feature_ranges_over_generations[feature_name].append((np.min(feature_values), np.max(feature_values)))

            selected_population = self._select(population, fitnesses)
            new_population = []
            elite_indices = np.argsort(fitnesses)[-2:]
            new_population.extend([population[i].copy() for i in elite_indices])

            while len(new_population) < self.population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([self._mutate(child1), self._mutate(child2)])

            population = np.array(new_population[:self.population_size])
            if best_fitness > previous_best_fitness:
                self.mutation_rate = self.initial_mutation_rate * (0.99 ** generation)
            else:
                self.mutation_rate = min(self.initial_mutation_rate, self.mutation_rate + 0.05)
            previous_best_fitness = best_fitness

            logging.info(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}, Mutation Rate = {self.mutation_rate:.4f}")

        if best_individual is not None:
            best_individual_dict = {self.feature_names[i]: best_individual[i] for i in range(len(best_individual))}
            best_individual_df = pd.DataFrame([best_individual], columns=self.feature_names)
            best_individual_imputed = self.imputer.transform(best_individual_df)
            best_individual_scaled = self.scaler.transform(best_individual_imputed)
            best_predictions = np.expm1(self.models[0].predict(best_individual_scaled)[0])
        else:
            best_individual_dict, best_predictions = None, None

        return best_individual_dict, best_predictions, feature_ranges_over_generations, temps_per_gen, energies_per_gen
# --- Heatmap Plotting Function ---
def plot_mean_temp_energy_trends(all_temps_per_gen_all_stages, all_energies_per_gen_all_stages, all_generation_numbers, stage_boundaries, save_path_prefix):
    # Organize data by generation
    data_by_generation = {}
    for i in range(len(all_temps_per_gen_all_stages)):
        temps_gen = all_temps_per_gen_all_stages[i]
        energies_gen = all_energies_per_gen_all_stages[i]
        generation_num = all_generation_numbers[i]

        if generation_num not in data_by_generation:
            data_by_generation[generation_num] = {'temps': [], 'energies': []}
        data_by_generation[generation_num]['temps'].extend(temps_gen)
        data_by_generation[generation_num]['energies'].extend(energies_gen)

    # Calculate mean values per generation
    generations = sorted(data_by_generation.keys())
    mean_temps = [np.mean(data_by_generation[gen]['temps']) for gen in generations]
    mean_energies = [np.mean(data_by_generation[gen]['energies']) for gen in generations]

    # Apply moving average to smooth the data
    window_size = 50  # 移动平均窗口大小
    mean_temps_smooth = pd.Series(mean_temps).rolling(window=window_size, min_periods=1, center=True).mean()
    mean_energies_smooth = pd.Series(mean_energies).rolling(window=window_size, min_periods=1, center=True).mean()

    # Sample data at larger intervals to reduce density
    sampling_interval = 50  # 增加采样间隔
    sampled_generations = generations[::sampling_interval]
    sampled_mean_temps = mean_temps_smooth[::sampling_interval]
    sampled_mean_energies = mean_energies_smooth[::sampling_interval]

    # Set up publication-quality plotting style
    try:
        plt.style.use('seaborn-whitegrid')  # Use the modern seaborn style name
    except (ValueError, OSError):
        print("Seaborn style 'seaborn-whitegrid' not available. Falling back to 'ggplot' style.")
        plt.style.use('ggplot')  # Fallback to a matplotlib built-in style

    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 2.5,
    })

    # Define the color gradient
    colors = ["#1E90FF", "#FF4500"]  # 更鲜明的颜色：深蓝到橙红
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    # Plot 1: Mean Dehydrogenation Temperature
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create a gradient-colored line
    points = np.array([sampled_generations, sampled_mean_temps]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(min(sampled_mean_temps), max(sampled_mean_temps))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(sampled_mean_temps))
    ax.add_collection(lc)
    
    # Add stage boundaries
    for stage_gen in stage_boundaries:
        ax.axvline(x=stage_gen, color='gray', linestyle='--', alpha=0.3)
    
    # Add stage labels (every 5 stages)
    for i, stage_gen in enumerate(stage_boundaries[:-1]):
        if (i + 1) % 5 != 0:  # 只显示每 5 个阶段的标签
            continue
        next_gen = stage_boundaries[i + 1]
        mid_point = (stage_gen + next_gen) / 2
        ax.text(mid_point, max(sampled_mean_temps) * 0.95, f"{i + 1}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add target range reference lines
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target Range')
    ax.axhline(y=150, color='green', linestyle='--', alpha=0.5)

    ax.set_title('Mean Dehydrogenation Temperature Across Generations', pad=15)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Temperature (°C)')
    
    # Dynamically adjust y-axis limits
    temp_min, temp_max = min(sampled_mean_temps), max(sampled_mean_temps)
    temp_range = temp_max - temp_min
    ax.set_ylim(max(temp_min - 0.1 * temp_range, -50), min(temp_max + 0.1 * temp_range, 200))  # 动态调整纵轴范围
    ax.set_xlim(min(generations), max(generations))
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_mean_temp.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mean temperature plot saved to: {save_path_prefix}_mean_temp.png")

    # Plot 2: Mean Activation Energy
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create a gradient-colored line
    points = np.array([sampled_generations, sampled_mean_energies]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(min(sampled_mean_energies), max(sampled_mean_energies))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(sampled_mean_energies))
    ax.add_collection(lc)
    
    # Add stage boundaries
    for stage_gen in stage_boundaries:
        ax.axvline(x=stage_gen, color='gray', linestyle='--', alpha=0.3)
    
    # Add stage labels (every 5 stages)
    for i, stage_gen in enumerate(stage_boundaries[:-1]):
        if (i + 1) % 5 != 0:  # 只显示每 5 个阶段的标签
            continue
        next_gen = stage_boundaries[i + 1]
        mid_point = (stage_gen + next_gen) / 2
        ax.text(mid_point, max(sampled_mean_energies) * 0.95, f"{i + 1}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add target range reference lines
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target Range')
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5)

    ax.set_title('Mean Activation Energy Across Generations', pad=15)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Activation Energy (kJ/mol)')
    
    # Dynamically adjust y-axis limits
    energy_min, energy_max = min(sampled_mean_energies), max(sampled_mean_energies)
    energy_range = energy_max - energy_min
    ax.set_ylim(max(energy_min - 0.1 * energy_range, -50), min(energy_max + 0.1 * energy_range, 150))  # 动态调整纵轴范围
    ax.set_xlim(min(generations), max(generations))
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Activation Energy (kJ/mol)', fontsize=12)
    
    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_mean_energy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mean activation energy plot saved to: {save_path_prefix}_mean_energy.png")
    
# --- Main Script ---
if __name__ == "__main__": # Add main block for better script structure
    seed =  22 # 可以选择任意整数作为种子42 15 36 16 20
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"optimization_log_{timestamp}.txt"
    log_file_path = os.path.join('logs', log_file_name)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console_handler)

    logging.info("Starting multi-objective optimization process for Onset_Temperature and Activation_Energy...")

    initial_continuous_features = {
        'hydrogen desorption catalysts particle size': (1, 100),
        'catalytic mass fraction(wt%)': (0, 90),
        'ball milling rotating speed(rpm)': (0, 2500),
        'ball milling time(min)': (0, 1800),
        'CBM': (-12, 18),
        'VBM': (-12, 18),
        'Density (g/cm^3)': (0, 15),
        'Energy per Atom (eV)': (-26, 0),
        'Band Gap (eV)': (0, 8),
        'Formation Energy (eV/atom)': (-4, 4),
        'Fermi Energy (eV)': (-10, 10),
        'Equilibrium Reaction Energy per Atom (eV/atom)': (-4, 4),
        'Metal_Count': (0, 3)
    }

    categorical_features = {
        'Catalysts_Component_Encoded': list(range(0, 818)),
        'Catalyst_Category_Encoded': list(range(0, 39))
    }

    population_size = 500
    crossover_rate = 0.9
    initial_mutation_rate = 0.8  # 提高初始变异率
    generations = 500
    num_stages = 20

    optimization_tasks = {
        "MultiOutput": {
            "model_path": 'models/MultiOutput_Model/XGBoost.joblib',
            "feature_names_path": 'models/MultiOutput_Model/feature_names.joblib',
            "scaler_path": 'models/MultiOutput_Model/scaler.joblib',
            "imputer_path": 'models/MultiOutput_Model/imputer.joblib',
            "label_encoder_path": 'models/MultiOutput_Model/label_encoder_component.joblib',
        },
    }

    target_onset_temperatures = [0, 150]
    target_activation_energies = [0, 50]

    current_continuous_features = initial_continuous_features.copy()
    all_temps_per_gen_all_stages = []
    all_energies_per_gen_all_stages = []
    all_generation_numbers = []

    try:
        le_component = joblib.load(optimization_tasks["MultiOutput"]["label_encoder_path"])  # 修正拼写
    except FileNotFoundError:
        le_component = None
        logging.warning("Label encoder file not found. Skipping component suggestion.")

    global_best_sum = float('inf')
    global_best_individual = None
    global_best_predictions = None
    previous_best_individual = None

    for stage in range(num_stages):
        logging.info(f"Stage {stage + 1} of {num_stages}")

        use_previous = stage < 5 and previous_best_individual is not None
        optimizer = GeneticAlgorithmOptimizer(
            model_paths=[optimization_tasks["MultiOutput"]["model_path"]],
            feature_names_path=optimization_tasks["MultiOutput"]["feature_names_path"],
            scaler_path=optimization_tasks["MultiOutput"]["scaler_path"],
            imputer_path=optimization_tasks["MultiOutput"]["imputer_path"],
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            initial_mutation_rate=initial_mutation_rate,
            continuous_features=current_continuous_features,
            categorical_features=categorical_features,
            target_temperatures=target_onset_temperatures,
            target_activation_energies=target_activation_energies,
            previous_best_individual=previous_best_individual if use_previous else None
        )

        best_individual, best_predictions, feature_ranges, temps_per_gen, energies_per_gen = optimizer.optimize()

        stage_start_gen = stage * generations
        all_temps_per_gen_all_stages.extend(temps_per_gen)
        all_energies_per_gen_all_stages.extend(energies_per_gen)
        all_generation_numbers.extend([stage_start_gen + i for i in range(generations)])

        logging.info(f"Experimental Guidance (Stage {stage + 1}):")
        if best_individual:
            logging.info("  - Best Individual Feature Values:")
            for feature, value in best_individual.items():
                logging.info(f"    {feature}: {value:.4f}")
            if 'Catalysts_Component_Encoded' in best_individual and le_component:
                encoded_value = int(best_individual['Catalysts_Component_Encoded'])
                try:
                    component_name = le_component.inverse_transform([encoded_value])[0]
                    logging.info(f"    Catalysts_Component: {component_name}")
                except ValueError:
                    logging.info(f"    Catalysts_Component: Unknown (Encoded: {encoded_value})")
            else:
                logging.info("    Catalysts_Component: N/A")
        if best_predictions is not None:
            logging.info(f"  - Best Predicted Onset_Temperature: {best_predictions[0]:.2f} °C")
            logging.info(f"  - Best Predicted Activation_Energy: {best_predictions[1]:.2f} kJ/mol")

        if best_predictions is not None:
            temp_penalty_global = 0
            energy_penalty_global = 0
            if target_onset_temperatures:
                lower_temp_target, upper_temp_target = target_onset_temperatures
                if best_predictions[0] > upper_temp_target:
                    temp_penalty_global = (best_predictions[0] - upper_temp_target)
                elif best_predictions[0] < lower_temp_target:
                    temp_penalty_global = (lower_temp_target - best_predictions[0])
            if target_activation_energies:
                lower_energy_target, upper_energy_target = target_activation_energies
                if best_predictions[1] > upper_energy_target:
                    energy_penalty_global = (best_predictions[1] - upper_energy_target)
                elif best_predictions[1] < lower_energy_target:
                    energy_penalty_global = (lower_energy_target - best_predictions[1])
            current_penalty_sum = temp_penalty_global + energy_penalty_global

        if current_penalty_sum < global_best_sum: # Corrected indentation
            global_best_sum = current_penalty_sum
            global_best_individual = best_individual
            global_best_predictions = best_predictions

        new_continuous_features = {}
        if best_predictions is not None and best_individual:
            temp_deviation = abs(best_predictions[0] - 75) / 75
            energy_deviation = abs(best_predictions[1] - 25) / 25
            deviation_factor = max(temp_deviation, energy_deviation, 0.1)
            for feature in current_continuous_features:
                if feature in best_individual:
                    current_min, current_max = current_continuous_features[feature]
                    best_value = best_individual[feature]
                    range_width = current_max - current_min
                    min_range_width = 0.2 * (initial_continuous_features[feature][1] - initial_continuous_features[feature][0])
                    new_range_width = max(deviation_factor * range_width, min_range_width)
                    new_min = max(best_value - new_range_width / 2, initial_continuous_features[feature][0])
                    new_max = min(best_value + new_range_width / 2, initial_continuous_features[feature][1])
                    new_continuous_features[feature] = (new_min, new_max)
                else:
                    new_continuous_features[feature] = current_continuous_features[feature]
            current_continuous_features = new_continuous_features
            previous_best_individual = [best_individual[f] for f in optimizer.feature_names]

        logging.info(f"Updated Feature Ranges (Stage {stage + 1}):")
        for feature, (min_val, max_val) in current_continuous_features.items():
            logging.info(f"    {feature}: ({min_val:.4f}, {max_val:.4f})")

    logging.info("Global Best Individual Across All Stages:")
    if global_best_individual:
        logging.info("  - Feature Values:")
        for feature, value in global_best_individual.items():
            logging.info(f"    {feature}: {value:.4f}")
        if 'Catalysts_Component_Encoded' in global_best_individual and le_component:
            encoded_value = int(global_best_individual['Catalysts_Component_Encoded'])
            try:
                component_name = le_component.inverse_transform([encoded_value])[0]
                logging.info(f"    Catalysts_Component: {component_name}")
            except ValueError:
                logging.info(f"    Catalysts_Component: Unknown (Encoded: {encoded_value})")
        else:
            logging.info("    Catalysts_Component: N/A")
        logging.info(f"  - Predicted Onset_Temperature: {global_best_predictions[0]:.2f} °C")
        logging.info(f"  - Predicted Activation_Energy: {global_best_predictions[1]:.2f} kJ/mol")
    else:
        logging.info("  - No best individual found.")

    # Replace the original plotting call
    stage_boundaries = [i * generations for i in range(num_stages + 1)]
    output_plots_dir = 'plots'
    plot_mean_temp_energy_trends(
        all_temps_per_gen_all_stages,
        all_energies_per_gen_all_stages,
        all_generation_numbers,
        stage_boundaries,
        os.path.join(output_plots_dir, f'temp_energy_trend_{timestamp}')
    )

    logging.info("Multi-stage optimization completed.")