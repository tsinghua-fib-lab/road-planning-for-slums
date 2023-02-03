import os
from pprint import pprint

import pygad
import setproctitle

import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'road_planning/envs'))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

from khrylib.utils import *
from road_planning.utils.config import Config
from road_planning.agents.road_planning_agent import RoadPlanningAgent


flags.DEFINE_string('root_dir', os.path.join(cwd,'train_data') , 'Root directory for writing '
                                                                      'logs/summaries/checkpoints.')
flags.DEFINE_string('slum_name', 'Epworth_Demo', 'data_dir')
flags.DEFINE_string('cfg', 'demo', 'Configuration file of rl training.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('mean_action', True, 'Whether to use greedy strategy.')
flags.DEFINE_bool('visualize', False, 'Whether to visualize the planning process.')
flags.DEFINE_bool('only_road', False, 'Whether to only visualize road planning.')
flags.DEFINE_bool('save_video', False, 'Whether to save a video of the planning process.')
flags.DEFINE_integer('global_seed', 0, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_integer('sol_per_pop', 20, 'The number of solutions per population.')
flags.DEFINE_integer('num_generations', 1, 'The number of generations.')
flags.DEFINE_integer('num_parents_mating', 2, 'The number of parents for mating.')
flags.DEFINE_integer('init_range_low', -5, 'Low range for gene initialization.')
flags.DEFINE_integer('init_range_high', 5, 'High range for gene initialization.')
flags.DEFINE_string('parent_selection_type', 'sss', 'Type of parent selection.')
flags.DEFINE_string('crossover_type', 'single_point', 'Type of crossover.')
flags.DEFINE_string('mutation_type', 'random', 'Type of mutation.')
flags.DEFINE_integer('mutation_percent_genes', 10, 'Percentage of genes for mutation.')

FLAGS = flags.FLAGS


def main_loop(_):

    setproctitle.setproctitle(f'road_planning_{FLAGS.cfg}_{FLAGS.global_seed}@suhy')

    cfg = Config(FLAGS.cfg,FLAGS.slum_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, 'ga')

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    """create agent"""
    agent = RoadPlanningAgent(cfg=cfg, dtype=dtype, device=device, num_threads=1,
                               training=False, checkpoint=0, restore_best_rewards=True)

    def fitness_func(solution, solution_idx):
        fitness, _ = agent.fitness_ga(solution, num_samples=1, mean_action=False, visualize=FLAGS.visualize)
        # fitness, _, _ = agent.fitness_ga(solution)
        return fitness

    def report_func(instance):
        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
        print(f'param:{best_solution}')
        print(f'Generation: {instance.generations_completed}')
        print(f'Best Fitness: {instance.best_solutions_fitness[-1]: .4f}')
        print(f'Last Generation Average Fitness: '
              f'{sum(instance.last_generation_fitness)/len(instance.last_generation_fitness): .4f}')
        print()

    # initial_population = agent.init_population(FLAGS.sol_per_pop)

    # def crossover_func(parents, offspring_size, ga_instance):
    #     offsprings = []
    #     idx = 0
    #     while len(offsprings) != offspring_size[0]:
    #         offspring = np.zeros(n, dtype=np.int32)

    #         parent1 = parents[idx % parents.shape[0], :].copy()
    #         parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
    #         facility_locations = np.arange(n)[(parent1 + parent2) > 0]
    #         random_indices = self._np_random.choice(facility_locations, p, replace=False)
    #         offspring[random_indices] = 1
    #         offsprings.append(offspring)

    #         idx += 1

    #     return np.array(offsprings)

    # def mutation_func(offsprings, ga_instance):
    #     offsprings = agent.mutation(offsprings)

    #     return offsprings

    ga_instance = pygad.GA(num_generations=FLAGS.num_generations,
                           num_parents_mating=FLAGS.num_parents_mating,
                           fitness_func=fitness_func,
                        #    initial_population = initial_population,
                           sol_per_pop=FLAGS.sol_per_pop,
                           num_genes=agent.node_dim + 5,
                           init_range_low=FLAGS.init_range_low,
                           init_range_high=FLAGS.init_range_high,
                           parent_selection_type=FLAGS.parent_selection_type,
                           keep_parents=1,
                           crossover_type=FLAGS.crossover_type,
                           mutation_type=FLAGS.mutation_type,
                           mutation_percent_genes=FLAGS.mutation_percent_genes,
                           on_generation=report_func,
                           stop_criteria="saturate_10",
                           random_seed=cfg.seed)

    ga_instance.run()
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=best_solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))

    agent.save_ga(best_solution, best_solution_fitness)
    best_solution,_ =agent.load_ga()

    _, plan = agent.fitness_ga(best_solution,visualize=True)
    # final,dis,total_cost = agent.fitness_ga(best_solution,visualize=True)
    # print(final,dis,total_cost)

    pprint(plan, indent=4, sort_dicts=False)


if __name__ == '__main__':
    app.run(main_loop)
