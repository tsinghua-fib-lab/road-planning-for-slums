import os
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
flags.DEFINE_bool('infer', False, 'Train or Infer.')
flags.DEFINE_bool('visualize', False, 'visualize plan.')
flags.DEFINE_enum('agent', 'rl-ngnn',
                  ['rl-sgnn', 'rl-ngnn', 'rl-mlp', 'rl-rmlp',
                   'random', 'road-cost'],
                  'Agent type.')
flags.DEFINE_integer('num_threads', 1, 'The number of threads for sampling trajectories.')
flags.DEFINE_bool('use_nvidia_gpu', True, 'Whether to use Nvidia GPU for acceleration.')
flags.DEFINE_integer('gpu_index', 0,'GPU ID.')
flags.DEFINE_integer('global_seed', 0, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('iteration', '0', 'The start iteration. Can be number or best. If not 0, the agent will load from '
                                      'a saved checkpoint.')
flags.DEFINE_bool('restore_best_rewards', True, 'Whether to restore the best rewards from a saved checkpoint. '
                                                'True for resume training. False for finetune with new reward.')

FLAGS = flags.FLAGS


def train_one_iteration(agent: RoadPlanningAgent, iteration: int) -> None:
    """Train one iteration"""
    agent.optimize(iteration)
    agent.save_checkpoint(iteration)

    """clean up gpu memory"""
    torch.cuda.empty_cache()


def main_loop(_):

    setproctitle.setproctitle(f'road_planning_{FLAGS.cfg}_{FLAGS.global_seed}@suhy')

    cfg = Config(FLAGS.cfg, FLAGS.slum_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if FLAGS.use_nvidia_gpu and torch.cuda.is_available():
        device = torch.device('cuda', index=FLAGS.gpu_index)
    else:
        device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    """create agent"""
    agent = RoadPlanningAgent(cfg=cfg, dtype=dtype, device=device, num_threads=FLAGS.num_threads,
                               training=True, checkpoint=checkpoint, restore_best_rewards=FLAGS.restore_best_rewards)
    
    if FLAGS.infer:
        agent.infer(visualize=FLAGS.visualize)
    else:

        start_iteration = agent.start_iteration
        for iteration in range(start_iteration, cfg.max_num_iterations):
            train_one_iteration(agent, iteration)

    agent.logger.info('training done!')


if __name__ == '__main__':
    # flags.mark_flags_as_required([
    #   'cfg'
    # ])
    app.run(main_loop)
