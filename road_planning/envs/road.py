import logging
import math
import copy
import pickle
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
from geopandas import GeoDataFrame
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# from urban_planning.envs.plan_client import PlanClient
from road_planning.envs.my_graph import MyGraph
from road_planning.utils.config import Config


class InfeasibleActionError(ValueError):
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error.

        Args:
          action: Infeasible action that was performed.
          mask: The mask associated with the current observation. mask[action] is
            `0` for infeasible actions.
        """
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(
            self.action, self.mask)


def load_graph(slum):
    with open("data/{}.mg".format(slum), 'rb') as mgfile:
        mg = pickle.loads(mgfile.read())
        mg.define_roads()
        mg.define_interior_parcels()
        mg.td_dict_init()
        mg.feature_init()
    return mg

def reward_info_function(mg: MyGraph, name: Text,
                         travel_distance_weight: float,
                         road_cost_weight: float) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """

    travel_distance = travel_distance_weight * mg.travel_distance()
    road_cost = road_cost_weight * mg.road_cost()
    connect_reward = mg.connected_ration()

    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    # face2face_avg = mg.face2face_avg()
    total_road_cost = mg.total_cost()

    # print(connect_reward , travel_distance , road_cost)
    # print(face2face_avg,total_road_cost)
    return connect_reward + travel_distance + road_cost, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': 0,
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,
    }


class RoadEnv:
    """ Environment for urban planning."""
    FAILURE_REWARD = -4.0
    INTERMEDIATE_REWARD = -4.0

    def __init__(self,
                 cfg: Config,
                 is_eval: bool = False,
                 reward_info_fn=reward_info_function):
        #  Callable[[MyGraph, Text, float, float], Tuple[float, Dict]] = reward_info_function):
        self.cfg = cfg
        self._is_eval = is_eval
        self._frozen = False
        self._action_history = []
        self._mg = load_graph(cfg.slum)
        self._cmg = copy.deepcopy(self._mg)

        self._reward_info_fn = partial(reward_info_fn,
                                       travel_distance_weight=cfg.reward_specs.get('dis_weight',1.0)*12,
                                       road_cost_weight=cfg.reward_specs.get('cost_weight',1.0)*-0.8)
        
        self.build_ration = cfg.reward_specs.get('build_ration', 0.5)

        self._all_stages = ['connecting', 'full_connected', 'done']
        self._set_stage()
        self._done = False
        self._set_cached_reward_info()

    def _set_stage(self):
        """
        Set the stage.
        """
        self._connecting_steps = 0
        self._full_connected_steps = 0
        self._stage = 'connecting'

    def _compute_total_road_steps(self) -> None:
        """
        Compute the total number of road steps.
        """
        self._total_road_steps = len(self._mg.myedges()) - len(
            self._mg.road_edges)

    def _set_cached_reward_info(self):
        """
        Set the cached reward.
        """
        if not self._frozen:
            self._cached_life_circle_reward = -1.0
            self._cached_greeness_reward = -1.0
            self._cached_concept_reward = -1.0

            self._cached_life_circle_info = dict()
            self._cached_concept_info = dict()

            self._cached_land_use_reward = -1.0
            self._cached_land_use_gdf = self.snapshot_land_use()

    def get_reward_info(self) -> Tuple[float, Dict]:
        """
        Returns the RL reward and info.

        Returns:
            The RL reward.
            Info dictionary.
        """
        if self._stage == 'connecting':
            return self._reward_info_fn(self._mg, 'connecting')
        elif self._stage == 'full_connected':
            return self._reward_info_fn(self._mg, 'full_connected')
        elif self._stage == 'done':
            return self._reward_info_fn(self._mg, 'full_connected')
        else:
            raise ValueError('Invalid Stage')

    # def _get_all_reward_info(self) -> Tuple[float, Dict]:
    #     """
    #     Returns the entire reward and info. Used for loaded plans.
    #     """
    #     land_use_reward, land_use_info = self._reward_info_fn(self._mg, 'land_use')
    #     road_reward, road_info = self._reward_info_fn(self._mg, 'road')
    #     reward = land_use_reward + road_reward
    #     info = {
    #         'road_network': road_info['road_network'],
    #         'life_circle': land_use_info['life_circle'],
    #         'greeness': land_use_info['greeness'],
    #         'road_network_info': road_info['road_network_info'],
    #         'life_circle_info': land_use_info['life_circle_info']
    #     }
    #     return reward, info

    def eval(self):
        """
        Set the environment to eval mode.
        """
        self._is_eval = True

    def train(self):
        """
        Set the environment to training mode.
        """
        self._is_eval = False

    def get_numerical_feature_size(self):
        """
        Returns the numerical feature size.

        Returns:
            feature_size (int): the feature size.
        """
        return self._mg.get_numerical_feature_size()

    def get_node_dim(self):
        """
        Returns the node dimension.

        Returns:
            node_dim (int): the node dimension.
        """
        # dummy_land_use = self._get_dummy_land_use()

        return 10

    # def _get_dummy_land_use(self):
    #     """
    #     Returns the dummy land use.

    #     Returns:
    #         land_use (dictionary): the dummy land use.
    #     """
    #     dummy_land_use = dict()        
    #     dummy_land_use['x'] = 0.5
    #     dummy_land_use['y'] = 0.5
    #     dummy_land_use['area'] = 0.0
    #     dummy_land_use['length'] = 0.0
    #     dummy_land_use['width'] = 0.0
    #     dummy_land_use['height'] = 0.0
    #     dummy_land_use['rect'] = 0.5
    #     dummy_land_use['eqi'] = 0.5
    #     dummy_land_use['sc'] = 0.5
    #     return dummy_land_use

    def _full_connected(self):
        if len(self._mg.interior_parcels) == 0:
            return True
        else:
            return False

    def _get_stage_obs(self) -> int:
        """
        Returns the current stage observation.

        Returns:
            obs (int): the current stage index.
        """
        return np.eye(len(self._all_stages))[self._all_stages.index(
            self._stage)]

    def _get_obs(self) -> List:
        """
        Returns the observation.

        Returns:
            observation (object): the observation
        """


        numerical, node_feature, edge_part_feature, edge_index, edge_mask = self._mg.get_obs()
        stage = self._get_stage_obs()

        return [
            numerical, node_feature, edge_part_feature, edge_index, edge_mask, stage
        ]

    def build_road(self, action: List):
        """
        Builds the road.

        Args:
            action (int): the action.
        """
        # print('action')
        self._mg.build_road_from_action(action)

    def snapshot_land_use(self):
        """
        Snapshot the land use.
        """
        return self._mg.snapshot()

    def build_all_road(self):
        """
        Build all the road.
        """
        self._mg.add_all_road()

    def transition_stage(self):
        """
        Transition to the next stage.
        """
        if self._stage == 'connecting':
            self._stage = 'full_connected'
        elif self._stage == 'full_connected':
            self._done = True
            self._stage = 'done'
        else:
            raise ValueError('Unknown stage: {}'.format(self._stage))
        
    def save_step_data(self):
        self._mg.save_step_data()

    def failure_step(self, logging_str, logger):
        """
        Logging and reset after a failure step.
        """
        logger.info('{}: {}'.format(logging_str, self._action_history))
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greeness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info

    def step(self, action: List,
             logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, you are responsible for calling `reset()` to reset
        the environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (np.ndarray of size 2): The action to take.
                                           1 is the land_use placement action.
                                           1 is the building road action.
            logger (Logger): The logger.

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        else:
            if self._stage == 'connecting':
                self._action_history.append(action)
                self.build_road(action)

                self._connecting_steps += 1
                if self._connecting_steps >= math.floor(
                        self._total_road_steps *
                        self.build_ration) or self._full_connected():
                    self.transition_stage()

            elif self._stage == 'full_connected':
                self._action_history.append(action)
                self.build_road(action)

                self._full_connected_steps += 1
                if (self._full_connected_steps + self._connecting_steps >
                        self._total_road_steps * self.build_ration):
                    self.transition_stage()

            reward, info = self.get_reward_info()
            if self._stage == 'done':
                self.save_step_data()

        return self._get_obs(), reward, self._done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation from the reset
        """
        self._mg = copy.deepcopy(self._cmg)
        self._action_history = []
        self._set_stage()
        self._done = False
        self._compute_total_road_steps()

        return self._get_obs()

    # @staticmethod
    # def filter_land_use_road(gdf: GeoDataFrame) -> GeoDataFrame:
    #     """
    #     Filter out the land use and road features.
    #     """
    #     land_use_road_gdf = copy.deepcopy(
    #         gdf[(gdf['existence'] == True)
    #             & (gdf['type'] != city_config.OUTSIDE) &
    #             (gdf['type'] != city_config.BOUNDARY) &
    #             (gdf['type'] != city_config.INTERSECTION)])
    #     return land_use_road_gdf

    # @staticmethod
    # def filter_road_boundary(gdf: GeoDataFrame) -> GeoDataFrame:
    #     """
    #     Filter out the road and boundary features.
    #     """
    #     road_boundary_gdf = copy.deepcopy(
    #         gdf[(gdf['existence'] == True)
    #             & ((gdf['type'] == city_config.ROAD)
    #                | (gdf['type'] == city_config.BOUNDARY))])
    #     return road_boundary_gdf

    # @staticmethod
    # def _add_legend_to_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
    #     """
    #     Add legend to the gdf.
    #     """
    #     gdf['legend'] = gdf['type'].apply(
    #         lambda x: city_config.LAND_USE_ID_MAP_INV[x])
    #     return gdf

    def plot_and_save_gdf(self,
                          save_fig: bool = False,
                          path: Text = None,
                          show=False) -> None:
        """
        Plot and save the gdf.
        """
        self._mg.plot_roads(new_plot=True, update=True)
        if save_fig:
            assert path is not None
            plt.savefig(path, format='svg', transparent=True)
        if show:
            plt.show()
        plt.close()

    def visualize(self,
                  save_fig: bool = False,
                  path: Text = None,
                  show=False) -> None:
        """
        Visualize the city plan.
        """
        self.plot_and_save_gdf(save_fig, path, show)

    # def load_plan(self, gdf: GeoDataFrame) -> None:
    #     """
    #     Load a city plan.
    #     """
    #     self._plc.load_plan(gdf)

    # def score_plan(self, verbose=True) -> Tuple[float, Dict]:
    #     """
    #     Score the city plan.
    #     """
    #     reward, info = self._get_all_reward_info()
    #     if verbose:
    #         print(f'reward: {reward}')
    #         pprint(info, indent=4, sort_dicts=False)
    #     return reward, info

    # def get_init_plan(self) -> Dict:
    #     """
    #     Get the gdf of the city plan.
    #     """
    #     return self._plc.get_init_plan()
