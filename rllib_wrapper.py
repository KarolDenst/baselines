from pdb import set_trace as T
import numpy as np

from collections import defaultdict

from tqdm import tqdm
import gym
import wandb
import trueskill

import torch
from torch import nn
from torch.nn.utils import rnn

from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import nmmo

from neural.policy import Recurrent

class RLlibPolicy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      #self.space  = actionSpace(self.config).spaces
      self.model  = Recurrent(self.config)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      logitDict, state = self.model(input_dict['obs'], state, seq_lens)

      logits = []
      #Flatten structured logits for RLlib
      #TODO: better to use the space directly here in case of missing keys
      for atnKey, atn in sorted(logitDict.items()):
         for argKey, arg in sorted(atn.items()):
            logits.append(arg)

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn


class RLlibEnv(nmmo.Env, rllib.MultiAgentEnv):
   '''Wrapper class for using Neural MMO with RLlib'''
   def __init__(self, config):
      self.config = config['config']
      super().__init__(self.config)

   def render(self):
      #Patch for RLlib dupe rendering bug
      if not self.config.RENDER:
         return

      super().render()

   def step(self, decisions):
      obs, rewards, dones, infos = super().step(decisions)
      config = self.config
      ts = config.TEAM_SPIRIT
      
      if config.COOPERATIVE:
          #Union of task rewards across population
          team_rewards = defaultdict(lambda: defaultdict(int))
          populations = {}
          for entID, info in infos.items():
              pop = info.pop('population')
              populations[entID] = pop
              team = team_rewards[pop]
              for task, reward in info.items():
                  team[task] = max(team[task], reward)

          #Team spirit interpolated between agent and team summed task rewards
          for entID, reward in rewards.items():
              pop = populations[entID]
              rewards[entID] = ts*sum(team_rewards[pop].values()) + (1-ts)*reward

      dones['__all__'] = False
      test = config.EVALUATE or config.RENDER
      
      if config.EVALUATE:
         horizon = config.EVALUATION_HORIZON
      else:
         horizon = config.TRAIN_HORIZON

      population  = len(self.realm.players) == 0
      hit_horizon = self.realm.tick >= horizon
      
      if not config.RENDER and (hit_horizon or population):
         dones['__all__'] = True

      return obs, rewards, dones, infos

class RLlibOverlayRegistry(nmmo.OverlayRegistry):
   '''Host class for RLlib Map overlays'''
   def __init__(self, realm):
      super().__init__(realm.config, realm)

      self.overlays['values']       = Values
      self.overlays['attention']    = Attention
      self.overlays['tileValues']   = TileValues
      self.overlays['entityValues'] = EntityValues

class RLlibOverlay(nmmo.Overlay):
   '''RLlib Map overlay wrapper'''
   def __init__(self, config, realm, trainer, model):
      super().__init__(config, realm)
      self.trainer = trainer
      self.model   = model

class Attention(RLlibOverlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      tiles      = self.realm.realm.map.tiles
      players    = self.realm.realm.players

      attentions = defaultdict(list)
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         player = players[playerID]
         r, c   = player.pos

         rad     = self.config.NSTIM
         obTiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()

         for tile, a in zip(obTiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      sz    = self.config.TERRAIN_SIZE
      data  = np.zeros((sz, sz))
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = nmmo.overlay.twoTone(data)
      self.realm.register(colorized)

class Values(RLlibOverlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      players = self.realm.realm.players
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         r, c = players[playerID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = nmmo.overlay.twoTone(self.values[:, :])
      self.realm.register(colorized)

def zeroOb(ob, key):
   for k in ob[key]:
      ob[key][k] *= 0

class GlobalValues(RLlibOverlay):
   '''Abstract base for global value functions'''
   def init(self, zeroKey):
      if self.trainer is None:
         return

      print('Computing value map...')
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.dense()
      values    = 0 * self.values

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in tqdm(obs):
         ob             = obs[agentID]
         batch[agentID] = ob
         zeroOb(ob, zeroKey)
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c         = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = nmmo.overlay.twoTone(values)

   def register(self, obs):
      print('Computing Global Values. This requires one NN pass per tile')
      self.init()

      self.realm.register(self.colorized)

class TileValues(GlobalValues):
   def init(self, zeroKey='Entity'):
      '''Compute a global value function map excluding other agents. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)

class EntityValues(GlobalValues):
   def init(self, zeroKey='Tile'):
      '''Compute a global value function map excluding tiles. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)


class Trainer:
   def __init__(self, config, env=None, logger_creator=None):
      super().__init__(config, env, logger_creator)
      self.env_config = config['env_config']['config']

      #1/sqrt(2)=76% win chance within beta, 95% win chance vs 3*beta=100 SR
      trueskill.setup(mu=1000, sigma=2*100/3, beta=100/3, tau=2/3, draw_probability=0)

      self.ratings = [{agent.__name__: trueskill.Rating(mu=1000, sigma=2*100/3)}
            for agent in set(self.env_config.EVAL_AGENTS)]

      self.reset_scripted()

   @classmethod
   def name(cls):
      return cls.__bases__[0].__name__

   def reset_scripted(self):
      for rating_dict in self.ratings:
         for agent, rating in rating_dict.items():
            if agent == 'Combat':
               rating_dict[agent] = trueskill.Rating(mu=1500, sigma=1)

   def post_mean(self, stats):
      for key, vals in stats.items():
          if type(vals) == list:
              stats[key] = np.mean(vals)

   def train(self):
      stats = super().train()
      self.post_mean(stats['custom_metrics'])
      return stats

   def evaluate(self):
      stat_dict = super().evaluate()
      stats = stat_dict['evaluation']['custom_metrics']
 
      ranks = {agent.__name__: -1 for agent in set(self.env_config.EVAL_AGENTS)}
      for key in list(stats.keys()):
         if key.startswith('Rank_'):
             stat = stats[key]
             del stats[key]
             agent = key[5:]
             ranks[agent] = stat

      #Getting a type(int) exception?
      #Either install error or (old) achievement system is off?
      ranks = list(ranks.values())
      assert type(ranks[0]) != int, 'Incorrect RLlib install. See required baseline setup on the install docs'
      nEnvs = len(ranks[0])
      
      #Once RLlib adds better custom metric support,
      #there should be a cleaner way to divide episodes into blocks
      for i in range(nEnvs): 
         env_ranks = [e[i] for e in ranks]
         self.ratings = trueskill.rate(self.ratings, env_ranks)
         self.reset_scripted()

      for rating in self.ratings:
         key  = 'SR_{}'.format(list(rating.keys())[0])
         val  = list(rating.values())[0]
         stats[key] = val.mu
     
      return stat_dict

def PPO(config):
   class PPO(Trainer, rllib.agents.ppo.ppo.PPOTrainer): pass
   extra_config = {'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE}
   return PPO, extra_config

def APPO(config):
   class APPO(Trainer, rllib.agents.ppo.appo.APPOTrainer): pass
   return APPO, {}

def Impala(config):
   class Impala(Trainer, rllib.agents.impala.impala.ImpalaTrainer): pass
   return Impala, {}

###############################################################################
### Logging
class RLlibLogCallbacks(DefaultCallbacks):
   def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
      assert len(base_env.envs) == 1, 'One env per worker'
      env    = base_env.envs[0]

      invMap = {agent.policyID: agent for agent in env.config.AGENTS}

      stats      = logs['Stats']
      policy_ids = stats['PolicyID']
 
      logs = env.terminal()
      for key, vals in logs['Stats'].items():
         policy_stat = defaultdict(list)
         for policy, v in zip(policy_ids, vals):
             policy_stat(policy).append(v)
         for policy, vals in policy_stat.items():
             episode.custom_metrics[key] = np.mean(vals)

      if not env.config.EVALUATE:
         return 

      agents = defaultdict(list)

      scores     = stats['Task_Reward']


      for policyID, score in zip(policy_ids, scores):
         policy = invMap[policyID]
         agents[policy].append(score)

      for agent in agents:
         agents[agent] = np.mean(agents[agent])

      policies = list(agents.keys())
      scores   = list(agents.values())

      idxs     = np.argsort(-np.array(scores))

      for rank, idx in enumerate(idxs):
          key = 'Rank_{}'.format(policies[idx].__name__)
          episode.custom_metrics[key] = rank
