from reinforcement_learning.stat_wrapper import BaseStatWrapper


class RewardWrapper(BaseStatWrapper):
    def __init__(
        # BaseStatWrapper args
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=True,
        # Custom reward wrapper args
        heal_bonus_weight=0,
        explore_bonus_weight=0,
        clip_unique_event=3,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.stat_prefix = stat_prefix

        self.heal_bonus_weight = heal_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

    def reset(self, **kwargs):
        """Called at the start of each episode"""
        self._reset_reward_vars()
        return super().reset(**kwargs)

    def _reset_reward_vars(self):
        self._history = {
            agent_id: {
                "prev_price": 0,
                "prev_moves": [],
            }
            for agent_id in self.env.possible_agents
        }

    """
    @functools.cached_property
    def observation_space(self):
        '''If you modify the shape of features, you need to specify the new obs space'''
        return super().observation_space
    """

    def observation(self, agent_id, agent_obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        # Mask the price of the previous action, to encourage agents to explore new prices
        agent_obs["ActionTargets"]["Sell"]["Price"][self._history[agent_id]["prev_price"]] = 0
        return agent_obs

    def action(self, agent_id, agent_atn):
        """Called before actions are passed from the model to the environment"""
        # Keep track of the previous price and moves for each agent
        self._history[agent_id]["prev_price"] = agent_atn["Sell"]["Price"]
        self._history[agent_id]["prev_moves"].append(agent_atn["Move"]["Direction"])
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        print(f"Agent {agent_id} received reward: {reward}")
        print(f"Agent {agent_id} terminated: {terminated}")
        print(f"Agent {agent_id} truncated: {truncated}")
        print(f"Agent {agent_id} info: {info}")
        print(f"Agent {agent_id} data: {self.env.realm.players[agent_id]}")
        realm = self.env.realm

        # Add "Healing" score based on health increase and decrease, due to food and water
        healing_bonus = 0
        if self.heal_bonus_weight > 0 and agent_id in realm.players:
            if realm.players[agent_id].resources.health_restore > 0:
                healing_bonus = self.heal_bonus_weight

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._unique_events[agent_id]
        uniq = self._unique_events[agent_id]
        explore_bonus = 0
        if self.explore_bonus_weight > 0 and uniq["curr_count"] > uniq["prev_count"]:
            explore_bonus = min(self.clip_unique_event, uniq["curr_count"] - uniq["prev_count"])
            explore_bonus *= self.explore_bonus_weight

        reward += healing_bonus + explore_bonus

        return reward, terminated, truncated, info
