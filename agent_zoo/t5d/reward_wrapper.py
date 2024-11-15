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
        # Custom args
        health_bonus_weight=0,
        water_bonus_weight=0,
        food_bonus_weight=0,
        gold_bonus_weight=0,
        meele_exp_bonus_weight=0,
        mage_exp_bonus_weight=0,
        range_exp_bonus_weight=0,
        fishing_exp_bonus_weight=0,
        herbalism_exp_bonus_weight=0,
        prospecting_exp_bonus_weight=0,
        carving_exp_bonus_weight=0,
        alchemy_exp_bonus_weight=0,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.stat_prefix = stat_prefix

        self.heal_bonus_weight = heal_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

        self.bonus_weights = {
            # Resources
            "gold": gold_bonus_weight,
            "health": health_bonus_weight,
            "food": food_bonus_weight,
            "water": water_bonus_weight,
            # Combat
            "melee_exp": meele_exp_bonus_weight,
            "range_exp": range_exp_bonus_weight,
            "mage_exp": mage_exp_bonus_weight,
            # Gathering
            "fishing_exp": fishing_exp_bonus_weight,
            "herbalism_exp": herbalism_exp_bonus_weight,
            "prospecting_exp": prospecting_exp_bonus_weight,
            "carving_exp": carving_exp_bonus_weight,
            "alchemy_exp": alchemy_exp_bonus_weight,
        }

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

        self.data = {
            # Resources
            "gold": 0,
            "health": 100,
            "food": 100,
            "water": 100,
            # Combat
            "melee_exp": 0,
            "range_exp": 0,
            "mage_exp": 0,
            # Gathering
            "fishing_exp": 0,
            "herbalism_exp": 0,
            "prospecting_exp": 0,
            "carving_exp": 0,
            "alchemy_exp": 0,
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
        agent = self.env.realm.players[agent_id]
        for resource in self.data:
            reward += (
                getattr(agent.resources, resource).val - self.data[resource]
            ) * self.bonus_weights[resource]
            self.data[resource] = getattr(agent.resources, resource).val

        return reward, terminated, truncated, info
