import numpy as np
import pandas as pd
import warnings
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from config import config

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class UpliftModel:
    """
    X-Learner for CATE estimation
    Predicts incremental CSAT from assigning customer to specific agent
    """
    def __init__(self):
        self.mu0_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.mu1_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.tau0_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.tau1_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.is_trained = False
    
    def _featurize(self, customer: pd.Series, agent: pd.Series) -> np.ndarray:
        """Create feature vector for (customer, agent) pair"""
        features = []
        
        # Customer features
        features.append(config.SKILLS.index(customer['skill_needed']))
        features.append(['low', 'medium', 'high', 'vip'].index(customer['priority']))
        features.append(customer['complexity'])
        features.append(customer['wait_time'])
        features.append(config.CHANNELS.index(customer['channel']))
        
        # Agent features
        features.append(agent[f'skill_{customer["skill_needed"]}'])
        features.append(agent['avg_csat'])
        features.append(agent['avg_aht'])
        features.append(agent['experience_years'])
        features.append(agent['current_load'])
        
        # Interaction features
        features.append(1 if customer['channel'] in agent['preferred_channels'] else 0)
        features.append(agent[f'skill_{customer["skill_needed"]}'] * customer['complexity'])
        
        return np.array(features)
    
    def train(self, historical_data: pd.DataFrame, agents_df: pd.DataFrame):
        """Train X-Learner on historical data"""
        print("Training Uplift Model (X-Learner)...")
        
        # Create feature matrix
        X = []
        y = []
        treatment = []  # 1 if high-skill match, 0 otherwise
        
        for _, row in historical_data.iterrows():
            agent = agents_df[agents_df['agent_id'] == row['agent_id']].iloc[0]
            
            # Mock customer from historical data
            customer = pd.Series({
                'skill_needed': row['customer_skill'],
                'priority': row['customer_priority'],
                'complexity': np.random.beta(2, 5),
                'wait_time': np.random.exponential(2),
                'channel': row['channel']
            })
            
            feat = self._featurize(customer, agent)
            X.append(feat)
            y.append(row['csat'])
            
            # Define treatment: high skill match = 1
            treatment.append(1 if row['skill_match'] > 0.6 else 0)
        
        X = np.array(X)
        y = np.array(y)
        treatment = np.array(treatment)
        
        # Step 1: Train outcome models for treated and control
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        self.mu1_model.fit(X_treated, y_treated)
        self.mu0_model.fit(X_control, y_control)
        
        # Step 2: Impute counterfactuals
        D1 = y_treated - self.mu0_model.predict(X_treated)
        D0 = self.mu1_model.predict(X_control) - y_control
        
        # Step 3: Train CATE models
        self.tau1_model.fit(X_treated, D1)
        self.tau0_model.fit(X_control, D0)
        
        self.is_trained = True
        print(f"✓ Uplift model trained on {len(X)} samples")
    
    def predict_uplift(self, customer: pd.Series, agent: pd.Series, 
                       exploration=False) -> Tuple[float, float]:
        """
        Predict CSAT uplift + uncertainty for (customer, agent) pair.
        
        For batch processing, use predict_uplift_batch() instead.
        
        Returns: (uplift, uncertainty)
        """
        if not self.is_trained:
            # Cold start: use heuristic
            skill_match = agent[f'skill_{customer["skill_needed"]}']
            return skill_match * 0.3, 0.2
        
        X = self._featurize(customer, agent).reshape(1, -1)
        
        # Get predictions from both models (suppress warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau1 = self.tau1_model.predict(X)[0]
            tau0 = self.tau0_model.predict(X)[0]
        
        # Average (propensity weighting would go here in production)
        uplift = 0.5 * (tau1 + tau0)
        
        # Uncertainty estimate (simplified - use bootstrap in production)
        uncertainty = np.abs(tau1 - tau0) / 2
        
        # Thompson sampling for exploration
        if exploration and config.THOMPSON_SAMPLING:
            uplift = np.random.normal(uplift, uncertainty)
        
        return uplift, uncertainty
    
    def predict_uplift_batch(self, 
                            customers: pd.DataFrame,
                            agents: pd.DataFrame,
                            customer_indices: np.ndarray,
                            agent_indices: np.ndarray,
                            exploration: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple (customer, agent) pairs.
        
        This is significantly faster than calling predict_uplift() individually.
        Performs single batch inference instead of K×M sequential calls.
        
        Args:
            customers: DataFrame of all customers
            agents: DataFrame of all agents
            customer_indices: Array of customer indices (length N)
            agent_indices: Array of agent indices (length N)
            exploration: Whether to apply Thompson sampling
        
        Returns:
            Tuple of (uplift_array, uncertainty_array) both shape (N,)
        """
        n_pairs = len(customer_indices)
        
        if not self.is_trained:
            # Cold start: use heuristic for all pairs
            skill_matches = np.array([
                agents.iloc[agent_indices[i]][f'skill_{customers.iloc[customer_indices[i]]["skill_needed"]}']
                for i in range(n_pairs)
            ])
            uplifts = skill_matches * 0.3
            uncertainties = np.full(n_pairs, 0.2)
            
            if exploration and config.THOMPSON_SAMPLING:
                uplifts = np.random.normal(uplifts, uncertainties)
            
            return uplifts, uncertainties
        
        # Batch featurize all pairs
        X_batch = np.zeros((n_pairs, 12))  # 12 features per pair
        
        for i in range(n_pairs):
            customer = customers.iloc[customer_indices[i]]
            agent = agents.iloc[agent_indices[i]]
            X_batch[i] = self._featurize(customer, agent)
        
        # Batch predict from both models (suppress warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau1_batch = self.tau1_model.predict(X_batch)
            tau0_batch = self.tau0_model.predict(X_batch)
        
        # Average (propensity weighting would go here in production)
        uplifts = 0.5 * (tau1_batch + tau0_batch)
        
        # Uncertainty estimate
        uncertainties = np.abs(tau1_batch - tau0_batch) / 2
        
        # Thompson sampling for exploration
        if exploration and config.THOMPSON_SAMPLING:
            uplifts = np.random.normal(uplifts, uncertainties)
        
        return uplifts, uncertainties


class CapacityModel:
    """Model for predicting AHT and capacity constraints"""
    def __init__(self):
        self.aht_model = LGBMRegressor(**config.UPLIFT_MODEL_PARAMS)
        self.is_trained = False
        
        # Capacity check cache for performance optimization
        # Maps (agent_idx, channel) -> (capacity_result, cache_timestamp)
        self._capacity_cache: dict = {}
        self._cache_enabled = config.ENABLE_CAPACITY_CACHE
        self._cache_ttl_seconds = 1.0  # Cache valid for 1 second (assumes rapid state changes)
    
    def train(self, historical_data: pd.DataFrame, agents_df: pd.DataFrame):
        """Train AHT prediction model"""
        print("Training Capacity Model (AHT Predictor)...")
        
        X = []
        y = []
        
        for _, row in historical_data.iterrows():
            agent = agents_df[agents_df['agent_id'] == row['agent_id']].iloc[0]
            
            features = [
                agent['avg_aht'],
                agent['experience_years'],
                row['skill_match'],
                config.CHANNELS.index(row['channel']),
                np.random.beta(2, 5)  # complexity proxy
            ]
            
            X.append(features)
            y.append(row['aht'])
        
        self.aht_model.fit(np.array(X), np.array(y))
        self.is_trained = True
        print("✓ Capacity model trained")
    
    def predict_aht(self, customer: pd.Series, agent: pd.Series) -> float:
        """
        Predict average handle time for single (customer, agent) pair.
        
        For batch processing, use predict_aht_batch() instead.
        """
        if not self.is_trained:
            return agent['avg_aht'] * (1.2 - agent[f'skill_{customer["skill_needed"]}'])
        
        features = np.array([[
            agent['avg_aht'],
            agent['experience_years'],
            agent[f'skill_{customer["skill_needed"]}'],
            config.CHANNELS.index(customer['channel']),
            customer['complexity']
        ]])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.aht_model.predict(features)[0]
    
    def predict_aht_batch(self,
                         customers: pd.DataFrame,
                         agents: pd.DataFrame,
                         customer_indices: np.ndarray,
                         agent_indices: np.ndarray) -> np.ndarray:
        """
        Batch prediction for AHT across multiple (customer, agent) pairs.
        
        Args:
            customers: DataFrame of all customers
            agents: DataFrame of all agents
            customer_indices: Array of customer indices (length N)
            agent_indices: Array of agent indices (length N)
        
        Returns:
            Array of predicted AHT values (shape N,)
        """
        n_pairs = len(customer_indices)
        
        if not self.is_trained:
            # Cold start: use heuristic
            ahts = np.array([
                agents.iloc[agent_indices[i]]['avg_aht'] * 
                (1.2 - agents.iloc[agent_indices[i]][f'skill_{customers.iloc[customer_indices[i]]["skill_needed"]}'])
                for i in range(n_pairs)
            ])
            return ahts
        
        # Batch featurize
        features_batch = np.zeros((n_pairs, 5))
        
        for i in range(n_pairs):
            customer = customers.iloc[customer_indices[i]]
            agent = agents.iloc[agent_indices[i]]
            
            features_batch[i] = [
                agent['avg_aht'],
                agent['experience_years'],
                agent[f'skill_{customer["skill_needed"]}'],
                config.CHANNELS.index(customer['channel']),
                customer['complexity']
            ]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.aht_model.predict(features_batch)
    
    def check_capacity(self, agent: pd.Series, channel: str, agent_idx: Optional[int] = None) -> bool:
        """
        Check if agent can accept another interaction on this channel.
        
        For batch processing, use check_capacity_batch() instead.
        
        Args:
            agent: Agent Series
            channel: Channel string
            agent_idx: Optional agent index for caching (if None, cache is not used)
        """
        # Use cache if enabled and agent_idx provided
        if self._cache_enabled and agent_idx is not None:
            cache_key = (agent_idx, channel)
            import time
            current_time = time.time()
            
            if cache_key in self._capacity_cache:
                cached_result, cache_time = self._capacity_cache[cache_key]
                if current_time - cache_time < self._cache_ttl_seconds:
                    return cached_result
            
            # Compute capacity
            result = self._compute_capacity(agent, channel)
            
            # Store in cache
            self._capacity_cache[cache_key] = (result, current_time)
            return result
        
        return self._compute_capacity(agent, channel)
    
    def _compute_capacity(self, agent: pd.Series, channel: str) -> bool:
        """Internal method to compute capacity (without caching)"""
        current_load = agent.get(f'load_{channel}', 0)
        max_capacity = config.CAPACITY_RULES[channel]
        
        # Cross-channel constraints
        for other_channel in config.CHANNELS:
            if other_channel != channel:
                other_load = agent.get(f'load_{other_channel}', 0)
                if other_load > 0:
                    cross_cap = config.CROSS_CHANNEL_CAPACITY[other_channel][channel]
                    max_capacity = min(max_capacity, cross_cap)
        
        return current_load < max_capacity
    
    def clear_capacity_cache(self):
        """Clear the capacity cache (call when agent states change significantly)"""
        self._capacity_cache.clear()
    
    def check_capacity_batch(self,
                            agents: pd.DataFrame,
                            agent_indices: np.ndarray,
                            channels: np.ndarray) -> np.ndarray:
        """
        Batch capacity check for multiple (agent, channel) pairs.
        
        Args:
            agents: DataFrame of all agents
            agent_indices: Array of agent indices (length N)
            channels: Array of channel strings (length N)
        
        Returns:
            Boolean array (shape N,) indicating capacity availability
        """
        n_pairs = len(agent_indices)
        capacity_mask = np.zeros(n_pairs, dtype=bool)
        
        for i in range(n_pairs):
            agent_idx = agent_indices[i]
            channel = channels[i]
            agent = agents.iloc[agent_idx]
            
            current_load = agent.get(f'load_{channel}', 0)
            max_capacity = config.CAPACITY_RULES[channel]
            
            # Cross-channel constraints
            for other_channel in config.CHANNELS:
                if other_channel != channel:
                    other_load = agent.get(f'load_{other_channel}', 0)
                    if other_load > 0:
                        cross_cap = config.CROSS_CHANNEL_CAPACITY[other_channel][channel]
                        max_capacity = min(max_capacity, cross_cap)
            
            capacity_mask[i] = current_load < max_capacity
        
        return capacity_mask
