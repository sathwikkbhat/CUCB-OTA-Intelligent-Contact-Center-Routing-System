import os

from dataclasses import dataclass

from typing import List


@dataclass
class Config:
    # Data Generation
    NUM_AGENTS = 30
    NUM_CUSTOMERS_PER_BATCH = 50
    SIMULATION_MINUTES = 120
    BATCH_WINDOW_SECONDS = 2.0
    
    # Channels
    CHANNELS = ['voice', 'chat', 'email']
    
    # Skills
    SKILLS = ['technical', 'billing', 'sales', 'general', 'vip']
    SKILL_LEVELS = ['junior', 'mid', 'senior', 'expert']
    
    # Agent Capacity Rules (max concurrent interactions per channel)
    CAPACITY_RULES = {
        'voice': 1,  # Can only handle 1 call at a time
        'chat': 3,   # Can handle up to 3 chats
        'email': 5   # Can handle up to 5 emails
    }
    
    # Multi-channel capacity (if agent on voice, can they handle chat?)
    CROSS_CHANNEL_CAPACITY = {
        'voice': {'chat': 0, 'email': 0},  # Voice blocks all
        'chat': {'voice': 0, 'chat': 3, 'email': 2},  # Chat allows some email
        'email': {'voice': 0, 'chat': 2, 'email': 5}   # Email allows chat
    }
    
    # Model Parameters
    UPLIFT_MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    # Constraint Budgets
    MAX_AHT_MINUTES = 8.0  # Maximum average handle time
    MAX_SLA_VIOLATION_RATE = 0.15  # 15% max SLA violations
    FAIRNESS_GINI_THRESHOLD = 0.3  # Max Gini coefficient for fairness
    
    # SLA Thresholds (in minutes) per channel
    SLA_THRESHOLDS = {
        'voice': 5,   # 5 minutes for voice calls
        'chat': 3,    # 3 minutes for chat
        'email': 60   # 60 minutes for email
    }
    
    # Fairness calculation constants
    FAIRNESS_NORMALIZATION_FACTOR = 0.5  # Used in fairness penalty calculation
    
    # Lagrangian Dual Learning
    LAMBDA_INIT = 0.1
    LAMBDA_LR = 0.01
    LAMBDA_MAX = 5.0
    
    # Exploration
    EPSILON = 0.1  # Epsilon-greedy exploration rate
    THOMPSON_SAMPLING = True
    
    # Paths
    DATA_DIR = 'data'
    LOGS_DIR = 'data/logs'
    MODELS_DIR = 'models/saved'
    
    # Logging
    LOG_INTERVAL = 10  # Log metrics every N batches
    
    # Performance optimization flags
    USE_BATCH_PREDICTIONS = True  # Enable batch model predictions (20-40x speedup)
    ENABLE_CAPACITY_CACHE = True  # Enable capacity check caching (2-3x speedup)
    ENABLE_PERFORMANCE_PROFILING = False  # Enable performance profiling hooks
    
    def __post_init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)


config = Config()

