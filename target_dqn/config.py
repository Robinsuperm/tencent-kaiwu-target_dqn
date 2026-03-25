class Config:
    
    LOAD_MODEL_ID = 362659              # 加载模型ID
    SAMPLE_DIM = 21624
    # ==================== 模型相关 ====================
    DIM_OF_OBSERVATION = 4096 + 404
    DIM_OF_ACTION_DIRECTION = 8
    DIM_OF_TALENT = 8

    DESC_OBS_SPLIT = [404, (4, 51, 51)]  # sum = 10808
    
    # ==================== DQN 核心超参数 ====================
    TARGET_UPDATE_FREQ = 1000                # Target Network 更新频率
    EPSILON_GREEDY_PROBABILITY = 300000      # ε-greedy 衰减步数
    GAMMA = 0.9                              # 折扣因子
    EPSILON = 0.9                            # 初始探索率
    START_LR = 1e-3                          # 初始学习率

    # ==================== 其他配置 ====================
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 
    2
