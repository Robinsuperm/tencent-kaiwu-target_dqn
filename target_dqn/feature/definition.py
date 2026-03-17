#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.back_to_the_realm.target_dqn.feature_process import (
    one_hot_encoding,
    read_relative_position,
    bump,
)


# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def reward_shaping(
    frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info, bump_cnt, step
):
    reward = 0

    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _env_info.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    # Get the grid-based distance of the current agent's position relative to the end point, buff, and treasure chest
    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    end_dist = _obs.feature.end_pos.grid_distance * 256 #逆归一化
    buff_dist = _obs.feature.buff_pos.grid_distance * 256
    treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos][2:15]

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = env_info.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the grid-based distance of the agent's position from the previous
    # frame relative to the end point, buff, and treasure chest
    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = obs.feature.end_pos.grid_distance * 256
    prev_buff_dist = obs.feature.buff_pos.grid_distance * 256
    prev_treasure_dists = [pos.grid_distance for pos in obs.feature.treasure_pos][2:15]

    # Get the status of the buff
    # 获取buff的状态
    buff_availability = 0
    for organ in env_info.frame_state.organs:
        if organ.sub_type == 2:
            buff_availability = organ.status

    # Get the acceleration status of the agent
    # 获取智能体的加速状态
    prev_speed_up = env_info.frame_state.heroes[0].speed_up
    speed_up = _env_info.frame_state.heroes[0].speed_up

    ### 判断宝箱全收集
    all_collect = all(x == 1.0 for x in treasure_dists)

    # ### 全收集奖励
    collect_reward = 0
    # if all_collect and treasure_dists.count(1.0) > prev_treasure_dists.count(1.0):
    #     collect_reward += 5

    # ### 得分奖励
    # score_reward

    ### 撞墙判断
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励

    # Boundary handling: At the first frame, prev_end_dist is initialized to 1,
    # and no reward is calculated at this time
    # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
    if prev_end_dist != 1:
        if all_collect and not is_bump and not terminated:
            reward_end_dist_count = 1 * (prev_end_dist - end_dist) / 300
            if reward_end_dist_count > 1:
                reward_end_dist += 1
            elif reward_end_dist_count < -1:
                reward_end_dist -= 1
            else :
                reward_end_dist += reward_end_dist_count

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        if all_collect:
            reward_win += 10
            reward_win += (2000 - step) / 500
            bump_win_reward = (50 - bump_cnt) * 0.1
            if bump_win_reward < -2:
                reward_win -= 2
            else :
                reward_win += bump_win_reward
        else :
            reward_win += 0.2
            count_left = [x for x in treasure_dists if x != 1.0]
            reward_win_cnt = 1 * len(count_left)
            reward_win -= reward_win_cnt / 13
            count_have = [x for x in treasure_dists if x == 1.0]
            reward_win_cnt = 1 * len(count_have)
            reward_win += reward_win_cnt / 13
    elif truncated:
        reward_win -= 10

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)

    ### 靠近最近的宝箱的奖励
    if min(treasure_dists) < min(prev_treasure_dists):
        if not is_bump and prev_treasure_dists.count(1.0) == treasure_dists.count(1.0):
            reward_treasure_dist_count = 0.5 * (min(prev_treasure_dists) - min(treasure_dists)) / 300
            if reward_treasure_dist_count > 3:
                reward_treasure_dist += 3
            elif reward_treasure_dist_count < -3:
                reward_treasure_dist -= 3
            else:
                reward_treasure_dist += reward_treasure_dist_count
        # else :
        #     reward_treasure_dist -= 0.001   #可能导致agent舍弃宝箱？

    ### 对全部宝箱距离同时敏感
    # dists_change = [curr - prev for curr, prev in zip(treasure_dists, prev_treasure_dists)]
    # if treasure_dists.count(1.0) == prev_treasure_dists.count(1.0):  # 防止收集宝箱后反而惩罚
    #     reward_treasure_dist += sum([100 if change < 0 else -100 for change in dists_change])   #对每个减少的差值reward
    #     # reward_treasure_dists += -sum(dists_change) * 50 # 对所有宝箱距离变化总差值减小reward，感觉没有上面的好用
    
    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    if prev_treasure_dists.count(1.0) < treasure_dists.count(1.0):
        reward_treasure += 5
        if not is_bump:
            reward_treasure += 1

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = 0
    if buff_dist < prev_buff_dist and not is_bump and step < 700:
        reward_buff_dist_count = 0.5 * (prev_buff_dist - buff_dist) / 700
        if reward_buff_dist_count > 0.5:
            reward_buff_dist += 0.5
        elif reward_buff_dist_count < -0.5:
            reward_buff_dist -= -0.5
        else :
            reward_buff_dist += reward_buff_dist_count

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励
    reward_buff = 0
    if buff_dist == 1:
        reward_buff += 2
        if not is_bump:
            reward_buff += 1

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # Reward 4.1 Penalty for flickering into the wall (TODO)
    # 奖励4.1 撞墙闪现的惩罚 (TODO)

    x_diff = curr_pos_x - prev_pos_x
    z_diff = curr_pos_z - prev_pos_z

    legal_act = [fli for fli in obs.legal_act]
    _legal_act = [_fli for _fli in _obs.legal_act]

    ### 用顺序作方向
    zero_list = [25 for a in range(0, 51)]
    column = [b for b in range(0, 51)][::-1]
    row = [a for a in range(0, 51)]
    if x_diff == 0:
        row = zero_list
    elif x_diff > 0:
        row = row[25:51]
    else:
        row = row[0:25]
    if z_diff ==0:
        column = zero_list
    elif z_diff > 0:
        column = column[25:51]
    else:
        column = column[0:25]

    if env_info:
        if legal_act[1] == 1 and _legal_act[1] == 0 and (x_diff ** 2 + z_diff ** 2) ** 0.5 > 1000 and (x_diff ** 2 + z_diff ** 2) ** 0.5 < 8000: # 使用闪现并且闪现距离小于八千
            reward_flicker -= 3

    # Reward 4.2 Reward for normal flickering (TODO)
    # 奖励4.2 正常闪现的奖励 (TODO)
        if legal_act[1] == 1 and _legal_act[1] == 0 and (x_diff ** 2 + z_diff ** 2) ** 0.5 == 8000:
            reward_flicker += 1

    # Reward 4.3 Reward for super flickering (TODO)
    # 奖励4.3 超级闪现的奖励 (TODO)
        obstacle_map = _obs.feature.obstacle_map
        if legal_act[1] == 1 and _legal_act[1] == 0 and (x_diff ** 2 + z_diff ** 2) ** 0.5 == 8000 and not is_bump: # 鼓励穿墙闪现。
            combines = zip(row, column)
            for x, y in combines:
                if obstacle_map[y * 51 + x + 1] == 1:
                    reward_flicker += 1
                    break
        if legal_act[1] == 1 and _legal_act[1] == 0 and (x_diff ** 2 + z_diff ** 2) ** 0.5 > 8000: # 闪现机制怪加分
            reward_flicker += 2

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 0
    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # (TODO: Give penalty after collecting all the treasure chests, encourage full collection)
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)
    if all_collect:
        if prev_end_dist <= end_dist:
            reward_step -= 1
        else:
            reward_step += 1 * (prev_end_dist - end_dist) / 700 if not is_bump else 0

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0
    memory_map = obs.feature.memory_map
    # reward_memory = memory_map[len(memory_map)//2] +(memory_map[len(memory_map)//2 + 1]  +memory_map[len(memory_map)//2 - 1]+memory_map[len(memory_map)//2 + 51] +memory_map[len(memory_map)//2 -51]) * 0.25
    reward_memory -= memory_map[25 * 51 + 25] * 0.1
    if memory_map[25 * 51 + 25] == 0 and not is_bump:
        reward_memory += 0.05

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0

    # Determine whether it bumps into the wall
    # 判断是否撞墙
    if is_bump:
        reward_bump -= 1
    else :
        reward_bump += 1

    #懒惰惩罚
    reward_laziness = 0
    if (x_diff ** 2 + z_diff ** 2) ** 0.5 < 200:
        reward_laziness -= 0.1

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    REWARD_CONFIG = {
        "reward_end_dist": "1",
        "reward_win": "1",
        "reward_buff_dist": "1",
        "reward_buff": "1",
        "reward_treasure_dists": "1",
        "reward_treasure": "1",
        "reward_flicker": "1",
        "reward_step": "1",
        "reward_bump": "1",
        "reward_memory": "1",
        "reward_laziness": "1",
        "collect_reward": "1",
    }

    reward = [
        reward_end_dist * float(REWARD_CONFIG["reward_end_dist"]),
        reward_win * float(REWARD_CONFIG["reward_win"]),
        reward_buff * float(REWARD_CONFIG["reward_buff"]),
        reward_treasure_dist * float(REWARD_CONFIG["reward_treasure_dists"]),
        reward_treasure * float(REWARD_CONFIG["reward_treasure"]),
        reward_flicker * float(REWARD_CONFIG["reward_flicker"]),
        reward_step * float(REWARD_CONFIG["reward_step"]),
        reward_bump * float(REWARD_CONFIG["reward_bump"]),
        reward_memory * float(REWARD_CONFIG["reward_memory"]),
        reward_laziness * float(REWARD_CONFIG["reward_laziness"]),
        collect_reward * float(REWARD_CONFIG["collect_reward"]),
        reward_buff_dist * float(REWARD_CONFIG["reward_buff_dist"]),
    ]

    return sum(reward), is_bump, reward


@attached
def observation_process(raw_obs, env_info=None):
    """
    This function is an important feature processing function, mainly responsible for:
        - Parsing information in the raw data
        - Parsing preprocessed feature data
        - Processing the features and returning the processed feature vector
        - Concatenation of features
        - Annotation of legal actions
    Function inputs:
        - raw_obs: Preprocessed feature data
        - env_info: Environment information returned by the game
    Function outputs:
        - observation: Feature vector
        - legal_action: Annotation of legal actions

    该函数是特征处理的重要函数, 主要负责：
        - 解析原始数据里的信息
        - 解析预处理后的特征数据
        - 对特征进行处理, 并返回处理后的特征向量
        - 特征的拼接
        - 合法动作的标注
    函数的输入：
        - raw_obs: 预处理后的特征数据
        - env_info: 游戏返回的环境信息
    函数的输出：
        - observation: 特征向量
        - legal_action: 合法动作的标注
    """
    feature, legal_act = [], []

    # Unpack the preprocessed feature data according to the protocol
    # 对预处理后的特征数据按照协议进行解包
    norm_pos = raw_obs.feature.norm_pos
    grid_pos = raw_obs.feature.grid_pos
    start_pos = raw_obs.feature.start_pos
    end_pos = raw_obs.feature.end_pos
    buff_pos = raw_obs.feature.buff_pos
    treasure_poss = raw_obs.feature.treasure_pos
    obstacle_map = list(raw_obs.feature.obstacle_map)
    memory_map = list(raw_obs.feature.memory_map)
    treasure_map = list(raw_obs.feature.treasure_map)
    end_map = list(raw_obs.feature.end_map)

    # Feature processing 1: One-hot encoding of the current position
    # 特征处理1：当前位置的one-hot编码
    one_hot_pos = one_hot_encoding(grid_pos)

    # Feature processing 2: Normalized position
    # 特征处理2：归一化位置
    norm_pos = [norm_pos.x, norm_pos.z]

    # Feature processing 3: Information about the current position relative to the end point
    # 特征处理3：当前位置相对终点点位的信息
    ens = read_relative_position(end_pos)

    # Feature processing 4: Information about the current position relative to the treasure position
    # 特征处理4: 当前位置相对宝箱位置的信息
    treasure_poss_features = []
    for treasure_pos in treasure_poss:
        treasure_poss_features = treasure_poss_features + list(
            read_relative_position(treasure_pos)
        )

    # Feature processing 5: Whether the buff is collectable
    # 特征处理5：buff是否可收集
    buff_availability = 0
    if env_info:
        for organ in env_info.frame_state.organs:
            if organ.sub_type == 2:
                buff_availability = organ.status

    # Feature processing 6: Whether the flash skill can be used
    # 特征处理6：闪现技能是否可使用
    talent_availability = 0
    if env_info:
        talent_availability = env_info.frame_state.heroes[0].talent.status


    # Feature concatenation:
    # Concatenate all necessary features as vector features (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    # 特征拼接：将所有需要的特征进行拼接作为向量特征 (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    feature = (
        norm_pos
        + one_hot_pos
        + end_pos_features
        + treasure_poss_features
        + [buff_availability, talent_availability]
        + obstacle_map
        + end_map
        + treasure_map
        + memory_map
    )

    # Legal actions
    # 合法动作
     
    return ObsData(feature=feature, legal_act=legal_act)


@attached
def action_process(act_data):
    result = act_data.move_dir
    result += act_data.use_talent * 8
    return result


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:10808],
        _obs=s_data[10808:21616],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
