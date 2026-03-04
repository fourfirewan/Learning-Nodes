import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. 超参数设置 ======================
# 环境：CartPole-v1
env = gym.make("CartPole-v1")

# 离散化参数：把每个连续维度切分成多少份
# 维度说明：[小车位置, 小车速度, 杆子角度, 杆子角速度]
DISCRETE_BINS = [20, 20, 20, 20]  # 每个维度切 20 份，可调整

# Q-Learning 核心超参数
ALPHA_INIT = 0.1       # 学习率：0-1，越大更新越快，但可能不稳定
ALPHA_MIN = 0.05
ALPHA_DECAY = 0.99995

GAMMA = 0.99      # 折扣因子：0-1，越大越看重未来奖励
EPSILON = 1.0     # 初始探索率：1.0 表示完全随机探索
EPSILON_MIN = 0.001# 最小探索率：保证后期仍有少量探索
EPSILON_DECAY = 0.999  # 探索率衰减：每回合乘以 0.995

# 训练参数
EPISODES = 20000   # 训练回合数
SHOW_EVERY = 1000  # 每 1000 回合测试并可视化一次

# ====================== 2. 状态离散化函数（核心） ======================
def get_discrete_state(state):
    """
    把连续的 4 维状态映射成离散的索引
    state: [小车位置, 小车速度, 杆子角度, 杆子角速度]
    """
    # 1. 先定义每个维度的合理范围（截断无限值）
    # CartPole-v1 官方观测范围：
    # 小车位置: [-2.4, 2.4]，小车速度: [-inf, inf] → 截断到 [-3, 3]
    # 杆子角度: [-0.2095, 0.2095] rad（±12°），杆子角速度: [-inf, inf] → 截断到 [-3, 3]
    state_bounds = [
        [-2.4, 2.4],    # 小车位置
        [-4, 4],         # 小车速度（截断）
        [-0.2095, 0.2095],  # 杆子角度
        [-8, 8]          # 杆子角速度（截断）
    ]
    
    discrete_state = []
    for i in range(len(state)):
        # 2. 把当前维度的值限制在合理范围内
        value = np.clip(state[i], state_bounds[i][0], state_bounds[i][1])
        # 3. 计算该值在区间中的位置（归一化到 0-1）
        normalized = (value - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        # 4. 映射到离散索引（0 到 DISCRETE_BINS[i]-1）
        discrete_index = int(normalized * (DISCRETE_BINS[i] - 1))
        # 防止索引越界（极端情况,防御性编程）
        discrete_index = np.clip(discrete_index, 0, DISCRETE_BINS[i]-1)
        discrete_state.append(discrete_index)
    
    return tuple(discrete_state)  # 转成 tuple 才能作为 Q 表的索引

# ====================== 3. 初始化 Q 表 ======================
# Q 表形状：(dim1_bins, dim2_bins, dim3_bins, dim4_bins, 动作数)
# 动作数是 2：0=向左，1=向右
q_table = np.random.uniform(
    low=-1, high=1,  # 初始 Q 值设为 -1 到 1 之间的随机数
    size=(DISCRETE_BINS + [env.action_space.n])
)

# ====================== 4. 训练循环 ======================
reward_list = []  # 记录每个回合的奖励，用于画曲线

for episode in range(EPISODES):
    # 重置环境，获取初始连续状态
    continuous_state, _ = env.reset(seed=42)
    # 转成离散状态
    discrete_state = get_discrete_state(continuous_state)
    
    done = False
    total_reward = 0  # 记录当前回合的总奖励
    # 动态衰减学习率
    alpha = max(ALPHA_MIN, ALPHA_INIT * (ALPHA_DECAY ** episode))
    
    # 回合内循环：直到杆子倒下或达到最大步数
    while not done:
        # ---------------------- ε-贪婪策略选动作 ----------------------
        if np.random.random() < EPSILON:
            # 探索：随机选动作
            action = env.action_space.sample()
        else:
            # 利用：选 Q 值最大的动作
            action = np.argmax(q_table[discrete_state])
        
        # ---------------------- 执行动作 ----------------------
        next_continuous_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 杆子倒了或超时都算结束
        total_reward += reward
        
        # ---------------------- 离散化下一个状态 ----------------------
        next_discrete_state = get_discrete_state(next_continuous_state)
        
        # ---------------------- 更新 Q 表（核心！） ----------------------
        current_q = q_table[discrete_state + (action,)]

        # ========== 关键：三种情况分别处理 ==========
        if terminated:
            # ★ 真正失败：给惩罚，不bootstrap（未来奖励为0）
            target = -200  # 惩罚信号，让智能体学会"避免倒杆"

        elif truncated:
            # ★ 活满500步被截断：继续bootstrap（未来奖励不为0）
            # 智能体还"活着"，用Q(s')估计它本来还能拿多少分
            max_future_q = np.max(q_table[next_discrete_state])
            target = reward + GAMMA * max_future_q

        else:
            # ★ 正常步骤：标准Q-learning更新
            # 更新逻辑和truncated一致，可合并，但列出来更清晰
            max_future_q = np.max(q_table[next_discrete_state])
            target = reward + GAMMA * max_future_q

        # 统一更新公式
        q_table[discrete_state + (action,)] = (1 - alpha) * current_q + alpha * target
        discrete_state = next_discrete_state
    
    # ---------------------- 回合结束后的处理 ----------------------
    # 衰减探索率
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    
    # 记录奖励
    reward_list.append(total_reward)
    
    # 每 SHOW_EVERY 回合打印一次进度，并测试可视化
    if episode % SHOW_EVERY == 0:
        avg_reward = np.mean(reward_list[-SHOW_EVERY:])
        print(f"Episode: {episode:4d} | 最近 {SHOW_EVERY} 回合平均奖励: {avg_reward:6.2f} | 探索率: {EPSILON:.4f} | 学习率: {alpha:.4f}")
        
        # 测试并可视化（单独开一个环境）
        test_env = gym.make("CartPole-v1", render_mode="human")
        test_state, _ = test_env.reset(seed=42)
        test_done = False
        test_reward = 0
        while not test_done:
            test_discrete_state = get_discrete_state(test_state)
            test_action = np.argmax(q_table[test_discrete_state])  # 完全利用，不探索
            test_state, r, test_terminated, test_truncated, _ = test_env.step(test_action)
            test_done = test_terminated or test_truncated
            test_reward += r
        print(f"测试回合奖励: {test_reward} (500=满分）")
        test_env.close()

# ====================== 5. 画奖励曲线 ======================
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning CartPole Training Curve")
plt.axhline(y=500, color='r', linestyle='--', label='perfect:500')
plt.show()

env.close()