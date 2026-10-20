import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioTradingEnv(gym.Env):
    """
    多资产投资组合交易环境（满足A股主要交易约束）
    
    状态空间（Observation Space）：
        - 每只股票的三个技术指标：20日动量、20日波动率、当前价/20日均价 - 1
        - 当前每只股票的持仓权重（共N个）
        - 当前现金权重（1个）
        总维度 = 3N + N + 1 = 4N + 1
    
    动作空间（Action Space）：
        - 一个长度为 N+1 的连续向量，前N个元素代表每只股票的目标权重（0~1），
          最后一个元素代表目标现金权重（0~1）。所有元素之和会被归一化为1。
    
    交易约束：
        - T+1：今日买入的股票明日才能卖出（用集合 bought_today 记录）
        - 不可卖空：权重非负
        - 交易成本：佣金（万三）、印花税（卖出千一）、滑点（千一）
        - 组合净值初始为1
    
    奖励（Reward）：
        - 使用对数收益 log(1 + 组合当日收益率 - 交易成本率)
        - 对数收益更平稳，适合强化学习训练
    """
    
    def __init__(self, prices_df, window=20, commission=0.0003, 
                 stamp_tax=0.001, slippage=0.001):
        """
        参数：
            prices_df: DataFrame，形状 (T, N)，列是股票代码，行是交易日，值为后复权收盘价
            window: 技术指标回看窗口（默认20日）
            commission: 佣金费率
            stamp_tax: 印花税（仅卖出）
            slippage: 滑点
        """
        super().__init__()   # 必须调用父类初始化
        
        # 将价格数据转为 numpy 数组，方便索引
        self.prices = prices_df.values  # shape (T, N)
        self.n_stocks = prices_df.shape[1]   # 股票数量 N
        self.window = window                 # 技术指标窗口
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        
        # ---------- 定义动作空间 ----------
        # 动作向量长度为 N+1，每个元素取值范围[0,1]，代表目标权重
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.n_stocks + 1,), 
            dtype=np.float32
        )
        
        # ---------- 定义状态空间 ----------
        # 维度 = 3N + N + 1
        obs_dim = 3 * self.n_stocks + self.n_stocks + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 初始化环境
        self.reset()
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态，返回初始观测"""
        super().reset(seed=seed)   # 设置随机种子（可选）
        
        # 当前时间步：从 window 开始，确保有足够历史数据计算指标
        self.t = self.window
        
        # 结束标志
        self.done = False
        
        # 组合状态初始化：全现金
        self.weights = np.zeros(self.n_stocks, dtype=np.float32)  # 每只股票权重
        self.cash_weight = 1.0                                     # 现金权重
        self.portfolio_value = 1.0                                 # 组合净值
        
        # T+1 约束：记录今日买入的股票（今日不可卖）
        self.bought_today = set()
        self.recent_returns = []  # 用于计算滚动夏普比率：保存最近 20 天的组合收益
        
        # 保存交易前的权重，用于计算成本
        self.weights_before_trade = self.weights.copy()
        
        # 返回初始观测和额外信息（gym要求）
        return self._get_obs(), {}

    def _get_obs(self):
        """
        构造当前状态向量。
        
        【核心原则】只使用第 t-1 天及之前的数据，绝不包含第 t 天的收盘价。
        原因：智能体在 t 时刻"做决策"，决策依据必须是 t 时刻已知的信息。
        第 t 天的收盘价要等到 t 日收盘后才产生，决策时不可能知道。
        
        返回：一维 float32 数组，长度为 4N+1
        """
        obs_list = []
        
        # ---------- 对每只股票计算三个技术指标 ----------
        for i in range(self.n_stocks):
            # 取 [t-window, t) 区间的价格，即 t-window 到 t-1 共 window 个交易日
            # 注意切片右端 self.t 是开区间，所以不包含第 t 天，这是正确的
            price_slice = self.prices[self.t - self.window : self.t, i]
            
            # 指标1：动量 = 窗口最后一天价格 / 窗口第一天价格 - 1
            # 这里的"最后一天"是 t-1，不是 t，符合"决策时可得"的要求
            momentum = price_slice[-1] / price_slice[0] - 1
            
            # 指标2：波动率 = 窗口内日收益率的标准差
            # daily_ret 计算相邻两天的收益率，全部基于 t-1 及之前的数据
            daily_ret = np.diff(price_slice) / price_slice[:-1]
            vol = np.std(daily_ret)
            
            # 指标3：价格偏离度 = (t-1 收盘价 / 窗口均价) - 1
            # 【本次修正点】原来用的是 self.prices[self.t, i]（第 t 天价格），
            # 那是未来信息，会直接泄露当日涨跌。改为 self.prices[self.t-1, i]。
            price_dev = self.prices[self.t - 1, i] / price_slice.mean() - 1
            
            # 把这只股票的三个指标加入状态列表
            obs_list.extend([momentum, vol, price_dev])
        
        # ---------- 加入当前持仓权重（N 个） ----------
        # 智能体需要知道自己的仓位，才能决定下一步加仓还是减仓
        obs_list.extend(self.weights)
        
        # ---------- 加入现金权重（1 个） ----------
        obs_list.append(self.cash_weight)
        
        # 转成 float32 数组返回（PPO 网络要求的类型）
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        """
        执行一个动作，推进环境一个时间步。
        
        【核心原则】决策与收益的时序必须正确：
        1. 在 t 时刻，智能体基于 _get_obs() 给出的状态（只含 t-1 及之前的信息）选择动作
        2. 动作决定 t-1 到 t 之间的持仓
        3. 收益 = 持仓权重 × (t 日收益率)，即赚的是 t 日这一天的涨跌
        4. 然后时间推进到 t+1，进入下一轮决策
        
        【本次修正点】原代码先用 self.t 算收益、再 self.t += 1，
        导致"用 t 天的价格算 t 天收益"，但状态里又包含 t 天价格，
        相当于当天收盘前就知道当天涨跌，存在未来函数。
        修正为：先推进 self.t，再用新的 self.t 算收益。
        
        参数：
            action: 长度为 N+1 的数组，前 N 个是股票目标权重，最后一个是现金目标权重
        返回：
            obs, reward, done, truncated, info
        """
        # ---------- 0. 记录交易前的权重（用于计算交易成本和 T+1 判断） ----------
        self.weights_before_trade = self.weights.copy()
        
        # ---------- 1. 归一化动作，保证总权重=1 ----------
        # 把动作值限制在 [0,1]，防止网络输出非法值
        action = np.clip(action, 0.0, 1.0)
        total = np.sum(action)
        
        if total > 0:
            # 前 N 个元素归一化为股票权重，最后一个归一化为现金权重
            target_weights = action[:self.n_stocks] / total
            target_cash = action[self.n_stocks] / total
        else:
            # 极端情况：全为 0，则全部持有现金
            target_weights = np.zeros(self.n_stocks, dtype=np.float32)
            target_cash = 1.0
        
        # ---------- 2. 应用 T+1 约束 ----------
        # 遍历每只股票：如果目标权重低于当前权重（想卖出），
        # 但这只股票是今日买入的（在 bought_today 集合里），则不允许卖，
        # 强制把目标权重拉回当前权重。
        for i in range(self.n_stocks):
            if target_weights[i] < self.weights[i] and i in self.bought_today:
                target_weights[i] = self.weights[i]
        
        # ---------- 3. 计算本次调仓的交易成本率 ----------
        trade_cost = self._calculate_cost(target_weights)
        
        # ---------- 4. 更新 bought_today（供下一时间步的 T+1 判断使用） ----------
        # 先清空，再记录本次买入的股票（目标权重 > 当前权重）
        self.bought_today = set()
        for i in range(self.n_stocks):
            if target_weights[i] > self.weights[i]:
                self.bought_today.add(i)
        
        # ---------- 5. 正式更新持仓权重 ----------
        self.weights = target_weights
        self.cash_weight = target_cash
        
        # ---------- 6. 【修正点】先推进时间步 ----------
        # 决策发生在 t-1 时刻，持仓作用于 t-1 到 t 的区间，
        # 所以要先把 t 推进到新的一天，再用新的 t 计算这一天的收益。
        self.t += 1
        
        # 判断是否到达数据末尾
        if self.t >= len(self.prices) - 1:
            self.done = True
        
        # ---------- 7. 用新的 t 计算当日组合收益 ----------
        if self.t >= len(self.prices):
            # 越界保护：已经超出数据范围，收益记为 0
            portfolio_return = 0.0
        else:
            # 计算每只股票从 t-1 到 t 的收益率
            returns = np.zeros(self.n_stocks)
            for i in range(self.n_stocks):
                if self.prices[self.t - 1, i] != 0:
                    returns[i] = self.prices[self.t, i] / self.prices[self.t - 1, i] - 1
            
            # 组合收益 = 权重加权收益 - 交易成本率
            portfolio_return = np.sum(self.weights * returns) - trade_cost
        
        # ---------- 8. 更新组合净值 ----------
        self.portfolio_value *= (1 + portfolio_return)

        # ---------- 9. 计算奖励（对数收益） ----------
        # 最终版本采用对数收益作为奖励：数值平稳，涨跌对称，适合 PPO 训练
        # 曾尝试滚动夏普奖励控制回撤，但模型过度保守，收益接近零，故回退
        reward = np.log(1 + portfolio_return)
        
        # ---------- 10. 返回新的观测、奖励、结束标志 ----------
        return self._get_obs(), reward, self.done, False, {}
    
    def _calculate_cost(self, target_weights):
        """
        计算从当前权重调整到目标权重所需的交易成本率（占总资产比例）
        成本 = |Δ权重| * (佣金 + 滑点) + 卖出部分额外印花税
        """
        cost = 0.0
        for i in range(self.n_stocks):
            # 权重变化量
            delta = target_weights[i] - self.weights[i]
            # 交易量（绝对值）
            trade_amount = abs(delta)
            
            # 佣金和滑点（双边都收）
            cost += trade_amount * (self.commission + self.slippage)
            
            # 如果卖出（目标权重小于当前权重），额外收印花税
            if delta < 0:
                cost += trade_amount * self.stamp_tax
        
        return cost