from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union


class BaseAgent(ABC):
    """
    智能体抽象基类。

    定义所有智能体必须实现的核心接口契约，提供训练/评估模式切换、
    步数统计等通用基础能力。所有具体算法智能体均需继承本类并实现全部抽象方法。

    Attributes:
        device: 模型计算设备，如 "cpu"、"cuda"。
        is_training: 训练模式标记，True为训练模式，False为评估模式。
        total_steps: 全局训练步数累计计数器。
    """

    def __init__(self, device="cpu"):
        """初始化基类通用属性。
        Args:
            device: 模型计算设备。
        """
        self.device = device
        self.is_training = True
        self.total_steps = 0

    @abstractmethod
    def select_action(self, state):
        """
        根据当前环境状态选择动作。
        Args:
            state: 环境当前状态，具体类型由对应算法定义。
        Returns:
                智能体输出的动作，离散动作通常为整型或张量，
                连续动作通常为浮点型张量。
        """
        pass

    @abstractmethod
    def update(self):
        """
        触发一次模型参数更新。

        智能体内部自行从经验回放池采样训练数据，外部仅调用触发更新。
        Returns:
            训练指标字典，键为指标名称，值为对应浮点数值，
            用于日志记录与训练可视化。
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        获取智能体的可序列化状态字典。
        Returns:
            包含模型权重、超参数、运行状态等所有可持久化信息的字典。
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
         从状态字典加载智能体状态。
        Args:
            state_dict: 状态字典，需与 state_dict() 输出格式一致。
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        保存智能体的模型权重与运行状态到文件。
        Args:
            path: 模型权重文件的保存路径。
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        从文件加载智能体的模型权重与运行状态。
        Args:
            path: 模型权重文件的读取路径。
        """
        pass

    def train(self):
        """
        切换为训练模式，开启探索机制与梯度更新。
        """
        self.is_training = True

    def eval(self):
        """
        切换为评估模式，关闭探索，使用确定性策略输出动作。
        """
        self.is_training = False

    def reset(self):
        """
        重置智能体内部状态。

        通常在每个回合开始前调用，默认空实现。
        含内部状态的子类可重写该方法。
        """
        pass
