from abc import ABC, abstractmethod
import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self, 
    ) -> torch.Tensor: ...

