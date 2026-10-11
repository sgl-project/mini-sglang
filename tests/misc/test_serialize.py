from __future__ import annotations
from dataclasses import dataclass
from typing import List

from minisgl.core import SamplingParams
import torch
from minisgl.message import BatchBackendMsg, UserMsg
from minisgl.message.utils import serialize_type, deserialize_type
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


@call_if_main(__name__)
def test_serialize_deserialize():

    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    logger.info(data)
    y = deserialize_type({"A": A}, data)
    logger.info(y)
    assert (y.x, y.y) == (10, "hello")
    assert torch.equal(y.w, t) and y.w.dtype == t.dtype
    assert len(y.z) == 1 and (y.z[0].x, y.z[0].y, y.z[0].z) == (20, "world", [])
    assert torch.equal(y.z[0].w, t)

    u = BatchBackendMsg([UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())])
    result = u.decoder(u.encoder())
    logger.info(u)
    logger.info(result)
    assert isinstance(result, BatchBackendMsg) and len(result.data) == 1
    msg = result.data[0]
    assert isinstance(msg, UserMsg) and msg.uid == 0
    assert torch.equal(msg.input_ids, t)
    assert msg.sampling_params == SamplingParams()
