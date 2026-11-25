import pytest
from minisgl.env import EnvMem


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1024B", 1024), (" 0b ", 0), ("1024", 1024), ("1KB", 1024), ("1.5M", 1572864)],
)
def test_pynccl_buffer_size_in_bytes(monkeypatch, value, expected):
    name = "MINISGL_PYNCCL_MAX_BUFFER_SIZE"
    monkeypatch.setenv(name, value)
    buffer_size = EnvMem(1024**3)
    buffer_size._init(name)

    assert buffer_size.value == expected
