from .context import src

from src.core import slice_list

def test_slice_list():
    assert slice_list(range(1, 4), 2) == [[1, 2], [3]]
    assert slice_list(range(1, 5), 2) == [[1, 2], [3, 4]]
    assert slice_list(range(1, 7), 3) == [[1, 2], [3, 4], [5, 6]]
    assert slice_list(range(1, 3), 1) == [[1, 2]]
