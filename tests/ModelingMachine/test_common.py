import collections
import pytest

from common import load_class


def test_load_class():
    """Test load class works correctly and raises right exceptions. """
    full_classname = 'collections.namedtuple'
    cls_ = load_class(full_classname)
    assert cls_ is collections.namedtuple

    with pytest.raises(ValueError):
        full_classname = 'collections.Foobar'
        load_class(full_classname)

    with pytest.raises(ImportError):
        full_classname = 'barfoo.Foobar'
        load_class(full_classname)
