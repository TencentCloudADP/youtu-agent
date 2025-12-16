"""Test DBService with database availability checks."""

from unittest.mock import patch

from utu.db import DBService, TrajectoryModel


def test_db_service_when_db_unavailable():
    """Test that DBService gracefully handles unavailable database."""
    with patch("utu.utils.SQLModelUtils.check_db_available", return_value=False):
        # Test add
        result = DBService.add(TrajectoryModel(trace_id="test"))
        assert result is False

        # Test query
        result = DBService.query(TrajectoryModel)
        assert result is None

        # Test get_by_id
        result = DBService.get_by_id(TrajectoryModel, 1)
        assert result is None


def test_db_service_add_empty_list():
    """Test that empty list returns False."""
    with patch("utu.utils.SQLModelUtils.check_db_available", return_value=True):
        result = DBService.add([])
        assert result is False
