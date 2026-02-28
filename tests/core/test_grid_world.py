import pytest
from synergygrid.core import AgentAction, BaseResource, ResourceMeta, GridWorld
import numpy as np

class TestGridWorld:
    @pytest.fixture
    def grid_world(self):
        """
        Fixture to create and initialize a GridWorld instance with 1 active resource.
        Resets the world before each test to ensure a clean state.
        """
        gw = GridWorld(max_active_resources=1, grid_rows=1, grid_cols=2)
        gw.reset()

        return gw
    
    def test_initialization(self, grid_world):
        """
        Test that verifies the initialization of the GridWorld object.
        Checks if the grid's rows and columns are correctly set and that the grid contains the expected number of resources.
        """
        resources = grid_world.get_resource_positions()
        
        assert len(resources) == 1  # There should be exactly one resource after initialization.
        assert grid_world.grid_rows == 1  # The grid should have 1 row.
        assert grid_world.grid_cols == 2  # The grid should have 2 columns.
    
    def test_perform_agent_action(self, grid_world):
        """
        Test that ensures performing an agent action modifies the resource status correctly.
        Initially, the resource is active. After an action, the status of the resource should change.
        """
        is_active = grid_world.get_resource_is_active_status()[0]
        
        # Perform an action and assert that the resource's active status has changed.
        grid_world.perform_agent_action(AgentAction.LEFT)
        
        assert grid_world.get_resource_is_active_status()[0] != is_active
        
    def test_resource_positions(self, grid_world):
        """
        Test to verify that the resource positions are correctly returned.
        Ensures each resource position is a list with exactly two elements (representing x, y coordinates) and that they are of type np.int64.
        """
        positions = grid_world.get_resource_positions()
        
        for pos in positions:
            assert isinstance(pos, list)  # Each position should be a list.
            assert len(pos) == 2  # Each list should contain two elements (x, y).
            assert isinstance(pos[0], np.int64) and isinstance(pos[1], np.int64)  # Each coordinate should be np.int64.

    def test_get_resource_is_active_status(self, grid_world):
        """
        Test that checks if the resource's active status is returned as a boolean.
        Ensures that each resource's status is either True or False.
        """
        statuses = grid_world.get_resource_is_active_status()
        
        for status in statuses:
            assert isinstance(status, bool)  # Each status should be a boolean.
    
    def test_get_resource_types(self, grid_world):
        """
        Test to verify that the resource types returned are valid and are subclasses of ResourceMeta.
        This ensures that the resources in the world are of correct types (e.g., PositiveResource, NegativeResource).
        """
        types = grid_world.get_resource_types()
        
        for t in types:    
            assert issubclass(t.__class__, ResourceMeta)  # The resource type should be a subclass of ResourceMeta.
            
    def test_get_resource_timers(self, grid_world):
        """
        Test to ensure that each resource's timer is correctly returned and is an instance of BaseResource.Timer.
        This confirms that the resources have valid timers.
        """
        timers = grid_world.get_resource_timers()
        
        for timer in timers:
            assert isinstance(timer, BaseResource.Timer)  # Each timer should be an instance of BaseResource.Timer.
    
    def test_get_resource_is_active_after_reset(self, grid_world):
        """
        Test to verify that after resetting the GridWorld, all resources are active.
        This ensures that the reset method initializes all resources correctly.
        """
        statuses = grid_world.get_resource_is_active_status()
        
        for status in statuses:
            assert status is True  # All resources should be active after reset.