from numpy.random import Generator, default_rng


class BaseResource:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, grid_rows=5, grid_cols=5):
        """Defines the game world so resources know their bounds"""

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def reset(self, rng: Generator|None=None) -> None:
        """Initializes the resource in the grid"""

        if rng == None:
            rng = default_rng()

        self.resource_pos = [
            rng.integers(1, self.grid_rows),
            rng.integers(1, self.grid_cols),
        ]
