from synergygrid.rendering.pygame_renderer import PygameRenderer
from synergygrid.core.grid_world import GridWorld


class HumanRunner:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, steps_left: int = 100):
        # TODO: these values will be switched to use the same as the ones for SYNGridEnv through
        # the config file so its more coherent testing the agent setup you attempt to train
        self._renderer = PygameRenderer(5, 5, 60)
        self._world = GridWorld(3, 5, 5)
        self._steps_left = steps_left

    # ================= #
    #        API        #
    # ================= #

    def human_player_loop(self) -> None:
        self._world.reset()
        self._render()
        action = None

        while True:
            if action is not None:
                self._world.perform_agent_action(action)
                self._steps_left -= 1
                truncated = self._steps_left <= 0
                terminated = self._world.agent.score <= 0
                self._render()

                if terminated or truncated:
                    break

            action = self._renderer.get_user_action()

    # ================= #
    #      Helpers      #
    # ================= #

    def _render(self):
        self._renderer.render(
            self._world.agent.position,
            self._world.get_resource_is_active_status(True),
            self._world.get_resource_positions(True),
            self._world.get_resource_meta(True),
            self._get_hud_data(),
        )

    def _get_hud_data(self) -> dict[str, int]:
        hud_data: dict[str, int] = {}
        hud_data["score"] = self._world.agent.score
        hud_data["moves"] = self._steps_left
        hud_data["current tier chain"] = (
            self._world.agent.digestion_engine.chained_tiers
        )

        return hud_data
