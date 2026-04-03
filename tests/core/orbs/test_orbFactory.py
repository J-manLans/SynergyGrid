from syn_grid.config.models import OrbFactoryConf
from syn_grid.core.orbs.orb_factory import OrbFactory
from tests.utils.config_helpers import get_test_config, update_conf

import pytest


class TestOrbFactory:

    @pytest.fixture
    def factory_tuple(self) -> tuple[OrbFactory, OrbFactoryConf]:
        conf = get_test_config().world

        return (
            OrbFactory(
                conf.orb_factory_conf, conf.negative_orb_conf, conf.tier_orb_conf
            ),
            conf.orb_factory_conf,
        )

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4, 5])
    def test_create_orbs_scales_up_to_min_pool_size(
        self, factory_tuple: tuple[OrbFactory, OrbFactoryConf], tier: int
    ):
        factory, conf = factory_tuple
        conf = update_conf(conf, {"max_tier": tier, "max_active_orbs": 3})
        orbs = factory.create_orbs()

        assert len(orbs) == conf.max_active_orbs * 3
