
import logging
from pangadfs.base import PluginBase

class TestPluginMutate(PluginBase):
    """test_plugin plugin for mutate stage."""

    def __init__(self, ctx=None):
        super().__init__()
        self.param = ctx.get('param', 0.5) if ctx else 0.5
        self.logger = logging.getLogger(__name__)

    def mutate(self, *args, **kwargs):
        """Your implementation goes here."""
        self.logger.info("Running test_plugin plugin.")
        return args[0]  # placeholder
