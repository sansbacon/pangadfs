from typing import List
from ....settings import BaseSettings, LineupPosition
from ....constants import Sport, Site
from ...sites_registry import SitesRegistry
from ....lineup_printer import DraftKingTiersLineupPrinter
from ...draftkings.tiers.importer import DraftKingsTiersCSVImporter


class DraftKingsTiersSettings(BaseSettings):
    site = Site.DRAFTKINGS_TIERS
    csv_importer = DraftKingsTiersCSVImporter
    min_games = 2
    budget = None
    positions = []  # type: List[LineupPosition]
    extra_rules = []
    lineup_printer = DraftKingTiersLineupPrinter


@SitesRegistry.register_settings
class DraftKingsTiersFootballSettings(DraftKingsTiersSettings):
    sport = Sport.FOOTBALL


