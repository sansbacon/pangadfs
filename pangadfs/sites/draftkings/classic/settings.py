from ....settings import BaseSettings, LineupPosition
from ....constants import Sport, Site
from ...sites_registry import SitesRegistry
from ....lineup_printer import IndividualSportLineupPrinter
from .importer import DraftKingsCSVImporter
from ..captain_mode.importer import DraftKingsCaptainModeCSVImporter


class DraftKingsSettings(BaseSettings):
    site = Site.DRAFTKINGS
    budget = 50000
    #csv_importer = DraftKingsCSVImporter


@SitesRegistry.register_settings
class DraftKingsFootballSettings(DraftKingsSettings):
    sport = Sport.FOOTBALL
    min_games = 2
    positions = [
        LineupPosition('QB', ('QB',)),
        LineupPosition('RB', ('RB',)),
        LineupPosition('RB', ('RB',)),
        LineupPosition('WR', ('WR',)),
        LineupPosition('WR', ('WR',)),
        LineupPosition('WR', ('WR',)),
        LineupPosition('TE', ('TE',)),
        LineupPosition('FLEX', ('WR', 'RB', 'TE')),
        LineupPosition('DST', ('DST',))
    ]


