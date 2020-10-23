from ....settings import BaseSettings, LineupPosition
from ....constants import Sport, Site
from ...sites_registry import SitesRegistry
from ..captain_mode.importer import DraftKingsCaptainModeCSVImporter


POSITIONS_WITH_FLEX = [
    LineupPosition('CPT', ('CPT',)),
    LineupPosition('FLEX', ('FLEX',)),
    LineupPosition('FLEX', ('FLEX',)),
    LineupPosition('FLEX', ('FLEX',)),
    LineupPosition('FLEX', ('FLEX',)),
    LineupPosition('FLEX', ('FLEX',)),
]


class DraftKingsCaptainModeSettings(BaseSettings):
    site = Site.DRAFTKINGS_CAPTAIN_MODE
    budget = 50000
    max_from_one_team = 5
    csv_importer = DraftKingsCaptainModeCSVImporter
    positions = [
        LineupPosition('CPT', ('CPT',)),
        LineupPosition('UTIL', ('UTIL',)),
        LineupPosition('UTIL', ('UTIL',)),
        LineupPosition('UTIL', ('UTIL',)),
        LineupPosition('UTIL', ('UTIL',)),
        LineupPosition('UTIL', ('UTIL',)),
    ]


@SitesRegistry.register_settings
class DraftKingsCaptainModeFootballSettings(DraftKingsCaptainModeSettings):
    sport = Sport.FOOTBALL
    positions = POSITIONS_WITH_FLEX[:]


if __name__ == '__main__':
    pass