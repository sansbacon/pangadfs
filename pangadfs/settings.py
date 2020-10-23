from typing import Optional, Type, List, Sequence, TYPE_CHECKING

from .lineup_printer import LineupPrinter, BaseLineupPrinter
from .lineup_exporter import LineupExporter, CSVLineupExporter


class LineupPosition:
    def __init__(self, name: str, positions: Sequence[str]):
        self.name = name
        self.positions = positions


class BaseSettings:
    site = None  # type: str
    sport = None  # type: str
    budget = 0  # type: Optional[float]
    positions = []  # type: List[LineupPosition]
    max_from_one_team = None  # type: Optional[int]
    min_teams = None  # type: Optional[int]
    min_games = None  # type: Optional[int]
    total_teams_exclude_positions = []  # type: List[str]
    lineup_printer = LineupPrinter  # type: Type[BaseLineupPrinter]
    extra_rules = []  # type: List[Type['OptimizerRule']]
    csv_importer = None  # type: Type['CSVImporter']
    csv_exporter = CSVLineupExporter  # type: Type[LineupExporter]

    @classmethod
    def get_total_players(cls) -> int:
        return len(cls.positions)
