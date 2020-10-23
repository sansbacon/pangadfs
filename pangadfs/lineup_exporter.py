import csv
from typing import Iterable, Callable, Any


class LineupExporter:
    def __init__(self, lineups: Iterable[Any]):
        self.lineups = lineups

    @staticmethod
    def render_player(player: Any) -> str:
        result = player.full_name  # type: str
        if player.id:
            result += '(%s)' % player.id
        return result

    def export(self, filename: str, render_func: Callable[[Any], str] = None):
        raise NotImplementedError


class CSVLineupExporter(LineupExporter):
    def export(self, filename, render_func=None):
        with open(filename, 'w') as csvfile:
            lineup_writer = csv.writer(csvfile, delimiter=',')
            for index, lineup in enumerate(self.lineups):
                if index == 0:
                    header = [player.lineup_position for player in lineup.lineup]
                    header.extend(('Budget', 'FPPG'))
                    lineup_writer.writerow(header)
                row = [(render_func or self.render_player)(player) for player in lineup.lineup]
                row.append(str(lineup.salary_costs))
                row.append(str(lineup.fantasy_points_projection))
                lineup_writer.writerow(row)

