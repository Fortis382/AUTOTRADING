from .base import Strategy

class BreakoutDarvas(Strategy):
    def should_long(self, row) -> bool:
        return bool(row.get('darvas_up', False))
    def should_short(self, row) -> bool:
        return bool(row.get('darvas_dn', False))
