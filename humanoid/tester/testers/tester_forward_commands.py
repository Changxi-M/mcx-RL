from .tester_base import TesterTypeBase
from .convert_to_visible_commands import convert_to_visible_commands


class TesterForwardCommands(TesterTypeBase):
    def set_commands(self) -> None:
        convert_to_visible_commands(self.env.commands)
        self.env.commands[:, 0] = abs(self.env.commands[:, 0])
        self.env.commands[:, 1:] = 0
        return None
