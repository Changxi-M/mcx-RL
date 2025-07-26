from .tester_base import TesterTypeBase
from .convert_to_visible_commands import convert_to_visible_commands


class TesterNormalCommands(TesterTypeBase):

    def set_commands(self) -> None:
        convert_to_visible_commands(self.env.commands)
        return None
