import sys
from types import ModuleType


def error_message_detail(error: Exception, error_detail: ModuleType) -> str:
    """Extract file name, line number, and error message from sys.exc_info()."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in python script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return f"Error: [{str(error)}]"


class AniQRException(Exception):
    """Custom exception class for AniQR project."""

    def __init__(self, error_message: Exception | str, error_detail: ModuleType | None = None):
        if error_detail is not None:
            self.error_message = error_message_detail(
                error_message if isinstance(error_message, Exception) else Exception(str(error_message)),
                error_detail=error_detail,
            )
        else:
            self.error_message = str(error_message)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message
