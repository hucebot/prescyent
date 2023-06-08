"""Here is a super class for all custom exceptions"""


class CustomException(Exception):
    """Defines the custom exception behavior in the lib
    Inherits from Exception
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return self.message
