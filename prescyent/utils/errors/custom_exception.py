"""Here is a super class for all custom exceptions"""


class CustomException(Exception):
    """Defines the custom exception behavior in the lib
        Inherits from Exception
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "%s raised: %s" % (self.__class__.__name__, self.message)
