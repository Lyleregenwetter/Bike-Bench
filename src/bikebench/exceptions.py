class UserInputException(Exception):
    pass


class InternalError(Exception):
    pass


def check_internal_precondition(precondition: bool, exception_message):
    if not precondition:
        raise InternalError(exception_message)
