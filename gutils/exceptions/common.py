# -*- coding: utf-8 -*-
""" gutils/exceptions/common """


class ExclusiveArguments(Exception):
    """
    Exception to be raised when two or more exclusive attributes are provided

    Usage:
        raise ExclusiveArguments([arg1, arg2, ..])
    """

    message_template = 'The arguments "{}" are exclusive. You must provide only one of them.'

    def __init__(self, args_list, message=''):
        """ Initialized the instance with a custom message """
        assert isinstance(args_list, (list, tuple))

        if not message:
            message = self.message_template.format(', '.join(args_list))

        super().__init__(message)
