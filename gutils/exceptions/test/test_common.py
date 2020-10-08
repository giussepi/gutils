# -*- coding: utf-8 -*-
""" gutils/exceptions/test/test_common """

import unittest


from gutils.exceptions.common import ExclusiveArguments


class Test_ExclusiveArguments(unittest.TestCase):

    def setUp(self):
        self.args_list = ['arg1', 'arg2', 'arg3']

    def test_raises(self):
        with self.assertRaises(ExclusiveArguments):
            raise ExclusiveArguments(self.args_list)

    def test_exception_message(self):
        message = ExclusiveArguments.message_template.format(', '.join(self.args_list))
        with self.assertRaisesRegex(ExclusiveArguments, message):
            raise ExclusiveArguments(self.args_list)
