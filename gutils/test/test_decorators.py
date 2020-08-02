# -*- coding: utf-8 -*-
""" gutils/test/test_decorators """

import io
import re
import unittest
import unittest.mock

from gutils.decorators import timing


mock_stdout = unittest.mock.patch('sys.stdout', new_callable=io.StringIO)


class Test_timing(unittest.TestCase):

    @mock_stdout
    def test_function(self, stdout):
        @timing
        def dummy_fun():
            pass

        dummy_fun()
        self.assertTrue(
            bool(re.match(r'func:dummy_fun processed in [\d.]+ seconds', stdout.getvalue()))
        )


if __name__ == '__main__':
    unittest.main()
