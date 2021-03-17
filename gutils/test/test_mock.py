# -*- coding: utf-8 -*-
""" gutils/test/test_mock """

import io

import unittest
import unittest.mock

from gutils.mock import notqdm


mock_stdout = unittest.mock.patch('sys.stdout', new_callable=io.StringIO)


class Test_no_tqdm(unittest.TestCase):
    @mock_stdout
    def test_notqdm(self, stdout):
        for _ in notqdm(range(1)):
            pass

        self.assertEqual('', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
