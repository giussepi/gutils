# -*- coding: utf-8 -*-
""" utils/test/test_utils """

import re
import unittest

from utils.utils import get_random_string


class Test_get_random_string(unittest.TestCase):

    def test_function(self):
        rand_string = get_random_string(5)
        self.assertEqual(len(rand_string), 5)
        self.assertTrue(bool(re.match(r'[\d\w]+', rand_string)))


if __name__ == '__main__':
    unittest.main()
