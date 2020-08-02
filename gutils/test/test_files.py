# -*- coding: utf-8 -*-
""" gutils/test/test_files """

import os
import re
import unittest

from gutils.files import clean_json_filename, get_filename_and_extension


class Test_get_filename_and_extension(unittest.TestCase):

    def test_function_len_greater_than_two(self):
        fullpath = os.path.join('directory1', 'directory2', 'file.name.txt')
        self.assertEqual(get_filename_and_extension(fullpath), tuple(['file.name', 'txt']))
        fullpath = os.path.join('directory1', 'file.name.txt')
        self.assertEqual(get_filename_and_extension(fullpath), tuple(['file.name', 'txt']))

    def test_function_len_equals_two(self):
        fullpath = os.path.join('filename.txt')
        self.assertEqual(get_filename_and_extension(fullpath), tuple(['filename', 'txt']))

    def test_function_len_equals_one(self):
        fullpath = os.path.join('filename')
        self.assertEqual(get_filename_and_extension(fullpath), tuple(['filename', '']))


class Test_clean_json_filename(unittest.TestCase):

    def test_function_no_str_argument(self):
        with self.assertRaises(AssertionError):
            clean_json_filename(9)

    def test_function_no_json_extension(self):
        with self.assertRaisesRegex(AssertionError, r'The filename does not have a .json extension'):
            clean_json_filename('filename.txt')

    def test_function_normal_argument(self):
        self.assertEqual(clean_json_filename('filename.json'), 'filename.json')

    def test_funcion_empty_argument(self):
        self.assertTrue(bool(re.match(r'[\d\w]+.json', clean_json_filename(''))))


if __name__ == '__main__':
    unittest.main()
