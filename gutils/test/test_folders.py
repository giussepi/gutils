# -*- coding: utf-8 -*-
""" gutils/test/test_folders """

import os
import unittest

from gutils.folders import remove_folder, clean_create_folder


class FolderMixin:

    def setUp(self):
        self.folder_name = 'dummy_test_folder'
        self.folder_structure = os.path.join(self.folder_name, 'subfolder1', 'subfolder2')


class Test_remove_folder(FolderMixin, unittest.TestCase):

    def test_function_nothing_to_delete(self):
        remove_folder(self.folder_name)
        self.assertTrue(True)

    def test_function_del_folder(self):
        os.makedirs(self.folder_structure)
        remove_folder(self.folder_name)
        self.assertFalse(os.path.isdir(self.folder_name))


class Test_clean_create_folder(FolderMixin, unittest.TestCase):

    def test_function_folder_does_no_exists(self):
        clean_create_folder(self.folder_name)
        self.assertTrue(os.path.isdir(self.folder_name))
        self.assertEqual(len(os.listdir(self.folder_name)), 0)

    def test_function_folder_already_exists(self):
        os.mkdir(self.folder_name)
        clean_create_folder(self.folder_name)
        self.assertTrue(os.path.isdir(self.folder_name))
        self.assertEqual(len(os.listdir(self.folder_name)), 0)


if __name__ == '__main__':
    unittest.main()
