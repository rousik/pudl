"""Unit tests for pudl.extract.excel module."""
import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
import pandas as pd

import pudl.extract.excel as excel


class TestMetadata(unittest.TestCase):
    """Tests basic operation of the excel.Metadata object."""

    def setUp(self):
        """Cosntructs test metadata instance for testing."""
        self._metadata = excel.Metadata('test')

    def test_basics(self):
        """Test that basic API method return expected results."""
        self.assertEqual('test', self._metadata.get_dataset_name())
        self.assertListEqual(
            ['books', 'boxes', 'shoes'],
            self._metadata.get_all_pages())
        self.assertListEqual(
            ['author', 'pages', 'title'],
            self._metadata.get_all_columns('books'))
        self.assertDictEqual(
            {'book_title': 'title', 'name': 'author', 'pages': 'pages'},
            self._metadata.get_column_map(2010, 'books'))
        self.assertEqual(10, self._metadata.get_skiprows(2011, 'boxes'))
        self.assertEqual(1, self._metadata.get_sheet_name(2011, 'boxes'))


class FakeExtractor(excel.GenericExtractor):
    """Test friendly fake extractor returns strings instead of files."""

    METADATA = excel.Metadata('test')
    BLACKLISTED_PAGES = ['shoes']

    def _load_excel_file(self, year, page):
        return f'{page}-{year}'


def _fake_data_frames(page_name, **kwargs):
    """Returns panda.DataFrames.

    This is suitable mock for pd.read_excel method when used together with FakeExtractor.
    """
    fake_data = {
        'books-2010': pd.DataFrame.from_dict(
            {'book_title': ['Tao Te Ching'], 'name': ['Laozi'], 'pages': [0]}),
        'books-2011': pd.DataFrame.from_dict(
            {'title_of_book': ['The Tao of Pooh'], 'author': ['Benjamin Hoff'], 'pages': [158]}),
        'boxes-2010': pd.DataFrame.from_dict(
            {'composition': ['cardboard'], 'size_inches': [10]}),
        'boxes-2011': pd.DataFrame.from_dict(
            {'composition': ['metal'], 'size_cm': [99]}),
    }
    return fake_data[page_name]


class FakeMetadata(excel.Metadata):
    """This provides fake metadata suitable for testing.

    Both skiprows and sheet_name return 0 for everything but this
    object can keep list of known pages and return column maps.
    """

    def __init__(self, dataset_name, **kwargs):
        """Set up FakeMetadata instance.

        Args:
            dataset_name (str): name of the dataset.
            **kwargs: maps each page to its column map
        """
        self._dataset_name = dataset_name
        self._column_maps = {}
        for page, column_map in kwargs.items():
            self._column_maps[page] = column_map

    @staticmethod
    def get_skiprows(year, page):
        """Returns 0 for everything."""
        return 0

    @staticmethod
    def get_sheet_name(year, page):
        """Returns 0 for everything."""
        return 0

    def get_column_map(self, year, page):
        """Returns column mapping for given page."""
        return self._column_maps[page]

    def get_all_columns(self, page):
        """Returns list of all pudl columns for a given page."""
        return sorted(self._column_maps[page].values())

    def get_all_pages(self):
        """Returns list of all pages."""
        return sorted(self._column_maps.keys())


class TestGenericExtractor(unittest.TestCase):
    """Test operation of the excel.GenericExtractor class."""

    @staticmethod
    @patch('pudl.extract.excel.pd.read_excel')
    def test_read_excel_calls(mock_read_excel):
        """Verifies that read_excel method is called with expected arguments."""
        mock_read_excel.return_value = pd.DataFrame()

        FakeExtractor('/blah').extract([2010, 2011])
        expected_calls = [
            mock.call('books-2010', sheet_name=0, skiprows=0, dtype={}),
            mock.call('books-2011', sheet_name=0, skiprows=1, dtype={}),
            mock.call('boxes-2010', sheet_name=1, skiprows=0, dtype={}),
            mock.call('boxes-2011', sheet_name=1, skiprows=10, dtype={})
        ]
        mock_read_excel.assert_has_calls(expected_calls, any_order=True)

    @patch('pudl.extract.excel.pd.read_excel', _fake_data_frames)
    def test_resulting_dataframes(self):
        """Checks that pages across years are merged and columns are translated."""
        dfs = FakeExtractor('/blah').extract([2010, 2011])
        self.assertEqual(set(['books', 'boxes']), set(dfs.keys()))
        pd.testing.assert_frame_equal(
            pd.DataFrame(data={
                'author': ['Laozi', 'Benjamin Hoff'],
                'pages': [0, 158],
                'title': ['Tao Te Ching', 'The Tao of Pooh'],
            }),
            dfs['books'])
        pd.testing.assert_frame_equal(
            pd.DataFrame(data={
                'material': ['cardboard', 'metal'],
                'size': [10, 99],
            }),
            dfs['boxes'])

    @patch('pudl.extract.excel.pd.read_excel')
    def test_missing_columns_added(self, fake_reader):
        """Test that missing "description" column is added to final page."""
        extractor = FakeExtractor(
            '/blah',
            metadata=FakeMetadata(
                'fake',
                first_page={
                    'orig_name': 'name', 'orig_year': 'year',
                    'orig_desc': 'description'
                }))
        fake_reader.return_value = pd.DataFrame(
            data={'orig_name': ['A'], 'orig_year': ['B']})
        dfs = extractor.extract([2010])
        self.assertEqual(['first_page'], list(dfs.keys()))
        print(dfs['first_page'])
        pd.testing.assert_frame_equal(
            pd.DataFrame(data={
                'description': [np.nan],
                'name': ['A'], 'year': ['B']
            }).astype({'description': 'object'}),
            dfs['first_page'])

    # TODO(rousik@gmail.com): test correct processor operation.
