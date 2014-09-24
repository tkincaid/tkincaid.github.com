# -*- coding: utf-8 -*-
import os
import unittest
import codecs

from common.encoding import detect_encoding, detect_file_contents_encoding
from cStringIO import StringIO
import numpy as np

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], '..', 'testdata')


class TestEncodingDetection(unittest.TestCase):

    def test_encoding_amazon_de_reviews_is_utf8(self):
        """Tests detect_encoding on 'amazon_de_reviews_200'
        """
        guessed_encoding = detect_encoding(
            os.path.join(TEST_DATA_DIR, 'amazon_de_reviews_200.csv'))
        self.assertEqual(guessed_encoding.lower(), u'utf-8')

    def test_encoding_win(self):
        """Tests detect_encoding on 'bad_codec.csv'
        Should give WINDOWS-1252
        """
        guessed_encoding = detect_encoding(
            os.path.join(TEST_DATA_DIR, 'bad_codec.csv'))
        self.assertEqual(guessed_encoding.lower(), u'windows-1252')

    def test_encoding_ascii(self):
        """Tests detect_encoding on 'amazon-sample-1000.csv'

        Should give ascii
        """
        guessed_encoding = detect_encoding(
            os.path.join(TEST_DATA_DIR, 'amazon-sample-1000.csv'))
        self.assertEqual(guessed_encoding.lower(), u'ascii')

    def test_encoding_with_continuation_characters_after_win_1252_chars(self):
        '''I found a dataset where the first chunk was determined to be
        WIN-1252, and a later block was determined to be 'UTF-8', but with the
        exact same confidence score, so with the implementation we had written
        the determination stayed as WIN-1252
        '''
        guessed_encoding = detect_encoding(
            os.path.join(TEST_DATA_DIR, 'text_encoding_test.csv'))
        self.assertEqual(guessed_encoding.lower(), u'utf-8')

    def test_encoding_empty(self):
        """Tests detect_encoding on empty file"""
        guessed_encoding = detect_encoding(
            os.path.join(TEST_DATA_DIR, 'empty_file.csv'))
        self.assertEqual(guessed_encoding.lower(), u'ascii')

class TestUnsupportedEncodings(unittest.TestCase):
    '''A smattering of test cases presenting encodings that we know exist, but
    since we haven't tested the full sytem with them, we will reject them
    '''
    SOURCE_TEXT = ('Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
                   ' Aenean id dui in tellus egestas blandit at id lacus.'
                   ' Maecenas pulvinar gravida vestibulum.'
                   ' Suspendisse sed odio dapibus, tempus velit nec.')

    def test_utf16be_not_supported_yet(self):
        '''Not really on purpose, but we haven't tested it system-wide'''
        text = self.SOURCE_TEXT.decode('utf-8').encode('utf-16be')
        text = codecs.BOM_UTF16_BE + text

        with self.assertRaises(ValueError) as ve:
            enc = detect_file_contents_encoding(StringIO(text))
            print 'Should not find UTF-16BE data to be {}'.format(enc)

    def test_utf16le_not_supported_yet(self):
        '''Not really on purpose, but we haven't tested it system-wide'''
        text = self.SOURCE_TEXT.decode('utf-8').encode('utf-16le')
        text = codecs.BOM_UTF16_LE + text

        with self.assertRaises(ValueError) as ve:
            enc = detect_file_contents_encoding(StringIO(text))
            print 'Should not find UTF-16LE data to be {}'.format(enc)

    def test_utf32be_not_supported_yet(self):
        '''Not really on purpose, but we haven't tested it system-wide'''
        text = self.SOURCE_TEXT.decode('utf-8').encode('utf-32be')
        text = codecs.BOM_UTF32_BE + text

        with self.assertRaises(ValueError) as ve:
            enc = detect_file_contents_encoding(StringIO(text))
            print 'Should not find UTF-32BE data to be {}'.format(enc)

    def test_utf16le_not_supported_yet(self):
        '''Not really on purpose, but we haven't tested it system-wide'''
        text = self.SOURCE_TEXT.decode('utf-8').encode('utf-32le')
        text = codecs.BOM_UTF32_LE + text

        with self.assertRaises(ValueError) as ve:
            enc = detect_file_contents_encoding(StringIO(text))
            print 'Should not find UTF-32LE data to be {}'.format(enc)


class TestEncodingDetectorImplementation(unittest.TestCase):
    '''Here we can use StringIO to make a bunch of test cases and not have to
    read the disk so much

    '''
    def test_latin_encoder_should_be_windows_1252(self):
        text = ' '.join(self.rng.choice(self.ASCII_WORDS, 100))
        text += ' '.join(self.rng.choice(self.LATIN_WORDS, 100))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'windows-1252')

    def test_unicode_encoder_should_not_be_latin(self):
        text = ' '.join(self.rng.choice(self.ASCII_WORDS, 100))
        text += ' '.join(self.rng.choice(self.UNICODE_WORDS, 50))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'utf-8')

    def test_can_find_utf8_bom(self):
        '''The UTF8 BOM is not necessary or recommended, but it is not
        disallowed either
        '''
        text = codecs.BOM_UTF8 + ' '.join(
            self.rng.choice(self.ASCII_WORDS, 100))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'utf-8-sig')

    def test_ascii_encoder(self):
        text = ' '.join(self.rng.choice(self.ASCII_WORDS, 100))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'ascii')

    def test_latin_CX_character_on_boundary(self):
        text = ''.join([chr(194) for i in range(20)])
        io = StringIO(text)
        enc = detect_file_contents_encoding(io, chunksize=10)
        self.assertEqual(enc.lower(), 'windows-1252')

    def test_unicode_CX_character_on_boundary(self):
        text = 'abcdefghi\xc3\xa1abcdefghi'
        print(text)
        io = StringIO(text)
        enc = detect_file_contents_encoding(io, chunksize=10)
        self.assertEqual(enc.lower(), 'utf-8')

    def test_unexpected_end_of_file_doesnt_kill(self):
        '''This case present an invalid UTF-8 string because C3 expects
        a continuation character.  This just makes sure that if a file
        ends with C3 that the detector won't crap out
        '''
        text = 'abcdefghi\xc3'
        io = StringIO(text)
        enc = detect_file_contents_encoding(io, chunksize=100)
        self.assertNotEqual(enc.lower(), 'utf-8')

    def test_complete_garbage(self):
        '''The following is not ASCII, WINDOWS-1252 or UTF-8'''
        text = 'IamNonsense\xa0DoYouNotSee\x81'
        io = StringIO(text)
        with self.assertRaises(ValueError):
            enc = detect_file_contents_encoding(io, chunksize=100)

    def test_CR_file(self):
        text = '\x0d'.join(self.rng.choice(self.ASCII_WORDS, 50))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'ascii')

    def test_LF_file(self):
        text = '\x0a'.join(self.rng.choice(self.ASCII_WORDS, 50))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'ascii')

    def test_CRLF_file(self):
        text = '\x0d\x0a'.join(self.rng.choice(self.ASCII_WORDS, 50))
        enc = detect_file_contents_encoding(StringIO(text))
        self.assertEqual(enc.lower(), 'ascii')

    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.ASCII_WORDS = ['apple', 'banana', 'cookie', 'doughnut']
        self.LATIN_WORDS = ['guaraná', 'e=mc²', 'Müeller', 'Maßstab',
                            '\xa0nobreak'] # \xa0 = NBSP in windows-1252
        self.UNICODE_WORDS = ['Ꮦ', 'ෟ', 'Ⴓ', 'Ⴟ']

