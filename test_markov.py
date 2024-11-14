import unittest
import HMM

class TestMarkov(unittest.TestCase):
    def test_load(self):
        h = HMM.HMM()
        h.load('cat')