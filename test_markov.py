import unittest
import HMM

class TestMarkov(unittest.TestCase):
    def test_load(self):
        e1 = {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
         'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
         'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}

        t1 = {'#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'}, 'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'}, 'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'}, 'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}}

        h = HMM.HMM()
        h.load('cat')
        self.assertEqual(h.emissions, e1)
        self.assertEqual(h.transitions, t1)
        seq = h.generate(10)
        h.forward(seq)
