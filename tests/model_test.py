import unittest
from napiod import model


class ModelTest(unittest.TestCase):
    def test_state_conversions(self):
        for d in [3, 5, 7, 9]:
            for p in [-1, 0, +1]:
                for i in range(- (d // 2), 1 + d//2):
                    x = model.state_variable_to_state_index(
                        p, i, d)
                    p_ = model.price_change_from_state_index(
                        x, d)
                    i_ = model.imbalance_from_state_index(x, d)
                    self.assertEqual(
                        p, p_)
                    self.assertEqual(i, i_)


if __name__ == '__main__':
    unittest.main()
