import unittest
from src.data_loader import load_mnist_data

class TestDataLoader(unittest.TestCase):

    def test_data_shape(self):
        """Test if the data shapes are as expected."""
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        self.assertEqual(x_train.shape, (60000, 28, 28, 1))
        self.assertEqual(y_train.shape, (60000, 10))
        self.assertEqual(x_test.shape, (10000, 28, 28, 1))
        self.assertEqual(y_test.shape, (10000, 10))

if __name__ == '__main__':
    unittest.main()