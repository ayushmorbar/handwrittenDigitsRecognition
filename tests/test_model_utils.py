import unittest
from src.basic_nn import build_basic_nn
from src.cnn import build_cnn

class TestModelUtils(unittest.TestCase):

    def test_basic_nn(self):
        """Test if the basic NN model has the correct number of layers."""
        model = build_basic_nn()
        self.assertEqual(len(model.layers), 3)

    def test_cnn(self):
        """Test if the CNN model has the correct number of layers."""
        model = build_cnn()
        self.assertEqual(len(model.layers), 6)

if __name__ == '__main__':
    unittest.main()
