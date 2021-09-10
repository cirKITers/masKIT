from maskit.examples.load_data import load_data


class TestLoadData:
    def test_iris_basic(self):
        data = load_data(
            "iris",
            train_size=120,
            test_size=30,
        )

        assert len(data.train_data) == 120
        assert len(data.train_target) == 120
        assert len(data.test_data) == 30
        assert len(data.test_target) == 30

        assert len(data.train_data[0]) == 4
        assert len(data.train_target[0]) == 4
        assert len(data.test_data[0]) == 4
        assert len(data.test_target[0]) == 4

    def test_iris_empty_test(self):
        data = load_data(
            "iris",
            train_size=150,
        )

        assert len(data.train_data) == 150
        assert len(data.train_target) == 150
        assert len(data.test_data) == 0
        assert len(data.test_target) == 0

    def test_circles_basic(self):
        data = load_data(
            "circles",
            train_size=150,
            test_size=50,
        )

        assert len(data.train_data) == 150
        assert len(data.train_target) == 150
        assert len(data.test_data) == 50
        assert len(data.test_target) == 50

        assert len(data.train_data[0]) == 2
        assert len(data.train_target[0]) == 2
        assert len(data.test_data[0]) == 2
        assert len(data.test_target[0]) == 2

    def test_mnist_basic(self):
        data = load_data(
            "mnist", wires=5, classes=[6, 7, 8], train_size=150, test_size=50
        )

        assert len(data.train_data) == 150
        assert len(data.train_target) == 150
        assert len(data.test_data) == 50
        assert len(data.test_target) == 50

        assert len(data.train_data[0]) == 5
        assert len(data.train_target[0]) == 4
        assert len(data.test_data[0]) == 5
        assert len(data.test_target[0]) == 4
