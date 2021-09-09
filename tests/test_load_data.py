from maskit.examples.load_data import load_data


class TestLoadData:
    def test_iris_basic(self):
        params = {"train_size": 120}
        train_data, train_target, test_data, test_target = load_data(
            "iris", 4, None, params
        )

        assert len(train_data) == 120
        assert len(train_target) == 120
        assert len(test_data) == 30
        assert len(test_target) == 30

        assert len(train_data[0]) == 4
        assert len(train_target[0]) == 4
        assert len(test_data[0]) == 4
        assert len(test_target[0]) == 4

    def test_iris_empty_test(self):
        params = {"train_size": 150}
        train_data, train_target, test_data, test_target = load_data(
            "iris", 4, None, params
        )

        assert len(train_data) == 150
        assert len(train_target) == 150
        assert len(test_data) == 0
        assert len(test_target) == 0

    def test_circles_basic(self):
        params = {}
        train_data, train_target, test_data, test_target = load_data(
            "circles", 4, None, params
        )

        assert len(train_data) == 150
        assert len(train_target) == 150
        assert len(test_data) == 50
        assert len(test_target) == 50

        assert len(train_data[0]) == 2
        # TODO: change to 2 once the circuit is generalised
        assert len(train_target[0]) == 4
        assert len(test_data[0]) == 2
        # TODO: change to 2 once the circuit is generalised
        assert len(test_target[0]) == 4

    def test_circles_custom_size(self):
        params = {"train_size": 100, "test_size": 25}
        train_data, train_target, test_data, test_target = load_data(
            "circles", 4, None, params
        )

        assert len(train_data) == 100
        assert len(train_target) == 100
        assert len(test_data) == 25
        assert len(test_target) == 25

    def test_mnist_basic(self):
        params = {"train_size": 150, "test_size": 50, "classes": [6, 7, 8]}
        train_data, train_target, test_data, test_target = load_data(
            "mnist", 4, None, params
        )

        assert len(train_data) == 150
        assert len(train_target) == 150
        assert len(test_data) == 50
        assert len(test_target) == 50

        assert len(train_data[0]) == 4
        # TODO: change once the circuit is generalised
        assert len(train_target[0]) == 4
        assert len(test_data[0]) == 4
        # TODO: change once the circuit is generalised
        assert len(test_target[0]) == 4
