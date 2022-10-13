from dimensional_reduction import BarnesHutSigmoidDecomposition
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def test_barnes_hut():
    iris = load_iris()
    X = iris.data
    model = BarnesHutSigmoidDecomposition(
        iterations=2,
        learning_rate=1,
        depth=4
    )
    result = model.fit_transform(X)
