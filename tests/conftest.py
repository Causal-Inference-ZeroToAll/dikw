import numpy as np
import pytest

from dikw.dataset import synthetic_data
from dikw.dataset import make_uplift_classification

from .const import RANDOM_SEED, N_SAMPLE, TREATMENT_NAMES, CONVERSION


@pytest.fixture(scope="module")
def generate_regression_data(mode: int = 1, p: int = 8, sigma: float = 0.1):
    generated = False

    def _generate_data(mode: int = mode, p: int = p, sigma: float = sigma):
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = synthetic_data(mode=mode, n=N_SAMPLE, p=p, sigma=sigma)

        return data

    yield _generate_data


@pytest.fixture(scope="module")
def generate_classification_data():
    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(
                n_samples=N_SAMPLE,
                treatment_name=TREATMENT_NAMES,
                y_name=CONVERSION,
                random_seed=RANDOM_SEED,
            )

        return data

    yield _generate_data


@pytest.fixture(scope="module")
def generate_classification_data_two_treatments():
    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(
                n_samples=N_SAMPLE,
                treatment_name=TREATMENT_NAMES[0:2],
                y_name=CONVERSION,
                random_seed=RANDOM_SEED,
            )

        return data

    yield _generate_data
