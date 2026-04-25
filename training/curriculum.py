# training/curriculum.py

from dataclasses import dataclass
import random


@dataclass
class CurriculumConfig:
    """
    Maps mean reward → difficulty distribution
    """

    thresholds: dict

    def get_distribution(
        self,
        mean_reward: float,
    ) -> dict:
        for threshold in sorted(
            self.thresholds.keys(),
            reverse=True,
        ):
            if mean_reward >= threshold:
                return self.thresholds[threshold]

        return self.thresholds[
            min(self.thresholds.keys())
        ]


DEFAULT_CURRICULUM = CurriculumConfig(
    thresholds={
        0.0: {
            1: 0.90,
            2: 0.10,
            3: 0.00,
            4: 0.00,
        },

        0.30: {
            1: 0.50,
            2: 0.40,
            3: 0.10,
            4: 0.00,
        },

        0.50: {
            1: 0.20,
            2: 0.40,
            3: 0.35,
            4: 0.05,
        },

        0.65: {
            1: 0.10,
            2: 0.30,
            3: 0.40,
            4: 0.20,
        },
    }
)


def sample_difficulty(
    curriculum: CurriculumConfig,
    mean_reward: float,
) -> int:
    """
    Sample difficulty from curriculum schedule.
    """

    dist = curriculum.get_distribution(
        mean_reward
    )

    return random.choices(
        list(dist.keys()),
        weights=list(dist.values()),
        k=1,
    )[0]