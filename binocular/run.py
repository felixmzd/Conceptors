"""Run a set of different setups, that exhibit the bistable phenomenon."""
from binocular.experiment import ex


def main():
    # Default experiments presented in detail in the paper.
    ex.run(config_updates={"t_run": 50_000})
    # ex.run(
    #     config_updates={
    #         "pattern_idxs": [50, 77],  # Sine and random periodic pattern.
    #         "SNR": 1.2,
    #         "aperture": 10,
    #         "t_run": 50_000,
    #         "seed": 123
    #     }
    # )
    # ex.run(
    #     config_updates={
    #         "patterns": [46, 51],  # Different sines.
    #         "aperture": 10,
    #         "t_run": 50_000,
    #         "seed": 123,
    #     }
    # )

    # ex.run(
    #     config_updates={
    #         "patterns": [27, 33],  # Other sines.
    #         "aperture": 10,
    #         "t_run": 50_000,
    #         "seed": 123,
    #     }
    # )


if __name__ == "__main__":
    main()
