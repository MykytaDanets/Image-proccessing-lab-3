"""Optional: run all prescribed experiments without manually editing parameters."""
import project

RUNS = [
    ("baseline", {}),
    ("no_smoothing", {"GAUSSIAN_SIGMA": 0.0}),
    ("strong_smoothing", {"GAUSSIAN_SIGMA": 3.0}),
    ("low_threshold", {"LOW_THRESHOLD": 10.0}),
    ("high_threshold", {"HIGH_THRESHOLD": 180.0}),
    ("no_noise", {"NOISE_STD": 0.0}),
    ("high_noise", {"NOISE_STD": 30.0}),
]


def main():
    # Every run starts from the same baseline. Only one parameter is changed.
    baseline = {"GAUSSIAN_SIGMA": 1.2, "NOISE_STD": 10.0,
                "LOW_THRESHOLD": 40.0, "HIGH_THRESHOLD": 100.0,
                "PHOTO_FILENAME": "camera.png", "SEED": 42, "EDGE_TOLERANCE": 2}
    for name, changes in RUNS:
        for key, value in baseline.items():
            setattr(project, key, value)
        project.RUN_NAME = name
        for key, value in changes.items():
            setattr(project, key, value)
        project.main()


if __name__ == "__main__":
    main()
