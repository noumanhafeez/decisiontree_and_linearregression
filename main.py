from decision_tree.src.mode_controller import main

if __name__ == "__main__":

    samples = [
        {"odor": 6, "gill-size": 1, "cap-surface": 2},
        {"odor": 3, "gill-size": 0, "cap-surface": 2},
        {"odor": 5, "gill-size": 0, "cap-surface": 2},
    ]

    mode = ("predict")   # or "train"

    main(mode=mode, samples=samples)