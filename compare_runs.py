import argparse
import json
import os
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare multiple MLP training runs.")
    parser.add_argument(
        "log_files",
        nargs="+",
        type=str,
        help="Paths to one or more JSON training history files.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    _, axes = plt.subplots(2, 2, figsize=(16, 10))
    (ax1, ax2), (ax3, ax4) = axes

    for filepath in args.log_files:
        with open(filepath, "r") as f:
            history = json.load(f)

        base_name = os.path.basename(filepath)
        run_label = os.path.splitext(base_name)[0]

        ax1.plot(history["validation_loss"], label=run_label, linewidth=2)
        ax2.plot(history["validation_accuracy"], label=run_label, linewidth=2)
        ax3.plot(history["validation_precision"], label=run_label, linewidth=2)
        ax4.plot(history["validation_recall"], label=run_label, linewidth=2)

    ax1.set_title("Validation Loss Comparison")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_title("Validation Accuracy Comparison")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3.set_title("Validation Precision Comparison")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Precision")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax4.set_title("Validation Recall Comparison")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Recall")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
