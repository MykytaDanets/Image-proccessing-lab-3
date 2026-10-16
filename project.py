"""Edge detection: change the parameters, run the file, and compare saved images.

This program uses no internet and opens no camera. All results are saved as files.
Only change the PARAMETERS block when you begin.
"""

# ======================== PARAMETERS ========================
RUN_NAME = "baseline"          # Use a NEW name for every run.
GAUSSIAN_SIGMA = 1.2            # Blur strength in pixels; 0 means no blur.
NOISE_STD = 10.0                # Added Gaussian noise, in 0-255 grey levels.
LOW_THRESHOLD = 40.0           # Canny's lower gradient threshold.
HIGH_THRESHOLD = 100.0         # Canny's upper gradient threshold.
PHOTO_FILENAME = "camera.png"  # Keep this input fixed during the experiments.
SEED = 42                      # Fixed noise seed makes runs comparable.
EDGE_TOLERANCE = 2              # Synthetic boundary matching distance, in pixels.
# ============================================================

from pathlib import Path
import csv
import json
import math
import re

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Save figures without an interactive window.
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def settings():
    """Collect the current parameters for a readable record."""
    return {
        "run_name": RUN_NAME, "gaussian_sigma": GAUSSIAN_SIGMA,
        "noise_std": NOISE_STD, "low_threshold": LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD, "photo_filename": PHOTO_FILENAME,
        "seed": SEED, "edge_tolerance_px": EDGE_TOLERANCE,
        "canny_aperture": 3, "canny_L2gradient": True,
        "laplacian_ksize": 3, "zero_crossing_min_range": 20.0,
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
    }


def validate_parameters():
    """Give clear messages before processing or creating an output folder."""
    if not isinstance(RUN_NAME, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_NAME):
        raise ValueError("RUN_NAME must use only letters, numbers, _ or -.")
    for name, value in [("GAUSSIAN_SIGMA", GAUSSIAN_SIGMA), ("NOISE_STD", NOISE_STD),
                        ("LOW_THRESHOLD", LOW_THRESHOLD), ("HIGH_THRESHOLD", HIGH_THRESHOLD)]:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + " must be a finite number.")
    if not 0 <= GAUSSIAN_SIGMA <= 5:
        raise ValueError("GAUSSIAN_SIGMA must be between 0 and 5.")
    if not 0 <= NOISE_STD <= 60:
        raise ValueError("NOISE_STD must be between 0 and 60.")
    if not 0 <= LOW_THRESHOLD < HIGH_THRESHOLD <= 1443:
        raise ValueError("Use 0 <= LOW_THRESHOLD < HIGH_THRESHOLD <= 1443.")
    if not isinstance(SEED, int) or not 0 <= SEED <= 4294967295:
        raise ValueError("SEED must be a whole number from 0 to 4294967295.")
    if not isinstance(EDGE_TOLERANCE, int) or not 0 <= EDGE_TOLERANCE <= 5:
        raise ValueError("EDGE_TOLERANCE must be a whole number from 0 to 5.")
    # Restrict input to a file in data: paths outside that folder are not accepted.
    if Path(PHOTO_FILENAME).name != PHOTO_FILENAME:
        raise ValueError("PHOTO_FILENAME must be a file name inside data/.")


def synthetic_scene():
    """Build known objects and their one-pixel INSIDE boundary reference.

    The reference comes from the drawing geometry, not from an edge detector.
    The image is synthetic. It is not a photograph of real objects.
    """
    image = np.full((320, 448), 45, dtype=np.uint8)
    reference = np.zeros_like(image, dtype=bool)
    masks = []
    rectangle = np.zeros_like(image)
    cv2.rectangle(rectangle, (45, 55), (170, 235), 255, -1)
    masks.append((rectangle, 205))
    circle = np.zeros_like(image)
    cv2.circle(circle, (300, 140), 70, 255, -1)
    masks.append((circle, 145))
    weak_rectangle = np.zeros_like(image)
    cv2.rectangle(weak_rectangle, (220, 255), (390, 285), 255, -1)
    masks.append((weak_rectangle, 70))
    for mask, intensity in masks:
        image[mask > 0] = intensity
        inside = cv2.erode(mask, np.ones((3, 3), np.uint8))
        reference |= (mask > 0) & (inside == 0)
    return image, reference


def blur_image(image):
    """Apply Gaussian blur explicitly. cv2.Canny does not do this step for us."""
    if GAUSSIAN_SIGMA == 0:
        return image.copy(), 1
    radius = max(1, int(math.ceil(3 * GAUSSIAN_SIGMA)))
    size = 2 * radius + 1
    result = cv2.GaussianBlur(image, (size, size), GAUSSIAN_SIGMA,
                              borderType=cv2.BORDER_REFLECT_101)
    return result, size


def zero_crossings(laplacian, minimum_range=20.0):
    """A simple teaching mask: both signs occur in a 3x3 area, with enough range.

    This produces a candidate BAND around sign changes, not a one-pixel contour.
    Zero-valued flat regions are not accepted. The range test rejects weak noise.
    """
    maximum = cv2.dilate(laplacian, np.ones((3, 3), np.uint8))
    minimum = cv2.erode(laplacian, np.ones((3, 3), np.uint8))
    return ((minimum < 0) & (maximum > 0) &
            ((maximum - minimum) >= minimum_range)).astype(np.uint8) * 255


def process(clean, rng):
    """Compute signed derivatives first; make display images only at the end."""
    noise = rng.normal(0, NOISE_STD, clean.shape)
    noisy = np.clip(np.rint(clean.astype(np.float64) + noise), 0, 255).astype(np.uint8)
    smooth, kernel_size = blur_image(noisy)
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)   # Radians, -pi to pi; x right and y down.
    laplacian = cv2.Laplacian(smooth, cv2.CV_64F, ksize=3)
    edges = cv2.Canny(smooth, LOW_THRESHOLD, HIGH_THRESHOLD,
                      apertureSize=3, L2gradient=True)
    return {"clean": clean, "noisy": noisy, "smooth": smooth, "gx": gx, "gy": gy,
            "magnitude": magnitude, "angle": angle, "laplacian": laplacian,
            "zero_crossings": zero_crossings(laplacian), "canny": edges,
            "gaussian_kernel_size": kernel_size}


def normalise_for_display(array):
    """Stretch a non-negative result to 0-255 FOR DISPLAY ONLY."""
    maximum = float(np.max(array))
    if maximum == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint(255 * array / maximum).astype(np.uint8)


def boundary_rates(predicted, reference, tolerance):
    """Calculate separate tolerant boundary precision and recall.

    Each predicted pixel is correct if some reference pixel is within tolerance.
    Each reference pixel is found if some predicted pixel is within tolerance.
    These are many-to-many boundary rates, not a strict one-to-one benchmark.
    """
    pred = predicted > 0
    truth = reference.astype(bool)
    if not np.any(truth):
        return None, None, None
    # distanceTransform measures distance to the nearest zero pixel.
    to_truth = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)
    precision = float(np.mean(to_truth[pred] <= tolerance)) if np.any(pred) else 0.0
    if np.any(pred):
        to_prediction = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2,
                                              cv2.DIST_MASK_PRECISE)
        recall = float(np.mean(to_prediction[truth] <= tolerance))
    else:
        recall = 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def panel(ax, array, title, cmap="gray", **kwargs):
    ax.imshow(array, cmap=cmap, **kwargs)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def save_comparison(result, filename, label, reference=None):
    """Use fixed grey-level limits for intensity images in every run."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), constrained_layout=True)
    panels = [(result["clean"], "Clean input (0-255)"),
              (result["noisy"], f"Added noise: std={NOISE_STD:g}"),
              (result["smooth"], f"Gaussian sigma={GAUSSIAN_SIGMA:g} px"),
              (normalise_for_display(result["magnitude"]), "Sobel magnitude (display stretch)"),
              (result["canny"], f"Canny: low={LOW_THRESHOLD:g}, high={HIGH_THRESHOLD:g}"),
              (result["zero_crossings"] if reference is None else reference,
               "Laplacian sign-change band" if reference is None else "Known inside boundary reference")]
    for ax, (array, title) in zip(axes.flat, panels):
        panel(ax, array, title, vmin=0, vmax=1 if array.dtype == bool else 255)
    fig.suptitle(label + " | " + RUN_NAME, fontsize=16)
    # Finish and validate the PNG before exposing its final name.
    temporary = filename.with_name(filename.stem + ".tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    if cv2.imread(str(temporary), cv2.IMREAD_UNCHANGED) is None:
        raise OSError("The comparison PNG could not be saved completely.")
    temporary.replace(filename)


def save_outputs(result, folder, label, reference=None):
    """Save actual arrays plus labelled figures that can be read without Python."""
    for name in ["clean", "noisy", "smooth", "canny", "zero_crossings"]:
        cv2.imwrite(str(folder / (label + "_" + name + ".png")), result[name])
    cv2.imwrite(str(folder / (label + "_sobel_display.png")),
                normalise_for_display(result["magnitude"]))
    np.savez_compressed(folder / (label + "_raw_arrays.npz"),
                        gx=result["gx"], gy=result["gy"], magnitude=result["magnitude"],
                        angle_radians=result["angle"], laplacian=result["laplacian"])
    if reference is not None:
        cv2.imwrite(str(folder / (label + "_reference.png")), reference.astype(np.uint8) * 255)
    save_comparison(result, folder / (label + "_comparison.png"), label.title(), reference)


def main():
    """Run one experiment. Existing result folders are preserved."""
    try:
        validate_parameters()
        photo = cv2.imread(str(ROOT / "data" / PHOTO_FILENAME), cv2.IMREAD_GRAYSCALE)
        if photo is None:
            raise ValueError("Cannot read data/" + PHOTO_FILENAME + ". Keep the supplied data folder next to project.py.")
        output = ROOT / "results" / RUN_NAME
        if output.exists():
            print("No files changed: results/" + RUN_NAME + " already exists. Set a NEW RUN_NAME and run again.")
            return None
        scene, reference = synthetic_scene()
        # Independent seeds keep each image's noise pattern fixed between runs.
        photo_result = process(photo, np.random.default_rng(SEED))
        synthetic_result = process(scene, np.random.default_rng(SEED + 1))
        output.mkdir(parents=True)
        save_outputs(photo_result, output, "photo")
        save_outputs(synthetic_result, output, "synthetic", reference)
        measured = []
        for label, result, truth in [("photo", photo_result, None),
                                      ("synthetic", synthetic_result, reference)]:
            precision, recall, f1 = boundary_rates(result["canny"], truth, EDGE_TOLERANCE) if truth is not None else (None, None, None)
            measured.append({"image": label, "edge_pixels": int(np.count_nonzero(result["canny"])),
                             "edge_density_percent": 100 * float(np.mean(result["canny"] > 0)),
                             "boundary_precision": precision, "boundary_recall": recall,
                             "boundary_f1": f1, "tolerance_px": EDGE_TOLERANCE if truth is not None else "not available"})
        with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=measured[0].keys())
            writer.writeheader()
            writer.writerows(measured)
        record = settings()
        record["gaussian_kernel_size"] = photo_result["gaussian_kernel_size"]
        record["metric_note"] = "Photo has no boundary truth: counts are descriptions, not quality scores. Synthetic rates use many-to-many matches within the stated tolerance."
        (output / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("Saved: " + str(output))
        for row in measured:
            print(row)
        return {"output": str(output), "measurements": measured}
    except (ValueError, OSError) as error:
        print("Please check the parameters or files: " + str(error))
        return None


if __name__ == "__main__":
    main()
