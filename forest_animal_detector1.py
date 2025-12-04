import cv2
import numpy as np
import os

# -----------------------------------
# 0. GLOBAL CONFIG
# -----------------------------------
# If True: whenever model predicts 'cow', we DISPLAY it as 'deer'
# (handy if your wildlife videos are mostly deer)
USE_DEER_OVERRIDE = True

# Default paths (used if user just presses Enter)
DEFAULT_IMAGE_PATH = "forest_test.jpeg"
DEFAULT_VIDEO_PATH = "forest_video.mp4"

WEIGHTS_PATH = "yolov3.weights"
CFG_PATH = "yolov3.cfg"
NAMES_PATH = "coco.names"

PANEL_WIDTH = 380
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


# -----------------------------------
# 1. Species info dictionary
# -----------------------------------
species_info = {
    "bird": {
        "common_name": "Bird",
        "scientific_name": "Aves",
        "diet": "Varies (seeds, insects, fruits, fish)",
        "habitat": "Forests, grasslands, wetlands, urban areas",
        "status": "Varies by species",
        "fun_fact": "Some birds migrate thousands of kilometers every year."
    },
    "cat": {
        "common_name": "Wild / Feral Cat",
        "scientific_name": "Felis catus",
        "diet": "Carnivore",
        "habitat": "Grasslands, forests, near human settlements",
        "status": "Least Concern",
        "fun_fact": "Cats can rotate their ears about 180 degrees."
    },
    "dog": {
        "common_name": "Wild / Feral Dog",
        "scientific_name": "Canis lupus familiaris",
        "diet": "Omnivore (mostly carnivorous in wild)",
        "habitat": "Near human settlements, forests, rural areas",
        "status": "Domesticated species",
        "fun_fact": "Dogs have a sense of smell much stronger than humans."
    },
    "horse": {
        "common_name": "Horse",
        "scientific_name": "Equus ferus caballus",
        "diet": "Herbivore (grasses, hay)",
        "habitat": "Grasslands, farms, open plains",
        "status": "Domesticated species",
        "fun_fact": "Horses can sleep both lying down and standing up."
    },
    "sheep": {
        "common_name": "Sheep",
        "scientific_name": "Ovis aries",
        "diet": "Herbivore (grass, plants)",
        "habitat": "Grasslands, farms, hills",
        "status": "Domesticated species",
        "fun_fact": "Sheep have wide peripheral vision due to rectangular pupils."
    },
    "cow": {
        "common_name": "Cow",
        "scientific_name": "Bos taurus",
        "diet": "Herbivore (grass, fodder)",
        "habitat": "Grasslands, farms",
        "status": "Domesticated species",
        "fun_fact": "Cows form strong social bonds and can have best friends."
    },
    "deer": {
        "common_name": "Deer",
        "scientific_name": "Cervidae (family)",
        "diet": "Herbivore (grass, leaves, shoots)",
        "habitat": "Forests, grasslands, meadows",
        "status": "Varies by species",
        "fun_fact": "Deer shed and re-grow their antlers every year."
    },
    "elephant": {
        "common_name": "Elephant",
        "scientific_name": "Elephas / Loxodonta spp.",
        "diet": "Herbivore (leaves, bark, fruits)",
        "habitat": "Forests, savannas, grasslands",
        "status": "Vulnerable / Endangered",
        "fun_fact": "Elephants can recognize themselves in a mirror."
    },
    "bear": {
        "common_name": "Bear",
        "scientific_name": "Ursidae (family)",
        "diet": "Omnivore (plants, fruits, fish, meat)",
        "habitat": "Forests, mountains, tundra",
        "status": "Varies by species",
        "fun_fact": "Some bears hibernate for months without eating or drinking."
    },
    "zebra": {
        "common_name": "Zebra",
        "scientific_name": "Equus quagga",
        "diet": "Herbivore (grasses)",
        "habitat": "Savannas, grasslands",
        "status": "Near Threatened (some species)",
        "fun_fact": "Each zebra has a unique stripe pattern like a fingerprint."
    },
    "giraffe": {
        "common_name": "Giraffe",
        "scientific_name": "Giraffa camelopardalis",
        "diet": "Herbivore (leaves, especially acacia)",
        "habitat": "Savannas, open woodlands",
        "status": "Vulnerable",
        "fun_fact": "Giraffes have the same number of neck vertebrae as humans (seven)."
    }
}


# -----------------------------------
# 2. Load YOLO model
# -----------------------------------
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

animal_classes = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# -----------------------------------
# 3. Detection on a single frame
# -----------------------------------
def detect_animals(frame):
    """Run YOLO on a frame. Returns:
       - annotated frame
       - list of detected species
       - dict of species -> count
    """
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(416, 416),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf > CONF_THRESHOLD:
                raw_label = classes[class_id]
                if raw_label not in animal_classes:
                    continue

                cx = int(det[0] * w)
                cy = int(det[1] * h)
                bw = int(det[2] * w)
                bh = int(det[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD
    )

    detected_species = []
    species_counts = {}

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf = confidences[i]
            raw_label = classes[class_ids[i]]

            # Deer override hack
            if USE_DEER_OVERRIDE and raw_label == "cow":
                label = "deer"
            else:
                label = raw_label

            if label not in detected_species:
                detected_species.append(label)

            species_counts[label] = species_counts.get(label, 0) + 1

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
            text = f"{label} {conf:.2f}"
            (tw, th), base = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x, y - th - base), (x + tw, y),
                (0, 220, 0), -1
            )
            cv2.putText(
                frame, text, (x, y - base),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA
            )

    return frame, detected_species, species_counts


# -----------------------------------
# 4. Draw counts on frame
# -----------------------------------
def draw_counts_on_frame(frame, species_counts):
    """Overlay species counts in the top-left corner."""
    if not species_counts:
        return frame

    y = 25
    # Title with black outline
    cv2.putText(
        frame, "Counts:",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, "Counts:",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    y += 22

    for sp, cnt in species_counts.items():
        text = f"{sp}: {cnt}"
        cv2.putText(
            frame, text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 20

    return frame


# -----------------------------------
# 5. Build right-side info panel
# -----------------------------------
def build_info_panel(height, detected_species):
    """Create the dark info panel with species details."""
    panel = np.full((height, PANEL_WIDTH, 3), (30, 30, 30), dtype=np.uint8)

    x_text = 15
    y_text = 35
    line_h = 22

    cv2.putText(
        panel, "Detected Species Info",
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    y_text += int(line_h * 1.5)

    if not detected_species:
        cv2.putText(
            panel, "No animals detected",
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (180, 180, 180), 2, cv2.LINE_AA
        )
        return panel

    max_chars = 34

    for sp in detected_species:
        info = species_info.get(sp)

        header = sp.capitalize()
        cv2.putText(
            panel, header,
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2, cv2.LINE_AA
        )
        y_text += line_h

        if info:
            info_lines = [
                f"Common: {info['common_name']}",
                f"Sci: {info['scientific_name']}",
                f"Diet: {info['diet']}",
                f"Habitat: {info['habitat']}",
                f"Status: {info['status']}",
                f"Fact: {info['fun_fact']}",
            ]
        else:
            info_lines = ["Info not in database."]

        for line in info_lines:
            while len(line) > 0:
                part = line[:max_chars]
                line = line[max_chars:]
                cv2.putText(
                    panel, part,
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA
                )
                y_text += int(line_h * 0.95)
                if y_text > height - 25:
                    break
            if y_text > height - 25:
                break

        y_text += int(line_h * 0.8)
        cv2.line(
            panel,
            (10, y_text - int(line_h * 0.5)),
            (PANEL_WIDTH - 10, y_text - int(line_h * 0.5)),
            (80, 80, 80),
            1
        )

        if y_text > height - 40:
            break

    return panel


# -----------------------------------
# 6. Image mode
# -----------------------------------
def run_on_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    frame, detected_species, species_counts = detect_animals(frame)
    frame = draw_counts_on_frame(frame, species_counts)
    panel = build_info_panel(frame.shape[0], detected_species)
    combined = np.concatenate((frame, panel), axis=1)

    # Build output filename
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_name = f"{base}_annotated.png"
    cv2.imwrite(out_name, combined)
    print(f"[INFO] Saved annotated image as {out_name}")

    cv2.namedWindow("Forest Animal Detection - Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Forest Animal Detection - Image", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------------
# 7. Video mode
# -----------------------------------
def run_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = width + PANEL_WIDTH
    out_h = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"{base}_annotated.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    cv2.namedWindow("Forest Animal Detection - Video", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_species, species_counts = detect_animals(frame)
        frame = draw_counts_on_frame(frame, species_counts)
        panel = build_info_panel(frame.shape[0], detected_species)
        combined = np.concatenate((frame, panel), axis=1)

        out.write(combined)

        cv2.imshow("Forest Animal Detection - Video", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video as {out_path}")


# -----------------------------------
# 8. Main: user chooses image/video + provides path
# -----------------------------------
if __name__ == "__main__":
    print("=== Forest Animal Detection (YOLO + OpenCV) ===")
    print("1 - Image")
    print("2 - Video")
    mode = input("Choose mode (1/2): ").strip()

    if mode == "1":
        path = input(
            f"Enter image path (leave blank for default: {DEFAULT_IMAGE_PATH}): "
        ).strip()
        if not path:
            path = DEFAULT_IMAGE_PATH
        run_on_image(path)

    elif mode == "2":
        path = input(
            f"Enter video path (leave blank for default: {DEFAULT_VIDEO_PATH}): "
        ).strip()
        if not path:
            path = DEFAULT_VIDEO_PATH
        run_on_video(path)

    else:
        print("Invalid choice. Please run again and select 1 or 2.")
