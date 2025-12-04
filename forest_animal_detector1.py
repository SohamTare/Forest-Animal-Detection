import cv2
import numpy as np
import os
import gdown
import streamlit as st

# -----------------------------------
# 0. GLOBAL CONFIG
# -----------------------------------
USE_DEER_OVERRIDE = True  # Convert cow→deer if wildlife

WEIGHTS_PATH = "yolov3.weights"
CFG_PATH = "yolov3.cfg"
NAMES_PATH = "coco.names"

WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"  # Auto-download

PANEL_WIDTH = 380
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# -----------------------------------
# 1. Auto download YOLO weights if missing (for Streamlit Cloud)
# -----------------------------------
if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("Downloading YOLO weights (~200MB)... Please wait ⏳"):
        gdown.download(WEIGHTS_URL, WEIGHTS_PATH, quiet=False)
        st.success("YOLO weights downloaded successfully!")

# -----------------------------------
# 2. Species info dictionary
# -----------------------------------
species_info = {
    "bird": {
        "common_name": "Bird",
        "scientific_name": "Aves",
        "diet": "Seeds, insects, fruits, fish",
        "habitat": "Forests, grasslands, wetlands",
        "status": "Varies by species",
        "fun_fact": "Some birds migrate thousands of kilometers every year."
    },
    "cat": {
        "common_name": "Wild/Feral Cat",
        "scientific_name": "Felis catus",
        "diet": "Carnivore",
        "habitat": "Forests, human settlements",
        "status": "Least Concern",
        "fun_fact": "Cats can rotate their ears 180 degrees."
    },
    "dog": {
        "common_name": "Wild/Feral Dog",
        "scientific_name": "Canis lupus familiaris",
        "diet": "Omnivore",
        "habitat": "Rural areas, forests",
        "status": "Domesticated species",
        "fun_fact": "Dogs sense of smell is extremely strong."
    },
    "horse": {
        "common_name": "Horse",
        "scientific_name": "Equus ferus caballus",
        "diet": "Herbivore",
        "habitat": "Grasslands and plains",
        "status": "Domesticated species",
        "fun_fact": "Horses can sleep sitting and standing!"
    },
    "sheep": {
        "common_name": "Sheep",
        "scientific_name": "Ovis aries",
        "diet": "Grass & plants",
        "habitat": "Grasslands and hills",
        "status": "Domesticated species",
        "fun_fact": "Sheep have rectangular pupils!"
    },
    "cow": {
        "common_name": "Cow",
        "scientific_name": "Bos taurus",
        "diet": "Grass & fodder",
        "habitat": "Grasslands, farms",
        "status": "Domesticated species",
        "fun_fact": "Cows form best friendships!"
    },
    "deer": {
        "common_name": "Deer",
        "scientific_name": "Cervidae",
        "diet": "Grass, leaves",
        "habitat": "Forests, meadows",
        "status": "Varies by species",
        "fun_fact": "Deer shed and re-grow antlers annually."
    },
    "elephant": {
        "common_name": "Elephant",
        "scientific_name": "Elephas/Loxodonta",
        "diet": "Leaves and fruits",
        "habitat": "Forests, savannas",
        "status": "Endangered",
        "fun_fact": "Elephants recognize themselves in mirrors!"
    },
    "bear": {
        "common_name": "Bear",
        "scientific_name": "Ursidae",
        "diet": "Omnivore",
        "habitat": "Mountains and forests",
        "status": "Varies",
        "fun_fact": "Some bears hibernate for months!"
    },
    "zebra": {
        "common_name": "Zebra",
        "scientific_name": "Equus quagga",
        "diet": "Grass",
        "habitat": "Savannas",
        "status": "Near Threatened",
        "fun_fact": "Each zebra has unique stripes like fingerprints!"
    },
    "giraffe": {
        "common_name": "Giraffe",
        "scientific_name": "Giraffa camelopardalis",
        "diet": "Leaves",
        "habitat": "Open woodlands",
        "status": "Vulnerable",
        "fun_fact": "Same number of neck vertebrae as humans!"
    }
}

# -----------------------------------
# 3. Load YOLO after weights are downloaded
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
# 4. Detector Logic
# -----------------------------------
def detect_animals(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, ids = [], [], []

    for out in outputs:
        for det in out:
            scores = det[5:]
            cid = np.argmax(scores)
            conf = scores[cid]

            label = classes[cid]
            if conf > CONF_THRESHOLD and label in animal_classes:
                cx, cy, bw, bh = (det[0]*w, det[1]*h, det[2]*w, det[3]*h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(conf))
                ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    detected = []
    counts = {}

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            lbl = classes[ids[i]]
            if USE_DEER_OVERRIDE and lbl == "cow":
                lbl = "deer"

            detected.append(lbl)
            counts[lbl] = counts.get(lbl, 0) + 1

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
            cv2.putText(frame, f"{lbl}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

    return frame, list(set(detected)), counts


# -----------------------------------
# 5. Count overlay
# -----------------------------------
def draw_counts_on_frame(frame, species_counts):
    y = 25
    for sp, cnt in species_counts.items():
        cv2.putText(frame, f"{sp}: {cnt}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,255), 2)
        y += 25
    return frame


# -----------------------------------
# 6. Right Info Panel
# -----------------------------------
def build_info_panel(h, species_list):
    panel = np.full((h, PANEL_WIDTH, 3), (30, 30, 30), np.uint8)
    y = 30
    for sp in species_list:
        info = species_info.get(sp)
        cv2.putText(panel, sp.upper(), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y += 22
        for k in ["common_name","scientific_name","diet",
                  "habitat","status","fun_fact"]:
            if info and k in info:
                cv2.putText(panel, f"{k.capitalize()}: {info[k]}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (230,230,230), 1)
                y += 18
        y += 10
    return panel
