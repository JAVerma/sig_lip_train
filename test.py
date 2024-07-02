import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
import open_clip
from glob import glob

# model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:jiviai/SigLIP-Derma")
# model = model.to("cuda")
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# tokenizer = open_clip.get_tokenizer("hf-hub:jiviai/SigLIP-Derma")

model = CLIPModel.from_pretrained("/home/ubuntu/partition/gitartha/clip/scripts/derma_recaption_more_layers_17164").to("cuda")
processor = CLIPProcessor.from_pretrained("/home/ubuntu/partition/gitartha/clip/scripts/derma_recaption_more_layers_17164")


# API endpoint
url = "http://44.221.108.182:7899/llm_zoo/med_image_text_similarity/"

# Headers
headers = {
    "accept": "application/json",
}
folder_path = "/home/ubuntu/partition/gitartha/benchmark_large_segregated/google_skin"
# folder_path = "/home/ubuntu/partition/gitartha/skin_bench"
prompts = {
    folder.strip().replace("no_pneumonia", "healthy lungs"):folder for folder in os.listdir(folder_path)
}

def process_folder_open_clip(folder_path):
    responses = {}
    for image_path in tqdm(glob(folder_path+"/*/*")):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")
        text = tokenizer(list(prompts.keys())).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).detach().cpu().numpy()
        
        pred = list(prompts.keys())[np.argmax(text_probs)]
            
        responses[image_path] = {
                "response": pred,
                "label": os.path.basename(os.path.dirname(image_path)),
            }

        # Save the responses to a JSON file
        with open(output_file, "w") as file:
            json.dump(responses, file, indent=4)

        # print(f"Responses saved to {output_file}")



###############################CHANGE HERE#####################################
# how to use this: update folder_path and prompts for respective folder and run this script

# folder_path = "/home/gitartha/jivi/fracture_data_val"
# prompts = {
#     (f"a photo of xray with fracture" if folder == "Fractured" else "a photo of xray without any visible fractures"): folder for folder in os.listdir(folder_path)
# }



# prompts = {
#     (
#         f"an xray with bone fracture"
#         if folder.strip() == "Fractured"
#         else "an xray with healthy bones"
#     ): folder
#     for folder in os.listdir(folder_path)
# }
# prompts = {
#     (f"healthy lungs in chest x-ray" if folder.strip() == "no pneumonia" else "an x-ray showing a person' chest with pneumonia"): folder for folder in os.listdir(folder_path)
# }
# prompts = {
#     (f"a chest x-ray with healthy lung" if folder.strip() == "no pneumonia" else "a chest x-ray with pneumonia"): folder for folder in os.listdir(folder_path)
# }
# prompts = {
#     (f"normal lung presented in image" if folder.strip() == "no pneumonia" else "a photo of pneumonia"): folder for folder in os.listdir(folder_path)
# } #bioclip
# prompts = {
#     f"A photo of skin lesion with {folder}":folder for folder in os.listdir(folder_path)
# }
# prompts = {
#     f"A photo of skin with {folder}": folder for folder in os.listdir(folder_path)
# }
model_name = "derma_recaption_more_layers"
###############################################################################

payload = {"text_prompt_input": json.dumps(list(prompts.keys()))}

class_names = sorted([a.strip() for a in os.listdir(folder_path)])

output_file = f"{os.path.basename(folder_path)}_model-{model_name}.json"
print(f"{prompts =}")


def call_clip(image_path):
    image = Image.open(image_path)

    inputs = processor(
        text=list(prompts.keys()), images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs.to("cuda"))
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    ).cpu().numpy()[0]  # we can take the softmax to get the label probabilities
    res = {prompt: prob for prompt, prob in zip(list(prompts.keys()), probs)}
    return res


def process_image(image_path):
    try:
        with open(image_path, "rb") as file:
            files = {
                "image_file": (os.path.basename(image_path), file, "image/png"),
                "model": (None, model),
                # 'image_url': ''
            }
            response = requests.post(url, headers=headers, data=payload, files=files)
            response.raise_for_status()
            return image_path, response.json()
    except Exception as e:
        print(e)


# List to hold futures
def process_folder(folder_path):

    responses = {}
    futures = []

    # Walk through each file in the folder and its subdirectories
    for root, _, files in os.walk(folder_path):
        for file_name in tqdm(files):
            image_path = os.path.join(root, file_name)
            result = call_clip(image_path)
            best_match_prompt = max(result.items(), key=lambda prompt: prompt[1])[0]
            pred = prompts[best_match_prompt]

            responses[image_path] = {
                "response": pred,
                "label": os.path.basename(os.path.dirname(image_path)),
            }

        # Save the responses to a JSON file
        with open(output_file, "w") as file:
            json.dump(responses, file, indent=4)



def get_count(data):
    labels = [item["label"] for item in data.values()]

    # Use Counter to count the occurrences of each label
    label_counts = dict(Counter(labels))
    return label_counts


def get_normalized_cm(conf_matrix):
    normalized_confusion_mat = np.zeros_like(conf_matrix, dtype=float)
    class_counts = np.sum(conf_matrix, axis=1)
    for i in range(len(class_names)):
        if class_counts[i] != 0:
            normalized_confusion_mat[i, :] = conf_matrix[i, :] / class_counts[i]
    return normalized_confusion_mat


def plot_cm(conf_matrix, labels, name_suffix):
    plt.figure(figsize=(len(labels) + 3, len(labels) + 3))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmax=1.0,
        vmin=0.0,
    )

    plt.xlabel("Predicted Labels", fontsize=15)
    plt.ylabel("True Labels", fontsize=15)
    plt.title("Confusion Matrix", fontsize=16)
    cm_path = f"./cm_{name_suffix}.png"
    plt.savefig(cm_path)
    print(f"confusion matrix saved to {cm_path}")
    return cm_path


def resize_image(image_path, max_size=(800, 800)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()


# def make_doc(response, image_path, name_suffix):
#     doc = Document()
#     table = doc.add_table(rows=1, cols=1)
#     table.autofit = False
#     table.columns[0].width = Inches(10)  # Image column width
#     row = table.rows[0]
#     cell = row.cells[0]
#     resized_image_data = resize_image(image_path)
#     cell.add_paragraph().add_run().add_picture(
#         BytesIO(resized_image_data), width=Inches(6)
#     )

#     table = doc.add_table(rows=1, cols=1)
#     cell = row.cells[0]
#     p = cell.add_paragraph()
#     run = p.add_run(json.dumps(response, indent=4))
#     font = run.font
#     font.size = Pt(5)
#     doc_path = f"./doc_{name_suffix}.docx"
#     doc.save(doc_path)
#     print(f"doc saved to {doc_path}")
#     return doc_path


# def convertdoc2pdf(doc_path):
#     # Path to save the PDF file
#     command = ["libreoffice", "--headless", "--convert-to", "pdf", doc_path]
#     subprocess.run(command)


def make_cm():
    with open(output_file, "r") as fp:
        preds = json.load(fp)

    # Extracting true labels and predicted labels from the dictionary
    true_labels = [data["label"].strip() for data in preds.values()]
    predicted_labels = [data["response"].strip() for data in preds.values()]
    total = len(true_labels)
    tp = [true == pred for true, pred in zip(true_labels, predicted_labels)]
    correct = tp.count(True)
    print(f"Accuracy: {correct/total}")
    # Creating a confusion matrix
    conf_matrix = get_normalized_cm(
        confusion_matrix(true_labels, predicted_labels, labels=class_names)
    )
    name_suffix = (
        f"{os.path.basename(output_file).rsplit('.', 1)[0]}_{correct/total: 0.4f}"
    )
    labels = sorted(set(true_labels + predicted_labels))
    cm_path = plot_cm(conf_matrix, labels, name_suffix)
    res = {
        **classification_report(
            true_labels, predicted_labels, labels=class_names, output_dict=True
        ),
        "count": get_count(preds),
        "prompts_used": prompts,
        "data_folder_path": folder_path,
        "cm path": os.path.abspath(cm_path),
    }
    report_path = f"./report_{name_suffix}.json"
    with open(report_path, "w") as fp:
        json.dump(res, fp, indent=4)
    print(f"classification report saved to {report_path}")
    doc_path = make_doc(res, cm_path, name_suffix)
    convertdoc2pdf(doc_path)


def main():
    process_folder(folder_path)
    make_cm()


if __name__ == "__main__":
    main()






