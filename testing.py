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
from transformers import (AutoImageProcessor, AutoModel, AutoProcessor,
                          AutoTokenizer)

# model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:jiviai/SigLIP-Derma")
# model = model.to("cuda")
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# tokenizer = open_clip.get_tokenizer("hf-hub:jiviai/SigLIP-Derma")

# model = CLIPModel.from_pretrained("/home/ubuntu/partition/gitartha/clip/scripts/rsna_refined_captions_2404").to("cuda")
# processor = CLIPProcessor.from_pretrained("/home/ubuntu/partition/gitartha/clip/scripts/rsna_refined_captions_2404")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
auto_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
for i in tqdm(range(1),total=1):
    i=0
    if i==10:
        print('original weights')
    else:
        model_dict=torch.load(f'/home/ubuntu/sig_lip_train/finetuned_weights/epoch_{i}.pth')
        model.load_state_dict(model_dict)
    model.to('cuda')
    folder_path = "/home/ubuntu/sig_lip_train/rsna_test"
    prompts_old = {
        folder.strip().replace("no_pneumonia", "photo of chest x-ray with healthy lungs"):folder for folder in os.listdir(folder_path)
    }
    prompts={}
    for k , v in prompts_old.items():
        if v=='pneumonia':
            prompts['photo of chest x-ray with pnuemonia']=v
        else:
            prompts[k]=v
    print(prompts)
    # break
    def process_folder_open_clip(folder_path):
        responses = {}
        for image_path in tqdm(glob(folder_path+"/*/*")):
            image = processor(Image.open(image_path)).unsqueeze(0).to("cuda")
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

    model_name = f"rsna_refined_siglip_{i}"
    ###############################################################################

    payload = {"text_prompt_input": json.dumps(list(prompts.keys()))}

    class_names = sorted([a.strip() for a in os.listdir(folder_path)])

    output_file = f"{os.path.basename(folder_path)}_model-{model_name}.json"
    print(f"{prompts =}")


    def call_clip(image_path):
        image = Image.open(image_path)
        image=image.convert('RGB')
        inputs = auto_processor(
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
            # vmax=1.0,
            # vmin=0.0,
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
        # conf_matrix = get_normalized_cm(
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_names)
        # )
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
        # doc_path = make_doc(res, cm_path, name_suffix)
        # convertdoc2pdf(doc_path)


    def main():
        process_folder(folder_path)
        make_cm()


    if __name__ == "__main__":
        main()






