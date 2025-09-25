import argparse
import numpy as np
import torch
from clip import clip
from tqdm import tqdm

from datasets.caltech101 import Caltech101
from datasets.dtd import DTD
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.ufc101 import UFC
from prompts.cupl import g_cupl
from prompts.template import g_templates


def zeroshot_classifier(classnames, g_type, model, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:

            if g_type == 'templates':
                texts = g_templates(classname)
            elif g_type == 'cupl':
                texts = g_cupl(classname)
            else:
                raise ValueError(f"Unknown g_type: {g_type}")

            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main(dataset_name, g_type, batch_size=32):
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    # Load dataset
    datasets_map = {
        'oxford_pets': OxfordPets,
        'oxford_flowers': OxfordFlowers,
        'caltech101': Caltech101,
        'dtd': DTD,
        'ufc': UFC
    }

    if dataset_name not in datasets_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    images = datasets_map[dataset_name](transform=preprocess)
    if len(images) == 0:
        raise RuntimeError(f"No images found in dataset {dataset_name}")

    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=2)

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", model.visual.input_resolution)
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)

    image_classes = images.classes
    zeroshot_weights = zeroshot_classifier(image_classes, g_type, model, device)

    # Evaluation
    top1, top5, n = 0., 0., 0.
    with torch.no_grad():
        for imgs, target in tqdm(loader, desc="Evaluating"):
            imgs, target = imgs.to(device), target.to(device)

            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += imgs.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Zero-Shot Image Classification with CLIP")
    parser.add_argument('--dataset', type=str, default='oxford_pets',
                        choices=['oxford_pets', 'oxford_flowers', 'caltech101', 'dtd', 'ufc'],
                        help='Dataset name')
    parser.add_argument('--g_type', type=str, choices=['templates', 'cupl'], default='templates',
                        help='Prompt generation type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')

    args = parser.parse_args()
    main(args.dataset, args.g_type, args.batch_size)
