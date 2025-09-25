# Prompt-tuning-for-language-vision-models-zero-shot-CLIP-
Zero-Shot Image Classification with CLIP
This repository provides a zero-shot image classification framework using OpenAI’s CLIP model. It allows classifying images across multiple benchmark datasets without training a classifier, using either template-based prompts or AI-generated prompts (CuPL).
Features
Supports multiple datasets:
Caltech101 – 101 object categories
DTD – Describable Textures Dataset
Oxford Flowers – 102 flower categories
Oxford Pets – Oxford-IIIT Pet Dataset
UCF101 – Midframes from UCF-101 videos
Flexible prompt generation:
Template-based prompts (g_templates)
AI-generated prompts (g_cupl)
Computes Top-1 and Top-5 accuracy
Modular PyTorch dataset loaders with automatic download and extraction
