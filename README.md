# Satellite Visual Search & Detection System

![Few-Shot Learning](https://img.shields.io/badge/Few--Shot%20Learning-blue)
![ResNet50](https://img.shields.io/badge/ResNet--50-orange)
![Satellite Vision](https://img.shields.io/badge/Satellite%20Vision-brightgreen)

A cutting-edge few-shot learning system for satellite imagery analysis that combines **ResNet-50 feature extraction**, **vector matching**, and **OpenCV preprocessing** to detect multiple objects with minimal training examples.

## 🚀 Core Innovation: Few-Shot Learning via Feature Matching

This system implements a **metric-based few-shot learning approach** that enables object detection with only 1-5 reference images per class. Unlike traditional deep learning methods that require thousands of labeled examples, our system:

1. **Leverages Pre-trained Representations**: Uses ResNet-50 as a feature extractor, capitalizing on knowledge learned from ImageNet
2. **Creates Class Prototypes**: Generates representative feature vectors by averaging embeddings from just a few reference "chip" images
3. **Performs Feature Matching**: Compares embeddings from target image windows to class prototypes using cosine similarity
4. **Implements Metric Learning**: Learns a similarity metric in embedding space rather than raw pixel space

### Technical Architecture

```mermaid
graph TD
    A["Reference Chips\n(1-5 per class)"] --> B["Feature Extraction\nResNet-50 + Custom Layers"]
    B --> C["Class Prototypes\nAvg Embeddings"]
    D["Target Satellite Image"] --> E["Sliding Window\nMulti-Scale Search"]
    E --> F["Window Feature Extraction"]
    F --> G["Cosine Similarity\nFeature Matching"]
    C --> G
    G --> H["Threshold Detection"]
    H --> I["NMS Post-Processing"]
    I --> J["Visual Output + CSV"]

    classDef process fill:none,stroke:#000,stroke-width:1px,color:#000;
    classDef data fill:none,stroke:#000,stroke-width:1px,color:#000;
    classDef output fill:none,stroke:#000,stroke-width:1px,color:#000;

    class A,D data;
    class B,C,E,F,G,H,I process;
    class J output;



```
