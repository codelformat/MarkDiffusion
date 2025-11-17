<div align="center">

<img src="img/markdiffusion-color-1.jpg" style="width: 65%;"/>

# Une Boîte à Outils Open-Source pour le Tatouage Numérique Génératif des Modèles de Diffusion Latente

[![Homepage](https://img.shields.io/badge/Homepage-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://generative-watermark.github.io/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.10569)
[![HF Models](https://img.shields.io/badge/HF--Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Generative-Watermark-Toolkits) 

**Versions linguistiques :** [English](README.md) | [中文](README_zh.md) | [Français](README_fr.md) | [Español](README_es.md)

</div>

> 🔥 **En tant que projet récemment publié, nous accueillons les PR !** Si vous avez implémenté un algorithme de tatouage numérique LDM ou si vous êtes intéressé à en contribuer un, nous serions ravis de l'inclure dans MarkDiffusion. Rejoignez notre communauté et aidez à rendre le tatouage numérique génératif plus accessible à tous !

## Sommaire
- [Remarques](#-remarques)
- [Mises à jour](#-mises-à-jour)
- [Introduction à MarkDiffusion](#introduction-à-markdiffusion)
  - [Vue d'ensemble](#vue-densemble)
  - [Caractéristiques clés](#caractéristiques-clés)
  - [Algorithmes implémentés](#algorithmes-implémentés)
  - [Module d'évaluation](#module-dévaluation)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Comment utiliser la boîte à outils](#comment-utiliser-la-boîte-à-outils)
  - [Génération et détection de médias tatoués](#génération-et-détection-de-médias-tatoués)
  - [Visualisation des mécanismes de tatouage](#visualisation-des-mécanismes-de-tatouage)
  - [Pipelines d'évaluation](#pipelines-dévaluation)
- [Citation](#citation)

## ❗❗❗ Remarques
Au fur et à mesure que le contenu du dépôt MarkDiffusion s'enrichit et que sa taille augmente, nous avons créé un dépôt de stockage de modèles sur Hugging Face appelé [Generative-Watermark-Toolkits](https://huggingface.co/Generative-Watermark-Toolkits) pour faciliter l'utilisation. Ce dépôt contient divers modèles par défaut pour les algorithmes de tatouage numérique qui impliquent des modèles auto-entraînés. Nous avons supprimé les poids des modèles des dossiers `ckpts/` correspondants de ces algorithmes de tatouage dans le dépôt principal. **Lors de l'utilisation du code, veuillez d'abord télécharger les modèles correspondants depuis le dépôt Hugging Face selon les chemins de configuration et les enregistrer dans le répertoire `ckpts/` avant d'exécuter le code.**

## 🔥 Mises à jour
🎯 **(2025.10.10)** Ajout des outils d'attaque d'image *Mask, Overlay, AdaptiveNoiseInjection*, merci à Zheyu Fu pour sa PR !

🎯 **(2025.10.09)** Ajout des outils d'attaque vidéo *VideoCodecAttack, FrameRateAdapter, FrameInterpolationAttack*, merci à Luyang Si pour sa PR !

🎯 **(2025.10.08)** Ajout des analyseurs de qualité d'image *SSIM, BRISQUE, VIF, FSIM*, merci à Huan Wang pour sa PR !

✨ **(2025.10.07)** Ajout de la méthode de tatouage [SFW](https://arxiv.org/pdf/2509.07647), merci à Huan Wang pour sa PR !

✨ **(2025.10.07)** Ajout de la méthode de tatouage [VideoMark](https://arxiv.org/abs/2504.16359), merci à Hanqian Li pour sa PR !

✨ **(2025.9.29)** Ajout de la méthode de tatouage [GaussMarker](https://arxiv.org/abs/2506.11444), merci à Luyang Si pour sa PR !

## Introduction à MarkDiffusion

### Vue d'ensemble

MarkDiffusion est une boîte à outils Python open-source pour le tatouage numérique génératif des modèles de diffusion latente. Alors que l'utilisation des modèles génératifs basés sur la diffusion s'étend, garantir l'authenticité et l'origine des médias générés devient crucial. MarkDiffusion simplifie l'accès, la compréhension et l'évaluation des technologies de tatouage numérique, les rendant accessibles tant aux chercheurs qu'à la communauté au sens large. *Remarque : si vous êtes intéressé par le tatouage LLM (tatouage de texte), veuillez vous référer à la boîte à outils [MarkLLM](https://github.com/THU-BPM/MarkLLM) de notre groupe.*

La boîte à outils comprend trois composants clés : un cadre d'implémentation unifié pour des intégrations rationalisées d'algorithmes de tatouage et des interfaces conviviales ; une suite de visualisation de mécanismes qui présente intuitivement les motifs de tatouage ajoutés et extraits pour aider à la compréhension du public ; et un module d'évaluation complet offrant des implémentations standard de 24 outils couvrant trois aspects essentiels — détectabilité, robustesse et qualité de sortie, plus 8 pipelines d'évaluation automatisés.

<img src="img/fig1_overview.png" alt="MarkDiffusion Overview" style="zoom:50%;" />

### Caractéristiques clés

- **Cadre d'implémentation unifié :** MarkDiffusion fournit une architecture modulaire prenant en charge huit algorithmes de tatouage d'image/vidéo génératifs de pointe pour les LDM.

- **Support algorithmique complet :** Implémente actuellement 8 algorithmes de tatouage de deux catégories principales : méthodes basées sur les motifs (Tree-Ring, Ring-ID, ROBIN, WIND) et méthodes basées sur les clés (Gaussian-Shading, PRC, SEAL, VideoShield).

- **Solutions de visualisation :** La boîte à outils comprend des outils de visualisation personnalisés qui permettent des vues claires et perspicaces sur le fonctionnement des différents algorithmes de tatouage dans divers scénarios. Ces visualisations aident à démystifier les mécanismes des algorithmes, les rendant plus compréhensibles pour les utilisateurs.

- **Module d'évaluation :** Avec 20 outils d'évaluation couvrant la détectabilité, la robustesse et l'impact sur la qualité de sortie, MarkDiffusion fournit des capacités d'évaluation complètes. Il comprend 5 pipelines d'évaluation automatisés : Pipeline de détection de tatouage, Pipeline d'analyse de qualité d'image, Pipeline d'analyse de qualité vidéo et outils d'évaluation de robustesse spécialisés.

### Algorithmes implémentés

| **Algorithme** | **Catégorie** | **Cible** | **Référence** |
|---------------|-------------|------------|---------------|
| Tree-Ring | Motif | Image | [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) |
| Ring-ID | Motif | Image | [RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](https://arxiv.org/abs/2404.14055) |
| ROBIN | Motif | Image | [ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization](https://arxiv.org/abs/2411.03862) |
| WIND | Motif | Image | [Hidden in the Noise: Two-Stage Robust Watermarking for Images](https://arxiv.org/abs/2412.04653) |
| SFW | Motif | Image | [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://arxiv.org/abs/2509.07647) |
| Gaussian-Shading | Clé | Image | [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) |
| GaussMarker | Clé | Image | [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](https://arxiv.org/abs/2506.11444) |
| PRC | Clé | Image | [An undetectable watermark for generative image models](https://arxiv.org/abs/2410.07369) |
| SEAL | Clé | Image | [SEAL: Semantic Aware Image Watermarking](https://arxiv.org/abs/2503.12172) |
| VideoShield | Clé | Vidéo | [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](https://arxiv.org/abs/2501.14195) |
| VideoMark | Clé | Vidéo | [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359) |

### Module d'évaluation
#### Pipelines d'évaluation

MarkDiffusion prend en charge huit pipelines, deux pour la détection (WatermarkedMediaDetectionPipeline et UnWatermarkedMediaDetectionPipeline), et six pour l'analyse de qualité. Le tableau ci-dessous détaille les pipelines d'analyse de qualité.

| **Pipeline d'analyse de qualité** | **Type d'entrée** | **Données requises** | **Métriques applicables** |  
| --- | --- | --- | --- |
| DirectImageQualityAnalysisPipeline | Image unique | Image tatouée/non tatouée générée | Métriques pour l'évaluation d'image unique | 
| ReferencedImageQualityAnalysisPipeline | Image + contenu de référence | Image tatouée/non tatouée générée + image/texte de référence | Métriques nécessitant un calcul entre image unique et contenu de référence (texte/image) | 
| GroupImageQualityAnalysisPipeline | Ensemble d'images (+ ensemble d'images de référence) | Ensemble d'images tatouées/non tatouées générées (+ ensemble d'images de référence) | Métriques nécessitant un calcul sur des ensembles d'images | 
| RepeatImageQualityAnalysisPipeline | Ensemble d'images | Ensemble d'images tatouées/non tatouées générées de manière répétée | Métriques pour évaluer des ensembles d'images générées de manière répétée | 
| ComparedImageQualityAnalysisPipeline | Deux images pour comparaison | Images tatouées et non tatouées générées | Métriques mesurant les différences entre deux images | 
| DirectVideoQualityAnalysisPipeline | Vidéo unique | Ensemble de cadres vidéo générés | Métriques pour l'évaluation vidéo globale |

#### Outils d'évaluation

| **Nom de l'outil** | **Catégorie d'évaluation** | **Description de la fonction** | **Métriques de sortie** |
| --- | --- | --- | --- |
| FundamentalSuccessRateCalculator | Détectabilité | Calculer les métriques de classification pour la détection de tatouage à seuil fixe | Diverses métriques de classification |
| DynamicThresholdSuccessRateCalculator | Détectabilité | Calculer les métriques de classification pour la détection de tatouage à seuil dynamique | Diverses métriques de classification |
| **Outils d'attaque d'image** | | | |
| Rotation | Robustesse (Image) | Attaque par rotation d'image, testant la résistance du tatouage aux transformations de rotation | Images/cadres pivotés |
| CrSc (Crop & Scale) | Robustesse (Image) | Attaque par recadrage et mise à l'échelle, évaluant la robustesse du tatouage aux changements de taille | Images/cadres recadrés/redimensionnés |
| GaussianNoise | Robustesse (Image) | Attaque par bruit gaussien, testant la résistance du tatouage aux interférences de bruit | Images/cadres corrompus par le bruit |
| GaussianBlurring | Robustesse (Image) | Attaque par flou gaussien, évaluant la résistance du tatouage au traitement de flou | Images/cadres flous |
| JPEGCompression | Robustesse (Image) | Attaque par compression JPEG, testant la robustesse du tatouage à la compression avec perte | Images/cadres compressés |
| Brightness | Robustesse (Image) | Attaque par ajustement de luminosité, évaluant la résistance du tatouage aux changements de luminosité | Images/cadres modifiés en luminosité |
| Mask | Robustesse (Image) | Attaque par masquage d'image, testant la résistance du tatouage à l'occlusion partielle par des rectangles noirs aléatoires | Images/cadres masqués |
| Overlay | Robustesse (Image) | Attaque par superposition d'image, testant la résistance du tatouage aux traits et annotations de type graffiti | Images/cadres superposés |
| AdaptiveNoiseInjection | Robustesse (Image) | Attaque par injection de bruit adaptatif, testant la résistance du tatouage au bruit adaptatif au contenu (Gaussien/Sel-poivre/Poisson/Speckle) | Images/cadres bruyants avec bruit adaptatif |
| **Outils d'attaque vidéo** | | | |
| MPEG4Compression | Robustesse (Vidéo) | Attaque par compression vidéo MPEG-4, testant la robustesse du tatouage vidéo à la compression | Cadres vidéo compressés |
| FrameAverage | Robustesse (Vidéo) | Attaque par moyennage de cadres, détruisant les tatouages par moyennage inter-cadres | Cadres vidéo moyennés |
| FrameSwap | Robustesse (Vidéo) | Attaque par échange de cadres, testant la robustesse en changeant les séquences de cadres | Cadres vidéo échangés |
| VideoCodecAttack | Robustesse (Vidéo) | Attaque par ré-encodage de codec simulant le transcodage de plateforme (H.264/H.265/VP9/AV1) | Cadres vidéo ré-encodés |
| FrameRateAdapter | Robustesse (Vidéo) | Attaque par conversion de fréquence d'images qui rééchantillonne les cadres tout en préservant la durée | Séquence de cadres rééchantillonnée |
| FrameInterpolationAttack | Robustesse (Vidéo) | Attaque par interpolation de cadres insérant des cadres mélangés pour modifier la densité temporelle | Cadres vidéo interpolés |
| **Analyseurs de qualité d'image** | | | |
| InceptionScoreCalculator | Qualité (Image) | Évaluer la qualité et la diversité des images générées | Score IS |
| FIDCalculator | Qualité (Image) | Distance d'Inception de Fréchet, mesurant la différence de distribution entre images générées et réelles | Valeur FID |
| LPIPSAnalyzer | Qualité (Image) | Similarité de patch d'image perceptuelle apprise, évaluant la qualité perceptuelle | Distance LPIPS |
| CLIPScoreCalculator | Qualité (Image) | Évaluation de cohérence texte-image basée sur CLIP | Score de similarité CLIP |
| PSNRAnalyzer | Qualité (Image) | Rapport signal sur bruit de crête, mesurant la distorsion d'image | Valeur PSNR (dB) |
| NIQECalculator | Qualité (Image) | Évaluateur de qualité d'image naturelle, évaluation de qualité sans référence | Score NIQE |
| SSIMAnalyzer | Qualité (Image) | Indice de similarité structurelle entre deux images | Valeur SSIM |
| BRISQUEAnalyzer | Qualité (Image) | Évaluateur de qualité spatiale d'image aveugle/sans référence, évaluant la qualité perceptuelle d'une image sans nécessiter de référence | Score BRISQUE |
| VIFAnalyzer | Qualité (Image) | Analyseur de fidélité d'information visuelle, comparant une image déformée avec une image de référence pour quantifier la quantité d'information visuelle préservée | Valeur VIF |
| FSIMAnalyzer | Qualité (Image) | Analyseur d'indice de similarité de caractéristiques, comparant la similarité structurelle entre deux images basée sur la congruence de phase et la magnitude du gradient | Valeur FSIM |
| **Analyseurs de qualité vidéo** | | | |
| SubjectConsistencyAnalyzer | Qualité (Vidéo) | Évaluer la cohérence des objets sujets dans la vidéo | Score de cohérence du sujet |
| BackgroundConsistencyAnalyzer | Qualité (Vidéo) | Évaluer la cohérence et la stabilité de l'arrière-plan dans la vidéo | Score de cohérence de l'arrière-plan |
| MotionSmoothnessAnalyzer | Qualité (Vidéo) | Évaluer la fluidité du mouvement vidéo | Métrique de fluidité du mouvement |
| DynamicDegreeAnalyzer | Qualité (Vidéo) | Mesurer le niveau dynamique et l'amplitude de changement dans la vidéo | Valeur de degré dynamique |
| ImagingQualityAnalyzer | Qualité (Vidéo) | Évaluation complète de la qualité d'imagerie vidéo | Score de qualité d'imagerie |

## Installation

### Configuration de l'environnement

- Python 3.10+
- PyTorch
- Installer les dépendances :

```bash
pip install -r requirements.txt
```

*Remarque :* Certains algorithmes peuvent nécessiter des étapes de configuration supplémentaires. Veuillez vous référer à la documentation des algorithmes individuels pour les exigences spécifiques.

## Démarrage rapide

Voici un exemple simple pour vous aider à démarrer avec MarkDiffusion :

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Configuration du périphérique
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration du pipeline de diffusion
scheduler = DPMSolverMultistepScheduler.from_pretrained("model_path", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("model_path", scheduler=scheduler).to(device)
diffusion_config = DiffusionConfig(
    scheduler=scheduler,
    pipe=pipe,
    device=device,
    image_size=(512, 512),
    num_inference_steps=50,
    guidance_scale=7.5,
    gen_seed=42,
    inversion_type="ddim"
)

# Charger l'algorithme de tatouage
watermark = AutoWatermark.load('TR', 
                              algorithm_config='config/TR.json',
                              diffusion_config=diffusion_config)

# Générer un média tatoué
prompt = "A beautiful sunset over the ocean"
watermarked_image = watermark.generate_watermarked_media(prompt)

# Détecter le tatouage
detection_result = watermark.detect_watermark_in_media(watermarked_image)
print(f"Watermark detected: {detection_result}")
```

## Comment utiliser la boîte à outils

Nous fournissons de nombreux exemples dans `MarkDiffusion_demo.ipynb`.

### Génération et détection de médias tatoués

#### Cas de génération et de détection de médias tatoués

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig

# Charger l'algorithme de tatouage
mywatermark = AutoWatermark.load(
    'GS',
    algorithm_config=f'config/GS.json',
    diffusion_config=diffusion_config
)

# Générer une image tatouée
watermarked_image = mywatermark.generate_watermarked_media(
    input_data="A beautiful landscape with a river and mountains"
)

# Visualiser l'image tatouée
watermarked_image.show()

# Détecter le tatouage
detection_result = mywatermark.detect_watermark_in_media(watermarked_image)
print(detection_result)
```

### Visualisation des mécanismes de tatouage

La boîte à outils comprend des outils de visualisation personnalisés qui permettent des vues claires et perspicaces sur le fonctionnement des différents algorithmes de tatouage dans divers scénarios. Ces visualisations aident à démystifier les mécanismes des algorithmes, les rendant plus compréhensibles pour les utilisateurs.

<img src="img/fig2_visualization_mechanism.png" alt="Watermarking Mechanism Visualization" style="zoom:40%;" />

#### Cas de visualisation du mécanisme de tatouage

```python
from visualize.auto_visualization import AutoVisualizer

# Obtenir les données pour la visualisation
data_for_visualization = mywatermark.get_data_for_visualize(watermarked_image)

# Charger le visualiseur
visualizer = AutoVisualizer.load('GS', 
                                data_for_visualization=data_for_visualization)

# Dessiner des diagrammes sur le canevas Matplotlib
fig = visualizer.visualize(rows=2, cols=2, 
                          methods=['draw_watermark_bits', 
                                  'draw_reconstructed_watermark_bits', 
                                  'draw_inverted_latents', 
                                  'draw_inverted_latents_fft'])
```

### Pipelines d'évaluation

#### Cas d'évaluation

1. **Pipeline de détection de tatouage**

```python
from evaluation.dataset import StableDiffusionPromptsDataset
from evaluation.pipelines.detection import (
    WatermarkedMediaDetectionPipeline, 
    UnWatermarkedMediaDetectionPipeline, 
    DetectionPipelineReturnType
)
from evaluation.tools.image_editor import JPEGCompression
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator

# Jeu de données
my_dataset = StableDiffusionPromptsDataset(max_samples=200)

# Configurer les pipelines de détection
pipeline1 = WatermarkedMediaDetectionPipeline(
    dataset=my_dataset,
    media_editor_list=[JPEGCompression(quality=60)],
    show_progress=True, 
    return_type=DetectionPipelineReturnType.SCORES
)

pipeline2 = UnWatermarkedMediaDetectionPipeline(
    dataset=my_dataset,
    media_editor_list=[],
    show_progress=True, 
    return_type=DetectionPipelineReturnType.SCORES
)

# Configurer les paramètres de détection
detection_kwargs = {
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
}

# Calculer les taux de réussite
calculator = DynamicThresholdSuccessRateCalculator(
    labels=labels, 
    rule=rules,
    target_fpr=target_fpr
)

results = calculator.calculate(
    pipeline1.evaluate(my_watermark, detection_kwargs=detection_kwargs),
    pipeline2.evaluate(my_watermark, detection_kwargs=detection_kwargs)
)
print(results)
```

2. **Pipeline d'analyse de qualité d'image**

```python
from evaluation.dataset import StableDiffusionPromptsDataset, MSCOCODataset
from evaluation.pipelines.image_quality_analysis import (
    DirectImageQualityAnalysisPipeline,
    ReferencedImageQualityAnalysisPipeline,
    GroupImageQualityAnalysisPipeline,
    RepeatImageQualityAnalysisPipeline,
    ComparedImageQualityAnalysisPipeline,
    QualityPipelineReturnType
)
from evaluation.tools.image_quality_analyzer import (
    NIQECalculator, CLIPScoreCalculator, FIDCalculator, 
    InceptionScoreCalculator, LPIPSAnalyzer, PSNRAnalyzer
)

# Exemples de différentes métriques de qualité :

# NIQE (Évaluateur de qualité d'image naturelle)
if metric == 'NIQE':
    my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
    pipeline = DirectImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[NIQECalculator()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# Score CLIP
elif metric == 'CLIP':
    my_dataset = MSCOCODataset(max_samples=max_samples)
    pipeline = ReferencedImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[CLIPScoreCalculator()],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# FID (Distance d'Inception de Fréchet)
elif metric == 'FID':
    my_dataset = MSCOCODataset(max_samples=max_samples)
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[FIDCalculator()],
        unwatermarked_image_source='generated',
        reference_image_source='natural',
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# IS (Score Inception)
elif metric == 'IS':
    my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
    pipeline = GroupImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[InceptionScoreCalculator()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# LPIPS (Similarité de patch d'image perceptuelle apprise)
elif metric == 'LPIPS':
    my_dataset = StableDiffusionPromptsDataset(max_samples=10)
    pipeline = RepeatImageQualityAnalysisPipeline(
        dataset=my_dataset,
        prompt_per_image=20,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[LPIPSAnalyzer()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# PSNR (Rapport signal sur bruit de crête)
elif metric == 'PSNR':
    my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
    pipeline = ComparedImageQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_image_editor_list=[],
        unwatermarked_image_editor_list=[],
        analyzers=[PSNRAnalyzer()],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )

# Charger le tatouage et évaluer
my_watermark = AutoWatermark.load(
    f'{algorithm_name}',
    algorithm_config=f'config/{algorithm_name}.json',
    diffusion_config=diffusion_config
)

print(pipeline.evaluate(my_watermark))
```

3. **Pipeline d'analyse de qualité vidéo**

```python
from evaluation.dataset import VBenchDataset
from evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline
from evaluation.tools.video_quality_analyzer import (
    SubjectConsistencyAnalyzer,
    MotionSmoothnessAnalyzer,
    DynamicDegreeAnalyzer,
    BackgroundConsistencyAnalyzer,
    ImagingQualityAnalyzer
)

# Charger le jeu de données VBench
my_dataset = VBenchDataset(max_samples=200, dimension=dimension)

# Initialiser l'analyseur en fonction de la métrique
if metric == 'subject_consistency':
    analyzer = SubjectConsistencyAnalyzer(device=device)
elif metric == 'motion_smoothness':
    analyzer = MotionSmoothnessAnalyzer(device=device)
elif metric == 'dynamic_degree':
    analyzer = DynamicDegreeAnalyzer(device=device)
elif metric == 'background_consistency':
    analyzer = BackgroundConsistencyAnalyzer(device=device)
elif metric == 'imaging_quality':
    analyzer = ImagingQualityAnalyzer(device=device)
else:
    raise ValueError(f'Invalid metric: {metric}. Supported metrics: 
                    subject_consistency, motion_smoothness, dynamic_degree,
                    background_consistency, imaging_quality')

# Créer le pipeline d'analyse de qualité vidéo
pipeline = DirectVideoQualityAnalysisPipeline(
    dataset=my_dataset,
    watermarked_video_editor_list=[],
    unwatermarked_video_editor_list=[],
    watermarked_frame_editor_list=[],
    unwatermarked_frame_editor_list=[],
    analyzers=[analyzer],
    show_progress=True,
    return_type=QualityPipelineReturnType.MEAN_SCORES
)

print(pipeline.evaluate(my_watermark))
```

## Citation
```
@article{pan2025markdiffusion,
  title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
  author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
  journal={arXiv preprint arXiv:2509.10569},
  year={2025}
}
```

