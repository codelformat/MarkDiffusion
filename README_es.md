<div align="center">

<img src="img/markdiffusion-color-1.jpg" style="width: 65%;"/>

# Un Kit de Herramientas de Código Abierto para Marcas de Agua Generativas de Modelos de Difusión Latente

[![Homepage](https://img.shields.io/badge/Homepage-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://generative-watermark.github.io/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.10569)
[![HF Models](https://img.shields.io/badge/HF--Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Generative-Watermark-Toolkits) 

**Versiones de idioma:** [English](README.md) | [中文](README_zh.md) | [Français](README_fr.md) | [Español](README_es.md)

</div>

> 🔥 **¡Como un proyecto recién lanzado, damos la bienvenida a PRs!** Si has implementado un algoritmo de marcas de agua LDM o estás interesado en contribuir con uno, nos encantaría incluirlo en MarkDiffusion. ¡Únete a nuestra comunidad y ayuda a hacer las marcas de agua generativas más accesibles para todos!

## Contenidos
- [Notas](#-notas)
- [Actualizaciones](#-actualizaciones)
- [Introducción a MarkDiffusion](#introducción-a-markdiffusion)
  - [Descripción general](#descripción-general)
  - [Características clave](#características-clave)
  - [Algoritmos implementados](#algoritmos-implementados)
  - [Módulo de evaluación](#módulo-de-evaluación)
- [Instalación](#instalación)
- [Inicio rápido](#inicio-rápido)
- [Cómo usar el kit de herramientas](#cómo-usar-el-kit-de-herramientas)
  - [Generación y detección de medios con marcas de agua](#generación-y-detección-de-medios-con-marcas-de-agua)
  - [Visualización de mecanismos de marcas de agua](#visualización-de-mecanismos-de-marcas-de-agua)
  - [Pipelines de evaluación](#pipelines-de-evaluación)
- [Citación](#citación)

## ❗❗❗ Notas
A medida que el contenido del repositorio MarkDiffusion se vuelve cada vez más rico y su tamaño crece, hemos creado un repositorio de almacenamiento de modelos en Hugging Face llamado [Generative-Watermark-Toolkits](https://huggingface.co/Generative-Watermark-Toolkits) para facilitar su uso. Este repositorio contiene varios modelos predeterminados para algoritmos de marcas de agua que involucran modelos auto-entrenados. Hemos eliminado los pesos de los modelos de las carpetas `ckpts/` correspondientes de estos algoritmos de marcas de agua en el repositorio principal. **Al usar el código, primero descarga los modelos correspondientes del repositorio de Hugging Face según las rutas de configuración y guárdalos en el directorio `ckpts/` antes de ejecutar el código.**

## 🔥 Actualizaciones
🎯 **(2025.10.10)** Agregadas herramientas de ataque de imagen *Mask, Overlay, AdaptiveNoiseInjection*, ¡gracias a Zheyu Fu por su PR!

🎯 **(2025.10.09)** Agregadas herramientas de ataque de video *VideoCodecAttack, FrameRateAdapter, FrameInterpolationAttack*, ¡gracias a Luyang Si por su PR!

🎯 **(2025.10.08)** Agregados analizadores de calidad de imagen *SSIM, BRISQUE, VIF, FSIM*, ¡gracias a Huan Wang por su PR!

✨ **(2025.10.07)** Agregado el método de marca de agua [SFW](https://arxiv.org/pdf/2509.07647), ¡gracias a Huan Wang por su PR!

✨ **(2025.10.07)** Agregado el método de marca de agua [VideoMark](https://arxiv.org/abs/2504.16359), ¡gracias a Hanqian Li por su PR!

✨ **(2025.9.29)** Agregado el método de marca de agua [GaussMarker](https://arxiv.org/abs/2506.11444), ¡gracias a Luyang Si por su PR!

## Introducción a MarkDiffusion

### Descripción general

MarkDiffusion es un kit de herramientas de Python de código abierto para marcas de agua generativas de modelos de difusión latente. A medida que se expande el uso de modelos generativos basados en difusión, garantizar la autenticidad y el origen de los medios generados se vuelve crítico. MarkDiffusion simplifica el acceso, la comprensión y la evaluación de tecnologías de marcas de agua, haciéndolo accesible tanto para investigadores como para la comunidad en general. *Nota: si estás interesado en marcas de agua LLM (marca de agua de texto), consulta el kit de herramientas [MarkLLM](https://github.com/THU-BPM/MarkLLM) de nuestro grupo.*

El kit de herramientas comprende tres componentes clave: un marco de implementación unificado para integraciones simplificadas de algoritmos de marcas de agua e interfaces fáciles de usar; un conjunto de visualización de mecanismos que muestra intuitivamente los patrones de marcas de agua agregados y extraídos para ayudar a la comprensión pública; y un módulo de evaluación integral que ofrece implementaciones estándar de 24 herramientas en tres aspectos esenciales: detectabilidad, robustez y calidad de salida, además de 8 pipelines de evaluación automatizados.

<img src="img/fig1_overview.png" alt="MarkDiffusion Overview" style="zoom:50%;" />

### Características clave

- **Marco de implementación unificado:** MarkDiffusion proporciona una arquitectura modular que admite ocho algoritmos de marcas de agua generativas de imagen/video de última generación para LDMs.

- **Soporte integral de algoritmos:** Actualmente implementa 8 algoritmos de marcas de agua de dos categorías principales: métodos basados en patrones (Tree-Ring, Ring-ID, ROBIN, WIND) y métodos basados en claves (Gaussian-Shading, PRC, SEAL, VideoShield).

- **Soluciones de visualización:** El kit de herramientas incluye herramientas de visualización personalizadas que permiten vistas claras y perspicaces sobre cómo operan los diferentes algoritmos de marcas de agua en varios escenarios. Estas visualizaciones ayudan a desmitificar los mecanismos de los algoritmos, haciéndolos más comprensibles para los usuarios.

- **Módulo de evaluación:** Con 20 herramientas de evaluación que cubren detectabilidad, robustez e impacto en la calidad de salida, MarkDiffusion proporciona capacidades de evaluación integral. Cuenta con 5 pipelines de evaluación automatizados: Pipeline de detección de marcas de agua, Pipeline de análisis de calidad de imagen, Pipeline de análisis de calidad de video y herramientas especializadas de evaluación de robustez.

### Algoritmos implementados

| **Algoritmo** | **Categoría** | **Objetivo** | **Referencia** |
|---------------|-------------|------------|---------------|
| Tree-Ring | Patrón | Imagen | [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) |
| Ring-ID | Patrón | Imagen | [RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](https://arxiv.org/abs/2404.14055) |
| ROBIN | Patrón | Imagen | [ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization](https://arxiv.org/abs/2411.03862) |
| WIND | Patrón | Imagen | [Hidden in the Noise: Two-Stage Robust Watermarking for Images](https://arxiv.org/abs/2412.04653) |
| SFW | Patrón | Imagen | [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://arxiv.org/abs/2509.07647) |
| Gaussian-Shading | Clave | Imagen | [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) |
| GaussMarker | Clave | Imagen | [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](https://arxiv.org/abs/2506.11444) |
| PRC | Clave | Imagen | [An undetectable watermark for generative image models](https://arxiv.org/abs/2410.07369) |
| SEAL | Clave | Imagen | [SEAL: Semantic Aware Image Watermarking](https://arxiv.org/abs/2503.12172) |
| VideoShield | Clave | Video | [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](https://arxiv.org/abs/2501.14195) |
| VideoMark | Clave | Video | [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359) |

### Módulo de evaluación
#### Pipelines de evaluación

MarkDiffusion admite ocho pipelines, dos para detección (WatermarkedMediaDetectionPipeline y UnWatermarkedMediaDetectionPipeline), y seis para análisis de calidad. La tabla a continuación detalla los pipelines de análisis de calidad.

| **Pipeline de análisis de calidad** | **Tipo de entrada** | **Datos requeridos** | **Métricas aplicables** |  
| --- | --- | --- | --- |
| DirectImageQualityAnalysisPipeline | Imagen única | Imagen generada con/sin marca de agua | Métricas para evaluación de imagen única | 
| ReferencedImageQualityAnalysisPipeline | Imagen + contenido de referencia | Imagen generada con/sin marca de agua + imagen/texto de referencia | Métricas que requieren cálculo entre imagen única y contenido de referencia (texto/imagen) | 
| GroupImageQualityAnalysisPipeline | Conjunto de imágenes (+ conjunto de imágenes de referencia) | Conjunto de imágenes generadas con/sin marca de agua (+ conjunto de imágenes de referencia) | Métricas que requieren cálculo en conjuntos de imágenes | 
| RepeatImageQualityAnalysisPipeline | Conjunto de imágenes | Conjunto de imágenes generadas repetidamente con/sin marca de agua | Métricas para evaluar conjuntos de imágenes generadas repetidamente | 
| ComparedImageQualityAnalysisPipeline | Dos imágenes para comparación | Imágenes generadas con y sin marca de agua | Métricas que miden diferencias entre dos imágenes | 
| DirectVideoQualityAnalysisPipeline | Video único | Conjunto de fotogramas de video generados | Métricas para evaluación general de video |

#### Herramientas de evaluación

| **Nombre de la herramienta** | **Categoría de evaluación** | **Descripción de la función** | **Métricas de salida** |
| --- | --- | --- | --- |
| FundamentalSuccessRateCalculator | Detectabilidad | Calcular métricas de clasificación para detección de marca de agua con umbral fijo | Varias métricas de clasificación |
| DynamicThresholdSuccessRateCalculator | Detectabilidad | Calcular métricas de clasificación para detección de marca de agua con umbral dinámico | Varias métricas de clasificación |
| **Herramientas de ataque de imagen** | | | |
| Rotation | Robustez (Imagen) | Ataque de rotación de imagen, probando la resistencia de la marca de agua a transformaciones de rotación | Imágenes/fotogramas rotados |
| CrSc (Crop & Scale) | Robustez (Imagen) | Ataque de recorte y escalado, evaluando la robustez de la marca de agua a cambios de tamaño | Imágenes/fotogramas recortados/escalados |
| GaussianNoise | Robustez (Imagen) | Ataque de ruido gaussiano, probando la resistencia de la marca de agua a interferencias de ruido | Imágenes/fotogramas corrompidos por ruido |
| GaussianBlurring | Robustez (Imagen) | Ataque de desenfoque gaussiano, evaluando la resistencia de la marca de agua al procesamiento de desenfoque | Imágenes/fotogramas desenfocados |
| JPEGCompression | Robustez (Imagen) | Ataque de compresión JPEG, probando la robustez de la marca de agua a la compresión con pérdida | Imágenes/fotogramas comprimidos |
| Brightness | Robustez (Imagen) | Ataque de ajuste de brillo, evaluando la resistencia de la marca de agua a cambios de brillo | Imágenes/fotogramas modificados en brillo |
| Mask | Robustez (Imagen) | Ataque de enmascaramiento de imagen, probando la resistencia de la marca de agua a la oclusión parcial por rectángulos negros aleatorios | Imágenes/fotogramas enmascarados |
| Overlay | Robustez (Imagen) | Ataque de superposición de imagen, probando la resistencia de la marca de agua a trazos y anotaciones estilo grafiti | Imágenes/fotogramas superpuestos |
| AdaptiveNoiseInjection | Robustez (Imagen) | Ataque de inyección de ruido adaptativo, probando la resistencia de la marca de agua al ruido consciente del contenido (Gaussiano/Sal-pimienta/Poisson/Moteado) | Imágenes/fotogramas ruidosos con ruido adaptativo |
| **Herramientas de ataque de video** | | | |
| MPEG4Compression | Robustez (Video) | Ataque de compresión de video MPEG-4, probando la robustez de compresión de marca de agua de video | Fotogramas de video comprimidos |
| FrameAverage | Robustez (Video) | Ataque de promedio de fotogramas, destruyendo marcas de agua a través del promedio entre fotogramas | Fotogramas de video promediados |
| FrameSwap | Robustez (Video) | Ataque de intercambio de fotogramas, probando la robustez cambiando secuencias de fotogramas | Fotogramas de video intercambiados |
| VideoCodecAttack | Robustez (Video) | Ataque de recodificación de códec simulando transcodificación de plataforma (H.264/H.265/VP9/AV1) | Fotogramas de video recodificados |
| FrameRateAdapter | Robustez (Video) | Ataque de conversión de velocidad de fotogramas que remuestrea fotogramas preservando la duración | Secuencia de fotogramas remuestreada |
| FrameInterpolationAttack | Robustez (Video) | Ataque de interpolación de fotogramas insertando fotogramas mezclados para alterar la densidad temporal | Fotogramas de video interpolados |
| **Analizadores de calidad de imagen** | | | |
| InceptionScoreCalculator | Calidad (Imagen) | Evaluar calidad y diversidad de imagen generada | Puntuación IS |
| FIDCalculator | Calidad (Imagen) | Distancia de Inception de Fréchet, midiendo la diferencia de distribución entre imágenes generadas y reales | Valor FID |
| LPIPSAnalyzer | Calidad (Imagen) | Similitud de parche de imagen perceptual aprendida, evaluando calidad perceptual | Distancia LPIPS |
| CLIPScoreCalculator | Calidad (Imagen) | Evaluación de consistencia texto-imagen basada en CLIP | Puntuación de similitud CLIP |
| PSNRAnalyzer | Calidad (Imagen) | Relación señal-ruido de pico, midiendo la distorsión de imagen | Valor PSNR (dB) |
| NIQECalculator | Calidad (Imagen) | Evaluador de calidad de imagen natural, evaluación de calidad sin referencia | Puntuación NIQE |
| SSIMAnalyzer | Calidad (Imagen) | Índice de similitud estructural entre dos imágenes | Valor SSIM |
| BRISQUEAnalyzer | Calidad (Imagen) | Evaluador de calidad espacial de imagen ciega/sin referencia, evaluando la calidad perceptual de una imagen sin requerir una referencia | Puntuación BRISQUE |
| VIFAnalyzer | Calidad (Imagen) | Analizador de fidelidad de información visual, comparando una imagen distorsionada con una imagen de referencia para cuantificar la cantidad de información visual preservada | Valor VIF |
| FSIMAnalyzer | Calidad (Imagen) | Analizador de índice de similitud de características, comparando similitud estructural entre dos imágenes basada en congruencia de fase y magnitud de gradiente | Valor FSIM |
| **Analizadores de calidad de video** | | | |
| SubjectConsistencyAnalyzer | Calidad (Video) | Evaluar consistencia de objetos sujeto en video | Puntuación de consistencia de sujeto |
| BackgroundConsistencyAnalyzer | Calidad (Video) | Evaluar coherencia y estabilidad del fondo en video | Puntuación de consistencia de fondo |
| MotionSmoothnessAnalyzer | Calidad (Video) | Evaluar suavidad del movimiento del video | Métrica de suavidad de movimiento |
| DynamicDegreeAnalyzer | Calidad (Video) | Medir nivel dinámico y magnitud de cambio en video | Valor de grado dinámico |
| ImagingQualityAnalyzer | Calidad (Video) | Evaluación integral de calidad de imagen de video | Puntuación de calidad de imagen |

## Instalación

### Configuración del entorno

- Python 3.10+
- PyTorch
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

*Nota:* Algunos algoritmos pueden requerir pasos de configuración adicionales. Consulta la documentación de algoritmos individuales para requisitos específicos.

## Inicio rápido

Aquí hay un ejemplo simple para comenzar con MarkDiffusion:

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Configuración del dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configurar pipeline de difusión
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

# Cargar algoritmo de marca de agua
watermark = AutoWatermark.load('TR', 
                              algorithm_config='config/TR.json',
                              diffusion_config=diffusion_config)

# Generar medios con marca de agua
prompt = "A beautiful sunset over the ocean"
watermarked_image = watermark.generate_watermarked_media(prompt)

# Detectar marca de agua
detection_result = watermark.detect_watermark_in_media(watermarked_image)
print(f"Watermark detected: {detection_result}")
```

## Cómo usar el kit de herramientas

Proporcionamos ejemplos extensos en `MarkDiffusion_demo.ipynb`.

### Generación y detección de medios con marcas de agua

#### Casos para generar y detectar medios con marcas de agua

```python
import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig

# Cargar algoritmo de marca de agua
mywatermark = AutoWatermark.load(
    'GS',
    algorithm_config=f'config/GS.json',
    diffusion_config=diffusion_config
)

# Generar imagen con marca de agua
watermarked_image = mywatermark.generate_watermarked_media(
    input_data="A beautiful landscape with a river and mountains"
)

# Visualizar la imagen con marca de agua
watermarked_image.show()

# Detectar marca de agua
detection_result = mywatermark.detect_watermark_in_media(watermarked_image)
print(detection_result)
```

### Visualización de mecanismos de marcas de agua

El kit de herramientas incluye herramientas de visualización personalizadas que permiten vistas claras y perspicaces sobre cómo operan los diferentes algoritmos de marcas de agua en varios escenarios. Estas visualizaciones ayudan a desmitificar los mecanismos de los algoritmos, haciéndolos más comprensibles para los usuarios.

<img src="img/fig2_visualization_mechanism.png" alt="Watermarking Mechanism Visualization" style="zoom:40%;" />

#### Casos para visualizar mecanismos de marcas de agua

```python
from visualize.auto_visualization import AutoVisualizer

# Obtener datos para visualización
data_for_visualization = mywatermark.get_data_for_visualize(watermarked_image)

# Cargar visualizador
visualizer = AutoVisualizer.load('GS', 
                                data_for_visualization=data_for_visualization)

# Dibujar diagramas en el lienzo de Matplotlib
fig = visualizer.visualize(rows=2, cols=2, 
                          methods=['draw_watermark_bits', 
                                  'draw_reconstructed_watermark_bits', 
                                  'draw_inverted_latents', 
                                  'draw_inverted_latents_fft'])
```

### Pipelines de evaluación

#### Casos para evaluación

1. **Pipeline de detección de marcas de agua**

```python
from evaluation.dataset import StableDiffusionPromptsDataset
from evaluation.pipelines.detection import (
    WatermarkedMediaDetectionPipeline, 
    UnWatermarkedMediaDetectionPipeline, 
    DetectionPipelineReturnType
)
from evaluation.tools.image_editor import JPEGCompression
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator

# Conjunto de datos
my_dataset = StableDiffusionPromptsDataset(max_samples=200)

# Configurar pipelines de detección
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

# Configurar parámetros de detección
detection_kwargs = {
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
}

# Calcular tasas de éxito
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

2. **Pipeline de análisis de calidad de imagen**

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

# Ejemplos de diferentes métricas de calidad:

# NIQE (Evaluador de calidad de imagen natural)
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

# Puntuación CLIP
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

# FID (Distancia de Inception de Fréchet)
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

# IS (Puntuación Inception)
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

# LPIPS (Similitud de parche de imagen perceptual aprendida)
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

# PSNR (Relación señal-ruido de pico)
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

# Cargar marca de agua y evaluar
my_watermark = AutoWatermark.load(
    f'{algorithm_name}',
    algorithm_config=f'config/{algorithm_name}.json',
    diffusion_config=diffusion_config
)

print(pipeline.evaluate(my_watermark))
```

3. **Pipeline de análisis de calidad de video**

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

# Cargar conjunto de datos VBench
my_dataset = VBenchDataset(max_samples=200, dimension=dimension)

# Inicializar analizador según métrica
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

# Crear pipeline de análisis de calidad de video
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

## Citación
```
@article{pan2025markdiffusion,
  title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
  author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
  journal={arXiv preprint arXiv:2509.10569},
  year={2025}
}
```

