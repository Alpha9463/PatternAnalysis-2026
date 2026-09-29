# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students in 2025 at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:

* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:

* segmentation
* classification
* graph neural networks
* StyleGAN
* Stable diffusion
* transformers
etc.

## ISIC melanoma experiments

Run a configured direct-CNN baseline from the repository root:

```bash
python -m src.evaluate --config configs/base.yaml
```

Run the Siamese model:

```bash
python -m src.evaluate --config configs/siamese.yaml
```

Both example YAML files use the loader's default local `data` directory. Every omitted setting uses the defaults in `src/config.py`. In particular, `device: auto` is the default: CUDA is chosen when available, then Apple MPS, then CPU. Add an explicit device only when you need to force one:

```bash
python -m src.evaluate --config configs/base.yaml --device mps
```

Command-line values override the YAML file, which makes short experiment variants easy to run:

```bash
python -m src.evaluate --config configs/base.yaml --epochs 10 --batch-size 8
```
