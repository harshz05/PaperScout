"""Mock retriever providing hardcoded AI/ML research papers for PaperScout."""

import re

_MOCK_PAPERS: list[dict] = [
    # cs.CL — Computation and Language
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": (
            "We explore a general-purpose fine-tuning recipe for retrieval-augmented "
            "generation (RAG) models that combine a pre-trained parametric seq2seq "
            "generator (BART-large) with a non-parametric dense vector index of Wikipedia "
            "accessed via a Dense Passage Retriever (DPR). Evaluated on open-domain "
            "question answering benchmarks including Natural Questions, TriviaQA, "
            "WebQuestions, and CuratedTrec, as well as MS-MARCO and FEVER fact "
            "verification, RAG sets state-of-the-art results and generates more "
            "specific, diverse, and factual language than parametric-only baselines. "
            "Key limitations include higher inference latency from top-k document "
            "marginalization and reliance on a static Wikipedia index that cannot "
            "handle facts absent from the corpus."
        ),
        "arxiv_id": "2005.11401",
        "category": "cs.CL",
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "abstract": (
            "We propose Low-Rank Adaptation (LoRA), which freezes pre-trained model "
            "weights and injects trainable rank-decomposition matrices into each "
            "Transformer attention layer, greatly reducing the number of trainable "
            "parameters for downstream tasks. Across RoBERTa, DeBERTa, GPT-2, and "
            "GPT-3 175B on GLUE, WikiSQL, MNLI, and SAMSum summarization benchmarks, "
            "LoRA matches or outperforms full fine-tuning while cutting GPU memory "
            "requirements by up to 3x and adding zero inference latency after weight "
            "merging. A key limitation is that batching inputs across multiple "
            "task-specific adapters in a single forward pass is non-trivial when "
            "weights are merged, and rank selection remains empirical."
        ),
        "arxiv_id": "2106.09685",
        "category": "cs.CL",
    },
    {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "abstract": (
            "While Reinforcement Learning from Human Feedback (RLHF) aligns large "
            "language models with human preferences, standard PPO-based pipelines are "
            "complex and unstable. We introduce Direct Preference Optimization (DPO), "
            "a closed-form reparameterization of the Bradley-Terry reward objective "
            "that optimizes the policy directly using a simple binary cross-entropy "
            "loss without fitting an explicit reward model or sampling during "
            "training. Tested on IMDb sentiment generation, Anthropic HH dialogue, and "
            "TL;DR summarization with Pythia and GPT-J models, DPO matches or "
            "exceeds PPO alignment quality. Limitations include sensitivity to "
            "out-of-distribution preference pairs and potential degradation when the "
            "reference policy diverges from the training data distribution."
        ),
        "arxiv_id": "2305.18290",
        "category": "cs.CL",
    },
    {
        "title": "Self-Refine: Iterative Refinement with Self-Feedback",
        "abstract": (
            "We present Self-Refine, an inference-time framework in which a single "
            "large language model generates an initial output, critiques its own "
            "response using actionable natural language feedback, and iteratively "
            "revises the output without additional supervised training or external "
            "verifiers. Across seven diverse NLP and code tasks—including GSM8K math "
            "reasoning, dialogue response generation, sentiment reversal, and "
            "acronym generation—Self-Refine improves performance by roughly 20% over "
            "single-pass decoding on GPT-3.5 and GPT-4. Key limitations are "
            "multiplied token cost and latency from iterative passes, and failure "
            "when the base model lacks the capability to recognize its own errors."
        ),
        "arxiv_id": "2303.17651",
        "category": "cs.CL",
    },
    # cs.LG — Machine Learning
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "abstract": (
            "We present high-quality image synthesis results using Denoising "
            "Diffusion Probabilistic Models (DDPM), a class of latent variable models "
            "trained via a weighted variational bound that connects directly to "
            "denoising score matching with Langevin dynamics. Using a U-Net backbone "
            "with sinusoidal timestep embeddings, DDPM achieves an FID of 3.17 on "
            "unconditional CIFAR-10 and competitive sample quality on CelebA-HQ and "
            "LSUN datasets. A primary limitation is slow iterative sampling, "
            "requiring hundreds to thousands of sequential Markov chain reverse "
            "diffusion steps per generated sample."
        ),
        "arxiv_id": "2006.11239",
        "category": "cs.LG",
    },
    {
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "abstract": (
            "Transformers scale poorly with sequence length due to quadratic time and "
            "memory complexity in self-attention. We propose FlashAttention, an "
            "IO-aware exact attention algorithm that uses tiling and online softmax "
            "recomputation to avoid materializing the full N-by-N attention matrix in "
            "GPU high-bandwidth memory (HBM), keeping intermediate states in fast "
            "on-chip SRAM. Benchmarked on BERT-large, GPT-2, and Long Range Arena "
            "(LRA) tasks, FlashAttention delivers up to 3x training speedups and "
            "enables context lengths up to 64K tokens with exact mathematical "
            "equivalence. Key limitations are dependence on custom CUDA kernels "
            "tailored to specific GPU SRAM sizes and reduced gains for very short "
            "sequences."
        ),
        "arxiv_id": "2205.14135",
        "category": "cs.LG",
    },
    {
        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "abstract": (
            "Structured state space models (SSMs) struggle on discrete, "
            "information-dense modalities like text because their time-invariant "
            "parameters cannot perform content-based reasoning. We introduce Mamba, "
            "a selective state space architecture whose transition parameters are "
            "input-dependent functions computed via a hardware-aware parallel scan. "
            "Evaluated on The Pile language modeling, DNA sequence modeling, and "
            "audio generation benchmarks, Mamba achieves linear-time scaling in "
            "sequence length, 5x higher inference throughput than Transformers, and "
            "strong performance up to million-token contexts. A key limitation is "
            "weaker exact associative recall over long contexts compared to full "
            "quadratic attention layers."
        ),
        "arxiv_id": "2312.00752",
        "category": "cs.LG",
    },
    {
        "title": "Sharpness-Aware Minimization for Efficiently Improving Generalization",
        "abstract": (
            "Overparameterized deep networks often converge to sharp minima that "
            "generalize poorly despite zero training loss. We introduce "
            "Sharpness-Aware Minimization (SAM), which simultaneously minimizes loss "
            "value and loss sharpness by seeking parameters whose entire local "
            "neighborhood has uniformly low training loss via a worst-case weight "
            "perturbation step. Evaluated across CIFAR-10, CIFAR-100, ImageNet, and "
            "SVHN with ResNet, WideResNet, and Vision Transformer architectures, SAM "
            "improves test accuracy and robustness to label noise. The main "
            "limitation is a 2x computational overhead per training step due to "
            "requiring two forward-backward gradient passes."
        ),
        "arxiv_id": "2010.01412",
        "category": "cs.LG",
    },
    # cs.CV — Computer Vision
    {
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "abstract": (
            "We show that a pure Transformer applied directly to sequences of "
            "fixed-size 16x16 image patches—the Vision Transformer (ViT)—performs "
            "strongly on image classification without convolutional inductive biases. "
            "When pre-trained on large-scale datasets such as ImageNet-21k and "
            "JFT-300M and fine-tuned on ImageNet, CIFAR-100, and VTAB, ViT matches "
            "or exceeds state-of-the-art CNNs like BiT while requiring fewer "
            "pre-training FLOPs. A key limitation is poor generalization when trained "
            "from scratch on mid-sized datasets like ImageNet-1k alone due to the "
            "lack of translation equivariance and locality priors."
        ),
        "arxiv_id": "2010.11929",
        "category": "cs.CV",
    },
    {
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "abstract": (
            "We introduce CLIP (Contrastive Language-Image Pre-training), which "
            "jointly trains an image encoder (ResNet or ViT) and a text encoder "
            "using a symmetric contrastive loss over 400 million image-text pairs "
            "collected from the internet. Across more than 30 computer vision "
            "datasets—including ImageNet, Food-101, Oxford-IIIT Pets, and STL-10—"
            "CLIP achieves strong zero-shot classification accuracy comparable to "
            "fully supervised ResNet-50 without task-specific training labels. "
            "Limitations include weak performance on fine-grained counting, abstract "
            "spatial reasoning, and out-of-distribution tasks like MNIST digit "
            "classification, as well as inherited web-data biases."
        ),
        "arxiv_id": "2103.00020",
        "category": "cs.CV",
    },
    {
        "title": "Segment Anything",
        "abstract": (
            "We introduce the Segment Anything Model (SAM), a promptable image "
            "segmentation architecture comprising an MAE-pretrained ViT image "
            "encoder, a flexible prompt encoder supporting points, boxes, masks, and "
            "text, and a lightweight mask decoder. Trained on SA-1B—a new dataset "
            "of over 1 billion segmentation masks across 11 million licensed images "
            "built via a data engine—SAM demonstrates strong zero-shot transfer on "
            "23 diverse segmentation benchmarks including ADE20K, LVIS, and COCO. "
            "Key limitations include high compute cost of the heavy ViT-H image "
            "encoder for real-time video and occasional boundary errors on thin "
            "structures without semantic class awareness."
        ),
        "arxiv_id": "2304.02643",
        "category": "cs.CV",
    },
    {
        "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "abstract": (
            "Neural Radiance Fields (NeRFs) achieve photorealistic novel view "
            "synthesis but suffer from slow volumetric ray-marching rendering. We "
            "present 3D Gaussian Splatting, which represents scenes as anisotropic 3D "
            "Gaussians initialized from Structure-from-Motion point clouds and "
            "optimized via interleaved adaptive density control and a fast "
            "tile-based differentiable rasterizer. Tested on Mip-NeRF360, Tanks and "
            "Temples, and Deep Blending datasets, the method reaches over 100 FPS "
            "at 1080p resolution while matching or surpassing NeRF visual quality. "
            "Limitations include large memory footprints from storing millions of "
            "Gaussian parameters and sensitivity to sparse or poorly calibrated input "
            "camera views."
        ),
        "arxiv_id": "2308.04079",
        "category": "cs.CV",
    },
]


def _score_paper(paper: dict, terms: list[str]) -> int:
    """Compute a simple keyword overlap score for ranking mock results."""
    if not terms:
        return 0
    haystack = f"{paper['title']} {paper['abstract']} {paper['category']}".lower()
    return sum(1 for term in terms if term in haystack)


def retrieve_papers(query: str, category: str, top_k: int = 4) -> list[dict]:
    """Return up to top_k mock AI/ML research papers, optionally filtered by category."""
    if top_k <= 0:
        return []

    if category and category != "All":
        candidates = [paper for paper in _MOCK_PAPERS if paper["category"] == category]
    else:
        candidates = list(_MOCK_PAPERS)

    terms = re.findall(r"[a-z0-9]+", (query or "").lower())
    if terms:
        # Sort by descending keyword overlap while preserving original order on ties.
        candidates.sort(key=lambda paper: _score_paper(paper, terms), reverse=True)

    return [dict(paper) for paper in candidates[:top_k]]
