# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Primary Goal for this Project:**

I am learning Red Hat OpenShift AI for model training, specifically focusing on LLM fine-tuning using distributed training capabilities. This project implements **music lyrics understanding fine-tuning** as a compelling demonstration of specialized AI capabilities. The goal is to transform a generic language model into a music expert that can analyze lyrics, understand artistic styles, identify genres, and provide cultural context.

**Project Evolution Context:**
This project evolved from a ResNet-18 CIFAR-10 training project. We successfully implemented single-node PyTorchJob training and added metrics annotations for OpenShift AI monitoring. The next phase focuses on LLM fine-tuning to demonstrate more advanced AI capabilities relevant to modern conference audiences.

## Project Overview

**Current Phase: Single-Node LLM Fine-Tuning**
We are implementing single-node fine-tuning first to perfect the demo quality and training pipeline. Future phases will scale to distributed training across multiple nodes to demonstrate performance benefits.

**Target Demo Scenario:**
- **Before Fine-tuning**: Generic model provides basic text analysis ("This appears to be about music and emotions")
- **After Fine-tuning**: Specialized model provides expert-level insights about artist styles, genre characteristics, lyrical themes, and cultural context

**Dataset**: HuggingFace rajtripathi/5M-Songs-Lyrics (5 million song entries with lyrics and artists). Specifically the data has a column with an instruction of:

` Generate a song verse in the style of <artist> in the genre of <genre type>.`

And then the next column contains the annotated lyrics of an actual song by that artist. 

**Model**: Microsoft Phi-3-small-8k-instruct with LoRA fine-tuning for efficient adaptation

**Training Technique**: High-performance LoRA (Low-Rank Adaptation) optimized for maximum VRAM utilization

## Environment and Infrastructure

**Red Hat OpenShift AI Cluster Details:**
- **Nodes**: 3 nodes available (currently using 1 for single-node training)
- **Per Node Resources**: 
  - RAM: 32 GB
  - CPU: 8 vCPU  
  - GPU: 48 GB VRAM NVIDIA L40S
- **Total Available**: 96 GB RAM, 24 vCPU, 144 GB VRAM
- **Current Utilization**: 1 node (45 GB VRAM for optimized training)

**Namespace/Project**: lyrical-professor (consistent across all phases)

**Storage Configuration:**
- **training-data-pvc**: Repurposed for lyrics dataset storage
- **trained-models-pvc**: Model outputs and checkpoints  
- **workspace-pvc**: Working directory and temporary files

**Key Technical Decisions:**
- **No RAG (Retrieval-Augmented Generation)**: Pure fine-tuning approach for reliable demo
- **No External Dependencies**: Self-contained model knowledge
- **Maximum VRAM Utilization**: 45GB/48GB per GPU (94% utilization)
- **High-Rank LoRA**: r=128 for maximum adaptation capacity

## Current Resource Optimization Assumptions

**VRAM Allocation Strategy (45GB per GPU):**
- Base Model (FP16): ~6GB
- LoRA Parameters (r=128): ~500MB  
- Large Batch Processing: ~25GB
- Optimizer States: ~1GB
- Gradient Buffers: ~2GB
- Working Memory: ~8GB
- Safety Buffer: ~2.5GB

**Training Configuration:**
- **Batch Size**: 24 samples per GPU (vs. previous 2-4)
- **Precision**: FP16 throughout (no quantization)
- **LoRA Rank**: 128 (vs. previous 32)
- **Training Time**: 45-90 seconds (vs. previous 3-5 minutes)
- **Throughput**: ~1000 samples/second

## Key Files and Architecture

### Core Training Files
- `training/finetune_music_model.py`: Main training script with LoRA fine-tuning
- `training/config.yaml`: Training configuration and hyperparameters
- `data/download_dataset.py`: Download and process 5M Songs Lyrics dataset
- `data/process_lyrics.py`: Data preprocessing and instruction format creation

### Deployment Files  
- `deploy/pytorch-training-job.yaml`: Single-node PyTorchJob for OpenShift AI
- `deploy/pvcbindings.yaml`: Persistent Volume Claims (copied from ResNet project)

### Evaluation and Demo
- `evaluation/demo_scenarios.py`: Demo scenarios showing before/after capabilities
- `models/`: Saved model checkpoints and outputs

### Documentation and Planning
- `MusicLyricUnderstandingPlan.md`: Comprehensive fine-tuning implementation plan
- `external-doc/`: OpenShift AI documentation for distributed training (copied from ResNet project)

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Download and process dataset  
python data/download_dataset.py
python data/process_lyrics.py

# Run training locally (single GPU)
python training/finetune_music_model.py
```

### OpenShift AI Deployment
```bash
# Deploy PVCs (if not already deployed)
kubectl apply -f deploy/pvcbindings.yaml

# Deploy single-node training job
kubectl apply -f deploy/pytorch-training-job.yaml

# Monitor training progress
kubectl logs -f pytorchjob/llm-music-finetuning

# Check job status
kubectl get pytorchjobs -n rhoai-learning
```

### Environment Variables
The training script accepts configuration through environment variables:
- `EPOCHS`: Number of training epochs (default: 2)
- `BATCH_SIZE`: Training batch size per GPU (default: 24)
- `LEARNING_RATE`: Learning rate (default: 2e-4)
- `LORA_RANK`: LoRA adaptation rank (default: 128)
- `DATA_DIR`: Directory for lyrics dataset (default: /shared/data)
- `OUTPUT_DIR`: Directory for model outputs (default: /shared/models)

## Training Pipeline Architecture

### Model Architecture: Microsoft Phi-3-small-8k-instruct

### Data Pipeline: 5M Songs Lyrics
- **Source**: HuggingFace rajtripathi/5M-Songs-Lyrics dataset
- **Processing**: Instruction-following format creation
- **Tasks**: Lyric analysis, artist style recognition, genre classification
- **Training Examples**: ~90,000 processed samples for demo

### Training Features:
- **High-Performance LoRA**: Rank 128 for maximum adaptation capacity
- **Large Batch Training**: 24 samples per GPU for fast convergence  
- **Automatic Checkpointing**: Best model saving based on validation loss
- **Demo Preparation**: Before/after model comparison capabilities
- **Progress Monitoring**: TensorBoard integration for training visualization

### Output Artifacts (saved to `OUTPUT_DIR`):
- `lora_adapter/`: LoRA adapter weights
- `best_model/`: Best performing model checkpoint
- `training_history.json`: Epoch-by-epoch metrics
- `training_summary.json`: Training configuration and results
- `demo_outputs/`: Sample outputs for demo scenarios

## Monitoring and Metrics

**OpenShift AI Integration:**
- PyTorchJob annotated for metrics collection in OpenShift AI console
- Resource usage monitoring via OpenShift AI → Distributed workloads → Project metrics
- GPU utilization and memory usage tracking

**Training Metrics:**
- Training/validation loss curves
- Learning rate scheduling
- Gradient norms and optimization stability
- Sample throughput and training speed

## Future Phases and Scaling

### Phase 2: Distributed Training (Optional)
- **Scale to 3 nodes**: 72 samples effective batch size (24 × 3)
- **Expected speedup**: 2-3x faster training time
- **Architecture**: Master + 2 Workers using same LoRA configuration
- **Benefits**: Demonstrate distributed training scaling for larger models

### Phase 3: Conference Demo Optimization
- **Dual Model Loading**: Load both models into Kserve through vLLM and serve them up
- **Instant Comparisons**: Real-time before/after demonstrations  
- **Interactive Scenarios**: Multiple demo examples ready for audience
- **Performance Metrics**: Show training speed improvements and resource utilization

## Success Metrics and Demo Criteria

### Quantitative Improvements
- **Training Speed**: 45-90 seconds vs. 3-5 minutes baseline
- **VRAM Utilization**: 94% (45GB/48GB) vs. 50% baseline
- **Model Quality**: Specialized music knowledge vs. generic responses
- **Throughput**: ~1000 samples/second training rate

### Demo Success Criteria
- **Clear Differentiation**: Obvious improvement in response sophistication
- **Technical Accuracy**: All music-related claims should be verifiable
- **Audience Engagement**: Relatable music examples that resonate
- **Reliability**: Consistent performance during live demonstration

## Technical Learning Objectives

This project teaches practical skills in:
1. **LLM Fine-tuning**: Modern LoRA techniques for efficient adaptation
2. **Resource Optimization**: Maximum VRAM utilization strategies
3. **OpenShift AI**: Training Operator and distributed workload management
4. **Performance Engineering**: Large batch training and memory optimization
5. **Demo Preparation**: Creating compelling before/after AI demonstrations

## Important Reminders

- **Focus on demo quality**: Single-node perfection before distributed scaling
- **No external dependencies**: Self-contained model for reliable demos
- **Maximum resource utilization**: Use full 45GB VRAM capacity
- **Preserve learning path**: Keep options open for distributed training exploration
- **Conference relevance**: LLM fine-tuning resonates with modern AI audience

This project builds upon the ResNet-18 foundation while advancing to more sophisticated LLM capabilities that demonstrate the full potential of Red Hat OpenShift AI for modern machine learning workloads.