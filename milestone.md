# 🔴 MiniTorch Roadmap

> Build a tiny deep learning framework from scratch and understand how modern libraries like PyTorch actually work.

---

# 🔴 Milestone 1 — Autograd

### 📌 Topics
- ✅ Autograd Engine
- ✅ `Value` Class

### 🎯 Goal
Review Lesson 3 and rebuild the automatic differentiation engine.

---

# 🔴 Milestone 2 — Module

### 📌 Topics
- ✅ `Module` Class

### 🎯 Goal
Create the parent class that every neural network component will inherit from.

---

# 🔴 Milestone 3 — Linear

### 📌 Topics
- ✅ `Linear` Layer

### 🎯 Goal
Take the neuron from earlier lessons and transform it into a fully reusable Linear layer.

> 🤯 Huge realization:
>
> A neuron is just a Linear layer with an activation function.

---

# 🔴 Milestone 4 — Sequential

### 📌 Topics
- ✅ `Sequential(...)`

### 🎯 Goal
Chain multiple layers together into one model.

---

# 🔴 Milestone 5 — Activations

### 📌 Topics
- ✅ ReLU
- ✅ Sigmoid
- ✅ Tanh
- ⭐ GELU

### 🎯 Goal
Implement the most commonly used activation functions from scratch.

---

# 🔴 Milestone 6 — Loss Functions

### 📌 Topics
- ✅ Mean Squared Error (MSE)
- ✅ Binary Cross Entropy (BCE)
- ⭐ Cross Entropy

### 🎯 Goal
Teach the model how to measure its mistakes.

---

# 🔴 Milestone 7 — Optimizers

### 📌 Topics
- ✅ SGD
- ✅ Momentum
- ⭐ Adam
- ⭐ AdamW

### 🎯 Goal
Learn how neural networks actually update their parameters.

---

# 🔴 Milestone 8 — Regularization

### 📌 Topics
- ✅ Dropout
- ✅ Normalization

### 🎯 Goal
Prevent overfitting and improve generalization.

---

# 🔴 Milestone 9 — Initialization

### 📌 Topics
- ✅ Xavier Initialization
- ✅ He Initialization

### 🎯 Goal
Initialize weights the smart way for faster and more stable training.

---

# 🔴 Milestone 10 — Learning Rate Schedulers

### 📌 Topics
- ✅ Warmup
- ✅ Cosine Scheduler

### 🎯 Goal
Control how the learning rate changes during training.

---

# 🔴 Milestone 11 — Trainer ⭐⭐⭐⭐⭐

### 📌 Topics
- ✅ Training Pipeline

### 🎯 Goal
Connect every component built so far into one complete deep learning workflow.

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
   ↓
Scheduler
   ↓
Repeat
```

> 🚀 This is where everything finally comes together.
>
> Lessons 1–10 become one complete training pipeline.

---

# 🔴 Milestone 12 — Example Project ⭐⭐⭐⭐⭐

### 📌 Topics
- ✅ Circle Classification

### 🎯 Goal
Train a neural network from scratch using everything you've built.

```text
Data
   ↓
Model
   ↓
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
   ↓
Repeat
   ↓
Prediction 🎉
```

---

# 🔴 Final Destination

At the end of these milestones, you'll have built your own mini deep learning framework featuring:

- ✅ Autograd Engine
- ✅ Value Class
- ✅ Module System
- ✅ Linear Layers
- ✅ Sequential Models
- ✅ Activation Functions
- ✅ Loss Functions
- ✅ Optimizers
- ✅ Regularization
- ✅ Weight Initialization
- ✅ Learning Rate Schedulers
- ✅ Complete Trainer
- ✅ End-to-End Neural Network Training

