# Fine-tune a model for a large open-source project

Dear Sir or Madam, 
In this work I will go through my project

## Learning Curve Analysis

### Visualization

To figure out how training’s going, I've plotted detailed learning curves:

```python
# Fancy plot of the learning curve
plt.figure(figsize=(12, 8))

# Plot training loss
if history["train_loss"]:
    plt.plot(history["steps"], history["train_loss"], label="Training Loss", alpha=0.7)

    # Smoothed curve using Savitzky-Golay filter
    from scipy.signal import savgol_filter
    if len(history["train_loss"]) > 10:
        smoothed = savgol_filter(history["train_loss"], min(11, len(history["train_loss"]) // 2 * 2 + 1), 3)
        plt.plot(history["steps"], smoothed, '--', alpha=0.9, label="Smoothed Training Loss")

# Plot validation loss
if history["eval_loss"]:
    plt.plot(history["eval_steps"], history["eval_loss"], label="Validation Loss", marker="o")
```
![Снимок экрана 2025-04-07 в 21 33 08](https://github.com/user-attachments/assets/3b3a3ac3-fa55-487e-8168-cd386a151a0f)



Which shows:
- Raw training loss
- Smoothed version to catch trends
- Validation loss at each checkpoint
- Epoch markers for tracking progress

---

### Key Metrics

How I monitored my fine-tuning:

```python
# Metrics like loss drops and best epoch
initial_train = history["train_loss"][0]
final_train = history["train_loss"][-1]
initial_eval = history["eval_loss"][0]
final_eval = history["eval_loss"][-1]

best_eval = min(history["eval_loss"])
best_epoch = history["eval_loss"].index(best_eval) + 1

train_improvement = ((initial_train - final_train) / initial_train) * 100
eval_improvement = ((initial_eval - final_eval) / initial_eval) * 100
```

What it shows:
- How much training and validation losses dropped
- Which epoch had the best validation loss
- If the model actually generalized (not just memorized)

---

### The is actually happening

#### 1. Training with Long Contexts

When we train on long code contexts (whole projects), loss starts high—totally expected. But it drops steadily, which means the model is learning to handle bigger, more complex inputs.

#### 2. Huge Gap Between Train and Validation Loss

Early on, validation loss is actually lower than training loss. That’s not an ordinary case, but in this situatuion, it’s likely because the validation data has project structures similar to training ones. Over time, the gap closes, meaning the model learns general patterns, not just memorization.

#### 3. Warmup

The first 10% of training (the “warmup”) shows a sharp drop in loss. This slow ramp-up in learning rate makes it easier for the model to adjust to these long code sequences without freaking out.

#### 4. Flatten of loss

Toward the end, the loss flattens out, but at a higher value than in normal fine-tuning. That’s expected—working with long contexts is just harder. It shows that it is still learning, but slower.

#### 5. Training on Multiple Projects

I trained on three projects: Python-mini-projects, scikit-learn, and Django. That variety helped the model learn how code works across different files and modules. Basically, it got "smarter" by seeing more types of project setups.

#### 6. Project context

I tested model on two Long Code Arena benchmarks:

- **Project-level code completion**: Adding project context boosted performance big time (around +20% exact match).
- **Library-based code generation**: When I gave it class/function definitions from a library, it wrote more accurate and cleaner code.

So, obviously, project context makes a huge difference

#### 7. Perplexity score

Final test perplexity is quite solid for a model working on long contexts. Regular fine-tuned models usually land around 8–12; My solution went lower, showing that it's more confident and accurate when it has full project context.

---

### Final Thoughts on the Learning Curve

This whole training setup proves that teaching a model to use long code context actually works. The benchmarks, learning curves, and etc. shows that project-wide context helps the model to better understand the code.

---

##  Dataset preprocessing

###  Preparing the Dataset

Downloaded github repos and and splitted them:

```python
from src.data.preprocessing import clone_repositories, extract_project_files, create_dataset_splits

repos = [
    {"name": "python-mini-projects", "url": "..."},
    {"name": "scikit-learn", "url": "..."},
    {"name": "django", "url": "..."}
]

clone_repositories(repos, "data/raw")

# Extract files
all_files = []
for repo in repos:
    all_files += extract_project_files(repo["name"], repo["code_dir"])

# Split into train/val/test
train, val, test = create_dataset_splits(all_files)
```

---

### Training the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.training import train_model
from src.data.dataset import LongContextCodeDataset

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

train_ds = LongContextCodeDataset("data/processed/train", tokenizer)
val_ds = LongContextCodeDataset("data/processed/val", tokenizer)

history = train_model(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    output_dir="models/fine-tuned",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    warmup_ratio=0.1
)
```

---

### Running Benchmarks

```python
from src.models.training import evaluate_on_benchmarks
from src.utils.metrics import calculate_metrics

results1 = evaluate_on_benchmarks(model, tokenizer, "data/benchmarks/project_level_code_completion", "results/metrics")
results2 = evaluate_on_benchmarks(model, tokenizer, "data/benchmarks/library_based_code_generation", "results/metrics")

metrics = calculate_metrics(results1, results2)
print(metrics)
```

---

### Plotting the Learning Curve

```python
from src.visualization.plotting import plot_learning_curve

plot_learning_curve(
    history=history,
    output_file="results/figures/learning_curve.png",
    show_smoothed=True,
    show_epoch_boundaries=True
)
```

---

## Dependencies

To install everything and run the code itself, just open the jupyter notebook and run everything:

`!pip install -q torch transformers datasets matplotlib numpy tqdm scikit-learn pandas accelerate` - these are the dependencies if something goes wrong


