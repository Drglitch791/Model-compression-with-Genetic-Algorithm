# Genetic Algorithm–Based CNN Filter Pruning
This project applies a Genetic Algorithm (GA) to prune convolutional filters from deep CNN models such as **VGG16** and **VGG19**. The goal is straightforward: reduce as many parameters as possible while keeping accuracy loss under a defined threshold.

---

## Overview
CNNs typically contain redundant filters that can be removed without significantly hurting model performance. By eliminating such filters, we reduce:
- Parameter count
- FLOPs
- Inference latency

Each filter is encoded as a bit in a chromosome:
- `1` → keep filter
- `0` → prune filter

A Genetic Algorithm searches the combinatorial space of these binary strings and evolves toward a configuration that balances pruning with acceptable accuracy.

---

## Method
### Chromosome Representation
A binary vector, where each position corresponds to a convolutional filter.

### Fitness Function
If the accuracy drop is within the allowed threshold:
```

fitness = parameter_drop

```
If accuracy drop exceeds the threshold:
```

fitness = (parameter_drop / (accuracy_drop + ε)) - λ * accuracy_drop

```
Where:
- **ε** is an offset to stabilize the score when accuracy improves
- **λ** penalizes high accuracy drops

### Genetic Algorithm Steps
1. Initialize a population close to the original model.
2. Evaluate each individual.
3. Select top-K individuals as parents.
4. Apply single-point crossover.
5. Mutate offspring.
6. Merge offspring into population and retain top-T individuals.
7. Track the best solution across generations.

---

## Experimental Setup
| Parameter | Value |
|----------|--------|
| Initial Population (I) | 50 |
| Parents per Generation (K) | 10 |
| Mutation Probability (P) | 0.0002 |
| Population Cap (T) | 300 |
| Accuracy Threshold | 2% |

Models used:
- **VGG16** on CIFAR-10
- **VGG19** on SVHN

---

## Results
### VGG16 – CIFAR-10
![VGG16 Metrics](https://github.com/Rakeshlal791/Model-compression-with-Genetic-Algorithm/blob/master/MTP_final/PruningAlgo/debug/save_vgg16_cifar10.png)

---

### VGG19 – SVHN
![VGG19 Metrics](https://github.com/Rakeshlal791/Model-compression-with-Genetic-Algorithm/blob/master/MTP_final/PruningAlgo/debug/save_svhn.png)

---

## Comparison Tables
### VGG16 on CIFAR-10
| Metric | Base Model | Pruned Model |
|--------|------------|--------------|
| Test Accuracy | 86.87% | 85.24% (−1.87%) |
| Parameters | 15,245,130 | 9,326,336 (−38.82%) |
| FLOPs | 313,725,952 | 252,446,852 (−19.52%) |

### VGG19 on SVHN
| Metric | Base Model | Pruned Model |
|--------|------------|--------------|
| Test Accuracy | 92.43% | 91.418% (−1.09%) |
| Parameters | 20,554,826 | 16,235,999 (−21.01%) |
| FLOPs | 398,660,608 | 349,536,612 (−12.32%) |

---

## Repository Structure
```

/models          → pretrained model checkpoints
/ga              → genetic algorithm implementation
/pruning         → filter-removal utilities
/evaluation      → accuracy and FLOP measurement scripts
/docs            → figures and write-up

```

## Conclusion
The GA-based pruning approach successfully removes redundant CNN filters without requiring retraining. With accuracy drop constrained to a small threshold, the algorithm consistently finds models with significant parameter savings.

Further gains are likely by alternating pruning with retraining.
