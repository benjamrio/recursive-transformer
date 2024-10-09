import math


MAX_LR = 6e-5
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10


def get_lr(it, max_steps):
    max_lr = MAX_LR
    min_lr = MIN_LR
    warmup_steps = WARMUP_STEPS
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

