# Competition Statistics Report

This report analyzes the performance of Entangled vs Separable agents in Pong competition.

## Clamped Condition (ActorParamClamped=True)

**Total Games Played:** 100

### Winning Rates

| Agent | Wins | Win Rate |
|-------|------|----------|
| Separable Agent (first_0) | 63/100 | 63.00% |
| Entangled Agent (second_0) | 37/100 | 37.00% |
| Ties | 0/100 | 0.00% |

### Score Statistics

| Metric | Separable Agent | Entangled Agent |
|--------|-----------------|------------------|
| Average Score | 16.91 | 14.05 |
| Min Score | 1.0 | 0.0 |
| Max Score | 21.0 | 21.0 |

### Summary

**Separable Agent** performs better with a **26.00%** advantage.

## Unclamped Condition (ActorParamClamped=False)

**Total Games Played:** 100

### Winning Rates

| Agent | Wins | Win Rate |
|-------|------|----------|
| Separable Agent (first_0) | 63/100 | 63.00% |
| Entangled Agent (second_0) | 37/100 | 37.00% |
| Ties | 0/100 | 0.00% |

### Score Statistics

| Metric | Separable Agent | Entangled Agent |
|--------|-----------------|------------------|
| Average Score | 16.91 | 14.05 |
| Min Score | 1.0 | 0.0 |
| Max Score | 21.0 | 21.0 |

### Summary

**Separable Agent** performs better with a **26.00%** advantage.

## Comparison: Clamped vs Unclamped Conditions

### Agent Performance Comparison

| Agent | Clamped Win Rate | Unclamped Win Rate | Improvement |
|-------|------------------|--------------------|--------------|
| Entangled Agent | 37.00% | 37.00% | +0.00% |
| Separable Agent | 63.00% | 63.00% | +0.00% |

### Performance Analysis

- **Entangled agent** shows no difference between conditions
- **Separable agent** shows no difference between conditions

### Overall Impact of Clamping

| Condition | Entangled Advantage |
|-----------|---------------------|
| Clamped | -26.00% |
| Unclamped | -26.00% |

🔍 **Key Finding:** Clamping has no net effect on relative performance

---
*Report generated automatically by competeStat.py*
