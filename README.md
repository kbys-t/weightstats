# weightstats

numpy-like weighted statics.

## Dependency

numpy

## Installation

```bash
git clone https://github.com/kbys-t/weightstats.git
cd weightstats
pip install -e .
```

## How to use

For example,

```python
import weightstats as ws

a = np.random.rand(6, 3, 2)
w = np.random.rand(6)

print ws.mean(a, axis=0, weights=w, keepdims=True)
print ws.var(a, axis=0, weights=w, keepdims=True)
print ws.std(a, axis=0, weights=w, keepdims=True)
print ws.percentile(a, 10, axis=0, weights=w, keepdims=True)
print ws.median(a, axis=0, weights=w, keepdims=True)
```
