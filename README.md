# Einops Implementation

A from-scratch implementation of core einops functionality, focusing on the `rearrange` operation for tensor manipulation using Einstein notation-inspired syntax.

## Key Features

- Full support for `rearrange` operations including reshaping, transposition, axis splitting/merging, and repeating
- Comprehensive pattern parsing with ellipsis (...) support for batch dimensions
- Optimized for NumPy arrays with minimal intermediate operations
- Robust error handling with detailed error messages

## Implementation Highlights

- Pattern parser using regex for accurate token extraction
- Single-pass dimension processing to minimize overhead
- Smart handling of composite dimensions with automatic size inference
- Optimized path for simple transpositions without unnecessary reshaping

## Usage Examples

Basic operations:
```python
import numpy as np
from einops_impl import rearrange

# Transpose
result = rearrange(np.random.rand(3, 4), 'h w -> w h')

# Split axis
result = rearrange(np.random.rand(12, 10), '(h w) c -> h w c', h=3)

# Merge axes with batch dimensions
result = rearrange(np.random.rand(2, 3, 4, 5), '... h w -> ... (h w)')
```

## Testing

The implementation includes extensive tests covering:
- Core operations (transpose, reshape, split, merge)
- Ellipsis handling
- Error conditions
- Edge cases
- Performance with large tensors

## Design Decisions

- Used custom `EinopsError` for clear error attribution
- Pattern parsing separates tokenization from processing for better maintainability
- Dimension tracking uses dictionaries for O(1) lookups
- Optimized transpose path avoids unnecessary reshaping operations
- Single-pass dimension processing minimizes memory overhead

## Requirements

- Python 3.6+
- NumPy

