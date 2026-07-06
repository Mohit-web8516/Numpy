# NumPy — Complete Concept Guide for Data Analysts
> A consolidated revision reference covering everything a Data Analyst needs from NumPy — beyond just tutorial basics. Pair with your `numpy_notes.md` (tutorial-based) for full coverage.

---

## Table of Contents
1. [Why NumPy Exists (the mental model)](#1-why-numpy-exists-the-mental-model)
2. [Array Creation](#2-array-creation)
3. [Array Properties & Data Types](#3-array-properties--data-types)
4. [Indexing, Slicing & Views vs Copies](#4-indexing-slicing--views-vs-copies)
5. [Boolean Masking & Fancy Indexing](#5-boolean-masking--fancy-indexing)
6. [Reshaping & Combining Arrays](#6-reshaping--combining-arrays)
7. [Broadcasting (the most important concept)](#7-broadcasting-the-most-important-concept)
8. [Aggregation & Statistics](#8-aggregation--statistics)
9. [Handling Missing Data (NaN)](#9-handling-missing-data-nan)
10. [Sorting, Searching & Set Operations](#10-sorting-searching--set-operations)
11. [Useful Utility Functions](#11-useful-utility-functions)
12. [Linear Algebra Basics](#12-linear-algebra-basics)
13. [Random Module](#13-random-module)
14. [Vectorization & Performance](#14-vectorization--performance)
15. [Where NumPy Fits vs Pandas](#15-where-numpy-fits-vs-pandas)
16. [Interview-Style Quick Answers](#16-interview-style-quick-answers)

---

## 1. Why NumPy Exists (the mental model)
Python lists are flexible but slow for numeric work — every element is a separate Python object scattered in memory. NumPy stores data in a single contiguous block of memory, all of the **same type**, which lets it use fast, compiled C loops instead of Python loops.

**The one-line mental model:** *"Operate on the whole array at once, not element by element."* This is called **vectorization**, and almost every NumPy concept below exists to support it.

```python
import numpy as np

# Slow (Python loop)
result = [x * 2 for x in range(1_000_000)]

# Fast (vectorized)
result = np.arange(1_000_000) * 2
```

---

## 2. Array Creation
| Function | What it does |
|---|---|
| `np.array([1,2,3])` | from a Python list |
| `np.zeros((r,c))` | array of zeros |
| `np.ones((r,c))` | array of ones |
| `np.full((r,c), val)` | array filled with a constant |
| `np.arange(start, stop, step)` | evenly spaced values, like `range()` |
| `np.linspace(start, stop, num)` | `num` evenly spaced values **between** start and stop (inclusive) |
| `np.eye(n)` | identity matrix (n×n, 1s on diagonal) |
| `np.random.random((r,c))` | random floats in [0,1) |

```python
np.linspace(0, 1, 5)   # [0. , 0.25, 0.5 , 0.75, 1. ]
np.arange(0, 1, 0.25)  # [0. , 0.25, 0.5 , 0.75]  -> stop is EXCLUDED
```
> **Interview trap:** `arange` excludes the stop value; `linspace` includes it and instead lets you specify *how many points you want*.

---

## 3. Array Properties & Data Types
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.shape    # (2, 3)
arr.ndim     # 2
arr.size     # 6
arr.dtype    # int64
```

**Data types matter for memory and correctness:**
```python
np.array([1, 2, 3], dtype=np.float32)   # explicit dtype
arr.astype(np.float64)                    # convert dtype (returns a new array)
```
| dtype | Use case |
|---|---|
| `int32` / `int64` | whole numbers |
| `float32` / `float64` | decimals (float64 is default, more precise but 2x memory) |
| `bool` | True/False masks |
| `object` | mixed types (avoid — loses NumPy's speed advantage) |

---

## 4.Indexing, Slicing & Views vs Copies
```python
arr = np.array([10, 20, 30, 40, 50])
arr[1:4]        # [20, 30, 40]
arr[::-1]       # reversed array

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[1, 2]     # 6           -> row 1, col 2
arr2d[:, 0]     # [1, 4, 7]   -> entire first column
```

**Critical concept — slicing returns a VIEW, not a copy:**
```python
original = np.array([1, 2, 3, 4, 5])
slice_ = original[1:3]
slice_[0] = 99
print(original)   # [1, 99, 3, 4, 5]  <- original CHANGED too!

# To avoid this, explicitly copy:
safe_slice = original[1:3].copy()
```
> This is one of the most common real-world bugs. Fancy indexing (using a list of indices) DOES return a copy; basic slicing does NOT.

---

## 5.Boolean Masking & Fancy Indexing
This is the #1 tool for real data filtering.

```python
data = np.array([12, 45, 7, 89, 34, 3, 67])

mask = data > 30            # [F, T, F, T, T, F, T]
filtered = data[mask]       # [45, 89, 34, 67]

# Combine conditions with & (and), | (or), ~ (not) — NOT Python's `and`/`or`
data[(data > 10) & (data < 50)]

# Fancy indexing: pass a list/array of specific positions
data[[0, 3, 5]]              # elements at index 0, 3, 5

# Conditional replacement
np.where(data > 30, 'high', 'low')
data[data < 10] = 0          # modify in place based on condition
```
> **Interview trap:** `and`/`or` don't work element-wise on arrays — you must use `&`, `|`, `~` with parentheses around each condition.

---

## 6. Reshaping & Combining Arrays
```python
arr = np.arange(12)
arr.reshape(3, 4)      # 3 rows, 4 cols
arr.reshape(3, -1)     # -1 = "figure this dimension out automatically"

arr.flatten()          # 2D -> 1D, always a COPY
arr.ravel()            # 2D -> 1D, a VIEW when possible (faster)
arr.T                  # transpose (swap rows/cols)

# Combining
np.concatenate([a, b])           # join along existing axis
np.vstack([a, b])                # stack vertically (add rows)
np.hstack([a, b])                # stack horizontally (add columns)

# Splitting
np.split(arr, 3)                 # split into 3 equal parts
```

---

## 7. Broadcasting (the most important concept)
Broadcasting lets NumPy perform operations on arrays of **different shapes** without writing loops or manually resizing.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
arr + 10                                   # adds 10 to every element -> scalar broadcasts

row = np.array([10, 20, 30])              # shape (3,)
arr + row                                  # row is "stretched" to match each row of arr
```

**Broadcasting rule:** Two shapes are compatible if, comparing dimensions from the right:
- they're equal, OR
- one of them is 1

```
arr shape: (2, 3)
row shape:    (3,)   -> treated as (1, 3), stretches to (2, 3). Compatible.
```

**Real analyst use case:** normalizing a column without a loop.
```python
column_means = data.mean(axis=0)
normalized = data - column_means   # subtracts the mean from every row automatically
```

---

## 8. Aggregation & Statistics
The `axis` parameter is the single most-confused concept — memorize this:
- **`axis=0`** → operate **down the columns** (collapse rows → one result per column)
- **`axis=1`** → operate **across the rows** (collapse columns → one result per row)

```python
data = np.array([[10, 20, 30], [40, 50, 60]])

data.sum()             # 210        -> total of everything
data.sum(axis=0)       # [50 70 90] -> column-wise sums
data.sum(axis=1)       # [60 150]   -> row-wise sums

data.mean()            # average
data.std()             # standard deviation
data.var()             # variance
np.median(data)        # median
np.percentile(data, 75)# 75th percentile
data.min(), data.max() # min/max
data.argmin(), data.argmax()  # INDEX of min/max value (very commonly asked)
```

---

## 9. Handling Missing Data (NaN)
Real datasets have gaps. Regular aggregation functions **break** on NaN — this trips up almost everyone at first.

```python
data = np.array([1, 2, np.nan, 4, 5])

data.mean()          # nan  <- silently ruins your whole calculation!
np.nanmean(data)      # 3.0  <- ignores NaN correctly

np.isnan(data)         # [False, False, True, False, False]
data[~np.isnan(data)]  # remove NaN values -> [1, 2, 4, 5]

# Other NaN-safe versions:
np.nansum(data)
np.nanstd(data)
np.nanmax(data)
np.nanmin(data)
```
> **Golden rule:** if your dataset might contain missing values, always reach for the `nan*` version of a function (`np.nanmean` not `np.mean`) — otherwise one missing value silently poisons the entire result.

---

## 10. Sorting, Searching & Set Operations
```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

np.sort(arr)             # sorted copy, ascending
np.argsort(arr)          # INDICES that would sort the array (very useful for ranking)

np.unique(arr)            # unique values, sorted
np.unique(arr, return_counts=True)  # values + how many times each appears

np.intersect1d(a, b)      # elements common to both arrays
np.union1d(a, b)          # all unique elements from both
np.setdiff1d(a, b)        # elements in a but NOT in b
```

---

## 11. Useful Utility Functions
```python
np.round(3.14159, 2)     # 3.14
np.clip(arr, 0, 100)     # cap all values between 0 and 100 (great for outlier control)
np.cumsum(arr)           # running total
np.cumprod(arr)           # running product
np.diff(arr)               # difference between consecutive elements
np.abs(arr)                 # absolute value
np.round(arr)                # round to nearest integer
```

**`np.clip` in practice (capping outliers):**
```python
sales = np.array([100, 250, -30, 5000, 300])
sales_clean = np.clip(sales, 0, 1000)   # negative -> 0, anything over 1000 -> 1000
```

---

## 12. Linear Algebra Basics
Less common day-to-day for a data analyst, but frequently asked in interviews and used in ML-adjacent work.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)                # dot product: sum of element-wise products
np.linalg.norm(a)           # magnitude/length of a vector

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

np.matmul(m1, m2)            # matrix multiplication (or use m1 @ m2)
np.linalg.inv(m1)            # matrix inverse
np.linalg.det(m1)            # determinant
```
> **Interview trap:** `m1 * m2` is element-wise multiplication; `m1 @ m2` (or `np.matmul`) is true matrix multiplication. Confusing these is a classic mistake.

---

## 13. Random Module
Used for sampling, simulations, and generating test data.

```python
np.random.seed(42)             # makes randomness REPRODUCIBLE — always set this for reports/demos

np.random.rand(3, 3)            # uniform floats in [0,1)
np.random.randn(3, 3)           # standard normal (mean 0, std 1) distribution
np.random.randint(0, 100, 5)     # 5 random integers between 0 and 99
np.random.choice([1,2,3,4,5], 3) # randomly pick 3 elements from a list
np.random.shuffle(arr)            # shuffle an array in place
```
> **Why `seed()` matters:** without it, results change every run, which makes analysis non-reproducible — a common ask in interviews and real reporting.

---

## 14. Vectorization & Performance
```python
# AVOID: Python-level loop over array elements
result = []
for x in data:
    result.append(x ** 2)

# PREFER: vectorized operation
result = data ** 2
```
- Vectorized operations are **10–100x faster** on large arrays because the loop happens in compiled C, not Python.
- `np.vectorize()` exists for cases where no vectorized version is available, but it's just a thin wrapper around a Python loop — it's convenient, **not** genuinely fast. Prefer true array operations wherever possible.

---

## 15. Where NumPy Fits vs Pandas
A common point of confusion once you start Pandas:

| | NumPy | Pandas |
|---|---|---|
| Core structure | `ndarray` (no labels) | `DataFrame`/`Series` (labeled rows/columns) |
| Best for | pure numeric computation, math, matrices | tabular data, mixed types, real datasets |
| Missing data | `np.nan`, needs `nan*` functions | built-in `NaN` handling (`.dropna()`, `.fillna()`) |
| Relationship | Pandas is **built on top of** NumPy — every DataFrame column is a NumPy array underneath | |

```python
df['sales'].values        # pulls the underlying NumPy array out of a Pandas column
```

---

## 16. Interview-Style Quick Answers
Short, ready-to-say answers for common questions:

- **"Why NumPy over lists?"** → Contiguous memory + same dtype → vectorized C-level operations → much faster and more memory-efficient for numeric work.
- **"Difference between `.flatten()` and `.ravel()`?"** → `flatten()` always returns a copy; `ravel()` returns a view when possible (faster, but changes can propagate back to the original).
- **"What is broadcasting?"** → NumPy's rule for automatically expanding smaller arrays to match a larger array's shape during arithmetic, without copying data.
- **"axis=0 vs axis=1?"** → `axis=0` collapses down columns (one result per column); `axis=1` collapses across rows (one result per row).
- **"How do you handle NaN in NumPy?"** → Use `nan*` functions (`np.nanmean`, `np.nansum`, etc.) or filter with `~np.isnan(arr)`, since normal aggregations propagate NaN silently.
- **"Difference between `*` and `@` for matrices?"** → `*` is element-wise multiplication; `@` (or `np.matmul`) is true matrix multiplication.
- **"View vs copy?"** → Slicing returns a view (shares memory with the original); fancy indexing and `.copy()` return independent copies.

---
*Use this alongside your tutorial-based notes for full NumPy coverage before moving deeper into Pandas, SQL, and Power BI interview prep.*