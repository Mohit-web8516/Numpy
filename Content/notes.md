# NumPy Complete Revision Notes
> Source: *Complete Numpy Course in Hindi | Chai aur Numpy* — Hitesh Choudhary (Chai aur Code)
> Video: https://www.youtube.com/watch?v=x7ULDYs4X84
> Code: https://github.com/hiteshchoudhary/Chai-aur-numpy

The course is split into **4 phases**. These notes summarize every topic with explanation + working code so you can revise quickly before interviews.

---

## Table of Contents
- [Phase 1 — NumPy Foundations](#phase-1--numpy-foundations)
  - [1. Creating Arrays from a List](#1-creating-arrays-from-a-list)
  - [2. Python List vs NumPy Array](#2-python-list-vs-numpy-array)
  - [3. Creating Arrays from Scratch](#3-creating-arrays-from-scratch)
  - [4. Vector, Matrix and Tensor](#4-vector-matrix-and-tensor)
  - [5. Array Properties](#5-array-properties)
  - [6. Array Reshaping](#6-array-reshaping)
- [Phase 2 — Array Operations](#phase-2--array-operations)
  - [1. Indexing and Slicing](#1-indexing-and-slicing)
  - [2. Sorting](#2-sorting)
  - [3. Filtering](#3-filtering)
  - [4. Filtering with Boolean Mask](#4-filtering-with-boolean-mask)
  - [5. Fancy Indexing vs np.where()](#5-fancy-indexing-vs-npwhere)
  - [6. Adding and Removing Data (concatenate)](#6-adding-and-removing-data-concatenate)
  - [7. Array Compatibility (Broadcasting Shapes)](#7-array-compatibility-broadcasting-shapes)
  - [8. Stacking — vstack / hstack](#8-stacking--vstack--hstack)
  - [9. Deleting Elements](#9-deleting-elements)
- [Phase 3 — Real World Data / Business Case Study](#phase-3--real-world-data--business-case-study)
  - [1. Structuring Real World Data](#1-structuring-real-world-data)
  - [2. Aggregate Functions (sum, min, max) with axis](#2-aggregate-functions-sum-min-max-with-axis)
  - [3. Cumulative Sum + Visualization](#3-cumulative-sum--visualization-with-matplotlib)
  - [4. Vector Operations (dot product, angle)](#4-vector-operations-dot-product-angle)
  - [5. np.vectorize()](#5-npvectorize)
  - [6. Broadcasting in Business Calculations](#6-broadcasting-in-business-calculations)
- [Phase 4 — Storing Images as Arrays](#phase-4--storing-images-as-arrays)
  - [1. Saving and Loading Arrays (.npy files)](#1-saving-and-loading-arrays-npy-files)
  - [2. Image as a Matrix + Dark Mode Conversion](#2-image-as-a-matrix--dark-mode-conversion)
- [Quick Revision Cheat Sheet](#quick-revision-cheat-sheet)

---

## Phase 1 — NumPy Foundations

### 1. Creating Arrays from a List
NumPy arrays (`ndarray`) are created using `np.array()`. They can be 1D (vector-like) or 2D (matrix-like), unlike plain Python lists which only nest.

```python
import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
print("1D array: ", arr_1d)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array: ", arr_2d)
```

**Key idea:** A NumPy array stores elements of the *same data type* in contiguous memory, which is what makes it fast and different from a Python list.

---

### 2. Python List vs NumPy Array
This is the classic interview point — **why use NumPy over lists?**

```python
py_list = [1, 2, 3]
print("Python list multiplication ", py_list * 2)   # [1,2,3,1,2,3] -> repeats the list

np_array = np.array([1, 2, 3])
print("Python array multiplication ", np_array * 2)  # [2,4,6] -> element-wise multiplication
```

- `list * 2` → **repeats** the list (concatenation), not math.
- `np_array * 2` → performs **element-wise** multiplication (vectorized operation).

**Speed comparison** (also shown in the tutorial using `time.time()`):
```python
import time
start = time.time()
py_list = [i * 2 for i in range(1_000_000)]
print("List operation time:", time.time() - start)

start = time.time()
np_arr = np.arange(1_000_000) * 2
print("NumPy operation time:", time.time() - start)
```
> NumPy is significantly faster for numeric operations because it uses vectorized C-level loops instead of Python-level loops.

---

### 3. Creating Arrays from Scratch
Different helper functions to generate arrays without typing every value:

```python
zeros = np.zeros((3, 4))          # 3x4 array filled with 0
ones = np.ones((2, 3))            # 2x3 array filled with 1
full = np.full((2, 2), 7)         # 2x2 array filled with 7
random = np.random.random((2, 3)) # 2x3 array with random floats [0,1)
sequence = np.arange(0, 11, 2)    # [0,2,4,6,8,10] -> like range() but returns an array
```

| Function | Purpose |
|---|---|
| `np.zeros(shape)` | array of all zeros |
| `np.ones(shape)` | array of all ones |
| `np.full(shape, value)` | array filled with a constant value |
| `np.random.random(shape)` | random floats between 0 and 1 |
| `np.arange(start, stop, step)` | evenly spaced values (like `range()`) |

---

### 4. Vector, Matrix and Tensor
NumPy naturally represents different "ranks" (dimensions) of data:

```python
vector = np.array([1, 2, 3])                  # 1D -> Vector

matrix = np.array([[1, 2, 3],
                    [4, 5, 6]])                # 2D -> Matrix

tensor = np.array([[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]])         # 3D+ -> Tensor
```

| Term | Dimensions | Example use |
|---|---|---|
| Scalar | 0D | a single number |
| Vector | 1D | a list of numbers |
| Matrix | 2D | table/spreadsheet, grayscale image |
| Tensor | 3D+ | color image (H×W×channels), deep learning data |

---

### 5. Array Properties
Every array carries metadata you can inspect:

```python
arr = np.array([[1, 2, 3],
                 [4, 5, 6]])

print("Shape ", arr.shape)   # (2, 3) -> rows x columns
print("Dimension ", arr.ndim) # 2 -> number of axes
print("Size ", arr.size)      # 6 -> total number of elements
print("DType ", arr.dtype)    # int64 (or int32 depending on OS) -> data type of elements
```

| Property | Meaning |
|---|---|
| `.shape` | tuple describing size along each dimension |
| `.ndim` | number of dimensions (axes) |
| `.size` | total number of elements |
| `.dtype` | data type of the elements |

---

### 6. Array Reshaping
Changing the "shape" of data without changing its values.

```python
arr = np.arange(12)
print("Original array", arr)          # [0 1 2 ... 11]

reshaped = arr.reshape((3, 4))         # reshape into 3 rows, 4 cols
print("Reshaped array", reshaped)

flattened = reshaped.flatten()         # returns a NEW COPY as 1D
print("Flattened array", flattened)

raveled = reshaped.ravel()             # returns a VIEW (not a copy) as 1D
print("Raveled array", raveled)

transposed = reshaped.T                # transpose -> swaps rows and columns
```

**Interview tip:**
- `flatten()` → always returns a **copy**. Modifying it does NOT affect the original array.
- `ravel()` → returns a **view** whenever possible. Modifying it CAN affect the original array. Faster, more memory-efficient.
- `.T` → transpose, flips rows ↔ columns.

---

## Phase 2 — Array Operations

### 1. Indexing and Slicing
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Basic Slicing", arr[2:7])        # elements index 2 to 6
print("With Step", arr[1:8:2])          # every 2nd element from index 1 to 7
print("Negative indexing", arr[-3])     # 3rd element from the end

arr_2d = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print("Specific element", arr_2d[1, 2])  # row 1, col 2 -> 6
print("Entire row: ", arr_2d[1])         # [4, 5, 6]
print("Entire column: ", arr_2d[:, 1])   # [2, 5, 8]
```
> Syntax: `array[start:stop:step]`. For 2D arrays: `array[row, column]`.

---

### 2. Sorting
```python
unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("Sorted Array", np.sort(unsorted))   # ascending order

arr_2d_unsorted = np.array([[3, 1], [1, 2], [2, 3]])
print("Sorted 2D array by column", np.sort(arr_2d_unsorted, axis=0))  # sorts each column independently
```
- `axis=0` → sort down each **column**
- `axis=1` → sort across each **row**

---

### 3. Filtering
Selecting elements that satisfy a condition, directly using a boolean condition inside `[]`.

```python
numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
even_number = numbers[numbers % 2 == 0]
print("Even numbers", even_number)   # [2 4 6 8 10]
```

---

### 4. Filtering with Boolean Mask
Same idea as above but the condition is stored in a variable first — a **"mask"**.

```python
mask = numbers > 5
print("Numbers greater than 5", numbers[mask])   # [6 7 8 9 10]
```
> `mask` is a boolean array `[False, False, ..., True, True]`, used to index the original array.

---

### 5. Fancy Indexing vs np.where()
**Fancy indexing** = passing a list/array of indices to select specific elements.
**np.where()** = returns indices where a condition is `True` (and can also act like an if-else).

```python
indices = [0, 2, 4]
print(numbers[indices])            # elements at index 0, 2, 4

where_result = np.where(numbers > 5)
print(where_result)                # (array of indices where condition is True,)
print("NP where", numbers[where_result])

# np.where can also work like a vectorized if-else:
condition_array = np.where(numbers > 5, "true", "false")
print(condition_array)
```
> `np.where(condition, value_if_true, value_if_false)` — very common interview question.

---

### 6. Adding and Removing Data (concatenate)
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

combined = np.concatenate((arr1, arr2))
print(combined)   # [1 2 3 4 5 6]
```

---

### 7. Array Compatibility (Broadcasting Shapes)
Before combining arrays, their **shapes** must be compatible.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])
c = np.array([7, 8, 9])

print("Compatibility shapes", a.shape == b.shape)   # False, a has 3 elements, b has 4
```
> Operations like element-wise addition/multiplication require matching shapes (or shapes that follow NumPy's broadcasting rules).

---

### 8. Stacking — vstack / hstack
```python
original = np.array([[1, 2], [3, 4]])
new_row = np.array([[5, 6]])

with_new_row = np.vstack((original, new_row))   # stack vertically (add a new row)
print(with_new_row)

new_col = np.array([[7], [8]])
with_new_col = np.hstack((original, new_col))   # stack horizontally (add a new column)
print("With new column", with_new_col)
```
| Function | Direction |
|---|---|
| `np.vstack()` | stacks arrays **vertically** (row-wise) |
| `np.hstack()` | stacks arrays **horizontally** (column-wise) |

---

### 9. Deleting Elements
```python
arr = np.array([1, 2, 3, 4, 5])
deleted = np.delete(arr, 2)     # removes element at index 2
print("Array after deletion: ", deleted)   # [1 2 4 5]
```

---

## Phase 3 — Real World Data / Business Case Study
This phase applies NumPy to a **restaurant sales dataset** to simulate real analytics work.

### 1. Structuring Real World Data
```python
import matplotlib.pyplot as plt

# Data structure: [restaurant_id, 2021, 2022, 2023, 2024]
sales_data = np.array([
    [1, 150000, 180000, 220000, 250000],  # Paradise Biryani
    [2, 120000, 140000, 160000, 190000],  # Beijing Bites
    [3, 200000, 230000, 260000, 300000],  # Pizza Hub
    [4, 180000, 210000, 240000, 270000],  # Burger place
])
```
> Real-world tabular data (rows = entities like restaurants, columns = features like yearly sales) maps naturally onto a 2D NumPy array — same mental model as a spreadsheet or a pandas DataFrame.

---

### 2. Aggregate Functions (sum, min, max) with axis
```python
# total sales per year (sum down the rows -> axis=0)
print(np.sum(sales_data, axis=0))

yearly_total = np.sum(sales_data[:, 1:], axis=0)   # ignore restaurant_id column
print(yearly_total)

# Minimum sales per restaurant (across years -> axis=1)
min_sales = np.min(sales_data[:, 1:], axis=1)
print(min_sales)

# Maximum sales per year (axis=0)
max_sales = np.max(sales_data[:, 1:], axis=0)
print(max_sales)
```
**Golden rule for `axis`:**
- `axis=0` → operate **down the columns** (collapses rows, one result per column)
- `axis=1` → operate **across the rows** (collapses columns, one result per row)

---

### 3. Cumulative Sum + Visualization with Matplotlib
```python
cumsum = np.cumsum(sales_data[:, 1:], axis=1)   # running total sales per restaurant over years
print(cumsum)

plt.figure(figsize=(8, 6))
plt.plot(np.mean(cumsum, axis=0))               # average cumulative sales across restaurants
plt.title("Average cumulative sales across all restaurants")
plt.xlabel("Years")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
```
> `np.cumsum()` returns a running total along an axis — useful for trend analysis. Combined with Matplotlib for visualization.

---

### 4. Vector Operations (dot product, angle)
```python
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([6, 7, 8, 9, 10])

print("Vector addition: ", vector1 + vector2)
print("Multiplication vector", vector1 * vector2)   # element-wise
print("Dot Product", np.dot(vector1, vector2))       # sum of element-wise products

angle = np.arccos(
    np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
)
print("Angle between vectors (radians):", angle)
```
- `np.dot()` → dot product (sum of `a[i]*b[i]`), core to linear algebra & ML (e.g. cosine similarity).
- `np.linalg.norm()` → magnitude (length) of a vector.
- Formula used: `cos(θ) = (a · b) / (|a| * |b|)`

---

### 5. np.vectorize()
Applies a normal Python function element-wise to a NumPy array (useful when there's no built-in vectorized version).

```python
restaurant_types = np.array(['biryani', 'chinese', 'pizza', 'burger', 'cafe'])
vectorized_upper = np.vectorize(str.upper)
print("Vectorized Upper", vectorized_upper(restaurant_types))
```
> `str.upper` normally works on a single string; `np.vectorize()` lets it apply to every element of an array at once.

---

### 6. Broadcasting in Business Calculations
```python
monthly_avg = sales_data[:, 1:] / 12   # divides every element by 12 (scalar broadcasting)
print(monthly_avg)
```
> **Broadcasting** = NumPy automatically "stretches" a smaller array (here, the scalar `12`) to match the shape of the larger array during arithmetic, without actually copying data.

---

## Phase 4 — Storing Images as Arrays

### 1. Saving and Loading Arrays (.npy files)
NumPy has its own binary file format `.npy` for fast, exact storage of arrays.

```python
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.random.rand(3, 3)
array3 = np.zeros((4, 4))

np.save('array1.npy', array1)
np.save('array2.npy', array2)
np.save('array3.npy', array3)

loaded_array1 = np.load('array1.npy')
print(loaded_array1)
```
| Function | Purpose |
|---|---|
| `np.save('file.npy', arr)` | saves a single array to disk |
| `np.load('file.npy')` | loads it back exactly (dtype & shape preserved) |

---

### 2. Image as a Matrix + Dark Mode Conversion
A key insight: **an image is just a NumPy array** of pixel values (height × width × color channels). This means image edits are just array operations.

```python
logo = np.load('numpy-logo.npy')

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(logo)
plt.title("Numpy logo")
plt.grid(False)

dark_logo = 1 - logo        # inverts pixel values -> creates a "dark mode" version
plt.subplot(122)
plt.imshow(dark_logo)
plt.title("Numpy Dark logo")
plt.grid(False)

plt.show()
```
> If pixel values are normalized between 0 and 1, `1 - image` flips every pixel's brightness — a simple but powerful demonstration of treating **images as matrices/tensors** and applying vectorized math to them (a core idea behind image processing & computer vision).

---

## Quick Revision Cheat Sheet

| Category | Functions/Concepts |
|---|---|
| **Creation** | `np.array()`, `np.zeros()`, `np.ones()`, `np.full()`, `np.random.random()`, `np.arange()` |
| **Structure types** | Scalar, Vector (1D), Matrix (2D), Tensor (3D+) |
| **Properties** | `.shape`, `.ndim`, `.size`, `.dtype` |
| **Reshaping** | `.reshape()`, `.flatten()` (copy), `.ravel()` (view), `.T` |
| **Indexing/Slicing** | `arr[start:stop:step]`, `arr[row, col]`, negative indexing |
| **Sorting** | `np.sort(arr, axis=0/1)` |
| **Filtering** | boolean condition `arr[arr > x]`, masks, `np.where()` |
| **Fancy indexing** | `arr[[0, 2, 4]]` |
| **Combining** | `np.concatenate()`, `np.vstack()`, `np.hstack()` |
| **Removing** | `np.delete(arr, index)` |
| **Aggregates** | `np.sum()`, `np.min()`, `np.max()`, `np.cumsum()` with `axis` |
| **Vector math** | `np.dot()`, `np.linalg.norm()`, `np.arccos()` |
| **Custom functions** | `np.vectorize()` |
| **Broadcasting** | automatic shape-stretching for scalar/array operations |
| **File I/O** | `np.save()`, `np.load()` (`.npy` format) |
| **Images as arrays** | image = array of pixels → `1 - image` inverts colors |

### axis rule (most confused interview topic)
- `axis=0` → down the **columns** (result has one value per column)
- `axis=1` → across the **rows** (result has one value per row)

---
*Notes generated for revision — pair with the original notebooks (`phase-1.ipynb` → `phase4.ipynb`) in the [GitHub repo](https://github.com/hiteshchoudhary/Chai-aur-numpy) for hands-on practice.*