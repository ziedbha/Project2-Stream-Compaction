CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ziad Ben Hadj-Alouane
  * [LinkedIn](https://www.linkedin.com/in/ziadbha/), [personal website](https://www.seas.upenn.edu/~ziadb/)
* Tested on: Windows 10, i7-8750H @ 2.20GHz, 16GB, GTX 1060

# Project Goal
In this project, I implemented 2 notable parallel algorithms: Scan (Prefix Sum) & Stream Compaction. The latter algorithm is widely used as a building block for other parallel algorithms. In fact, stream compaction is used in path tracing, a project I intend to implement soon.

# Features
## Scan
### Input
A stream of ints of predetermined length

### Output
A stream of ints `out` such that for `out[idx]`, we store the running sum of the previous elements. For *exclusive* scan, we exclude the element at `input[idx]`. Otherwise, we have an *inclusive* scan.

### GPU Implementation
#### Naive
<p align="center">
  <img width="400" height="400" src="https://github.com/ziedbha/Project2-Stream-Compaction/blob/master/images/naiveScanCompaction.gif"/>
</p>

This algorithm is based on the scan algorithm presented by Hillis and Steele (1986). However, this algorithm performs **O(n log2 n)** addition operations. In contrast, the sequential scan performs **O(n)** adds. Therefore, the naive implementation is not work-efficient. 

#### Work-Efficient
This algorithm has 2 phases: an upsweep (reduction) phase, and a downsweep phase. Both of these phases are implemented as separate kernel functions in CUDA, with each outlined level designating a kernel call of the corresponding phase. We run upsweep on the entire array until we hit all **log(n)** levels, then we run downsweep on the output of upsweep. The result is an exclusive scan of the original input.

##### Upsweep
<p align="center">
  <img width="400" height="400" src="https://github.com/ziedbha/Project2-Stream-Compaction/blob/master/images/efficientUpsweep.gif"/>
</p>

##### Downsweep
<p align="center">
  <img width="400" height="400" src="https://github.com/ziedbha/Project2-Stream-Compaction/blob/master/images/efficientDownsweep.gif"/>
</p>

## Compact
<p align="center">
  <img width="400" height="400" src="https://github.com/ziedbha/Project2-Stream-Compaction/blob/master/images/streamCompaction.gif"/>
</p>

### Input
A stream of ints of predetermined length

### Output
A stream of ints equal to the input, but with all 0s removed

### GPU Implementation
Using scan as a building block, we can implement a parallel version of compact by going through some extra steps:
1. We obtain a boolean array version of the input array (`True` if `input[idx != 0]`, else `False`)
2. We Scan the boolean array
3. We use Scatter, an algorithm that fills an output array using the following condition: if `bool[idx] == 1` then `out[scan[idx]] = input[idx]`

For reference, I used Nvidia's [GPU Gems](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html) to implement this algorithm.

Add your performance analysis (see below).
