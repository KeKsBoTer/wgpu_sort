# WebGPU Radix Key-Value Sort

This package implements a GPU version of radix sort. A good introduction to general purpose radix sort can be found here: <http://www.codercorner.com/RadixSortRevisited.htm>

The GPU radix sort implemented here is a re-implementation of the Vulkan radix sort found in the fuchsia repos: <https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/>.

Currently only the sorting for 32-bit key-value pairs is implemented.
It can be used to sort unsigned integers and non negative float numbers. See [Limitations](#limitations) for more details.
The keys are sorted in ascending order.

It was originally implemented for [our 3D Gaussian Splatting Renderer](https://github.com/KeKsBoTer/web-splat) to sort splats according to their depth in real time. It can be seen in action in this [web demo](https://keksboter.github.io/web-splat/demo.html).

## Example

```rust,ignore
// find best subgroup size
let subgroup_size = guess_workgroup_size(&device, &queue).await.unwrap();
let sorter = GPUSorter::new(&device, subgroup_size);

// setup buffers to sort 100 key-value pairs
let n = 100;
let sort_buffers = sorter.create_sort_buffers(&device, NonZeroU32::new(n).unwrap());

let keys_scrambled: Vec<u32> = (0..n).rev().collect();
let values_scrambled:Vec<u32> = keys_scrambled.clone();

let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: None});

upload_to_buffer(
    &mut encoder,
    &sort_buffers.keys(),
    &device,
    keys_scrambled.as_slice(),
);
upload_to_buffer(
    &mut encoder,
    &sort_buffers.values(),
    &device,
    values_scrambled.as_slice(),
);

// sorter.sort(&mut encoder, &sort_buffers);
sorter.sort(&mut encoder,&queue,&sort_buffers,None);
queue.submit([encoder.finish()]);

// key and value buffer is now sorted.
```
Indirect dispatching is also supported. See [examples/sort_indirect.rs](examples/sort_indirect.rs);

## Benchmarks

To measure the performance we sort the key-value pairs 1000 times and report the average duration per run.
Measurements were performed for different number of pairs.
Take a look at [benches/sort.rs](benches/sort.rs) for more details.

| Device                 | 10k       | 100k      | 1 Million | 8 Million  | 20 Million |
|------------------------|-----------|-----------|-----------|------------|------------|
| NVIDIA RTX A5000       | 108.277µs | 110.179µs | 317.191µs | 1.641699ms | 3.980834ms |
| AMD Radeon R9 380      | 803.527µs | 829.003µs | 2.76469ms | 18.81558ms | 46.12854ms |
| Intel HD Graphics 4600 | 790.382µs | 4.12287ms | 38.7421ms | 295.2937ms | 732.3900ms |

## Limitations

This sorter comes with a number of limitations that are explained in the following.

**Subgroups**

This renderer makes use of [subgroups](https://docs.vulkan.org/guide/latest/subgroups.html) to reduce synchronization and increase performance.
Unfortunately subgroup operations are not supported bei WebGPU/wgpu right now.

To overcome this issue we "guess" the subgroup size by trying out different subgroups and pick the largest one that works (see [utils::guess_workgroup_size](src/utils.rs)). 
This works in almost all cases but can fail because the subgroup size can change over time.
Once subgroups are support this will be fixed. 
Status can be found [here](https://github.com/gpuweb/gpuweb/issues/4306).

**Floating Point Numbers**

The sorting algorithm interprets the values as integers and sorts the keys in ascending order. 
Non-negative float values can be interpreted as unsigned integers without affecting the ordering.
Therefore this sorter can be used to sort 32-bit float keys.
Note that NaN and Inf values lead to unexpected results as theses are interpreted as integers as well.
An example for sorting float values can be found [here](examples/sort_indirect.rs).

