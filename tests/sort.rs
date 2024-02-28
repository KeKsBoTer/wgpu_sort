use std::{fmt::Debug, num::NonZeroU32};

use bytemuck::bytes_of;
use float_ord::FloatOrd;
use rand::{
    distributions::{Distribution, Standard},
    rngs::StdRng,
    Rng, SeedableRng,
};
use wgpu::util::DeviceExt;
use wgpu_sort::{
    utils::{download_buffer, guess_workgroup_size, upload_to_buffer},
    GPUSorter, SortBuffers, HISTO_BLOCK_KVS,
};


/// tests sorting of two u32 keys
#[pollster::test]
async fn sort_u32_small() {
    test_sort::<u32>(2,&apply_sort,None).await;
}

/// tests sorting of one million pairs with u32 keys
#[pollster::test]
async fn sort_u32_large() {
    test_sort::<u32>(1_000_00,&apply_sort,None).await;
}

/// tests sorting of one million pairs with f32 keys
#[pollster::test]
async fn sort_f32_large() {
    test_sort::<Float>(1_000_00,&apply_sort,None).await;
}

/// tests sorting only first half of one million pairs
#[pollster::test]
async fn sort_half() {
    test_sort::<u32>(1_000_000,&apply_sort,Some(500_00)).await;
}

// INDIRECT SORTING


/// tests sorting of two u32 keys
/// indirect dispatch
#[pollster::test]
async fn sort_indirect_small() {
    test_sort::<u32>(2,&apply_sort_indirect,None).await;
}

/// tests sorting of one million pairs with u32 keys
/// indirect dispatch
#[pollster::test]
async fn sort_indirect_large() {
    test_sort::<u32>(1_000_00,&apply_sort,None).await;
}


/// tests sorting only first half of one million pairs
/// indirect dispatch
#[pollster::test]
async fn sort_indirect_half() {
    test_sort::<u32>(1_000_000,&apply_sort_indirect,Some(500_00)).await;
}



async fn setup() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        )
        .await
        .unwrap();

    return (device, queue);
}

type SortFn = dyn Fn(&mut wgpu::CommandEncoder,&wgpu::Device,&wgpu::Queue,&GPUSorter,&SortBuffers,Option<u32>)->();


/// applies gpu sort with direct dispatch
fn apply_sort(encoder:&mut wgpu::CommandEncoder,_device:&wgpu::Device,queue:&wgpu::Queue,sorter:&GPUSorter,sort_buffers:&SortBuffers,n:Option<u32>){
    sorter.sort(encoder, queue,&sort_buffers,n);
}


/// applies gpu sort with indirect dispatch
fn apply_sort_indirect(encoder:&mut wgpu::CommandEncoder,device:&wgpu::Device,queue:&wgpu::Queue,sorter:&GPUSorter,sort_buffers:&SortBuffers,n:Option<u32>){

    // round to next larger multiple of HISTO_BLOCK_KVS
    let nelm = n.unwrap_or(sort_buffers.len());
    let num_wg = (nelm + HISTO_BLOCK_KVS- 1)/HISTO_BLOCK_KVS;

    let dispatch_indirect = wgpu::util::DispatchIndirectArgs{
        x: num_wg,
        y: 1,
        z: 1
    };

    queue.write_buffer(sort_buffers.state_buffer(), 0, bytes_of(&nelm));

    let dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: Some("dispatch indirect buffer"),
        contents: dispatch_indirect.as_bytes(),
        usage: wgpu::BufferUsages::INDIRECT,
    });

    sorter.sort_indirect(encoder, &sort_buffers,&dispatch_buffer);
}

async fn test_sort<T>(n: u32,sort_fn:&SortFn,sort_first_n:Option<u32>)
where
    Standard: Distribution<T>,
    T: PartialEq + Clone + Copy + Debug + bytemuck::Pod + Ord
{
    let (device, queue) = setup().await;
    let subgroup_size = guess_workgroup_size(&device, &queue).await;
    assert_ne!(subgroup_size, None);
    let sorter = GPUSorter::new(&device, subgroup_size.unwrap());

    let sort_buffers = sorter.create_sort_buffers(&device, NonZeroU32::new(n).unwrap());
    let n_sorted = sort_first_n.unwrap_or(sort_buffers.len());


    let mut rng = StdRng::seed_from_u64(0);
    let keys_scrambled: Vec<T> = (0..n).map(|_| rng.gen()).collect();
    let mut keys_sorted = keys_scrambled.clone();
    keys_sorted[0..n_sorted as usize].sort();


    let values_scrambled = keys_scrambled.clone();
    let values_sorted = keys_sorted.clone();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });

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
    sort_fn(&mut encoder,&device,&queue,&sorter,&sort_buffers,sort_first_n);

    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));

    let keys_sorted_gpu = download_buffer::<T>(
        &sort_buffers.keys(),
        &device,
        &queue,
        0..sort_buffers.keys_valid_size(),
    )
    .await;
    assert_eq!(
        keys_sorted_gpu[0..n_sorted as usize], keys_sorted[0..n_sorted as usize],
        "GPU keys equal to keys sorted on CPU"
    );

    let values_sorted_gpu = download_buffer::<T>(&sort_buffers.values(), &device, &queue, ..).await;
    assert_eq!(
        values_sorted_gpu[0..n_sorted as usize], values_sorted[0..n_sorted as usize],
        "GPU values equal to values sorted on CPU"
    );
}


// ordered float
#[repr(C)]
#[derive(PartialEq,Debug,Clone, Copy,bytemuck::Pod,bytemuck::Zeroable)]
struct Float(f32);

impl Eq for Float{}

impl Ord for Float{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        FloatOrd(self.0).cmp(&FloatOrd(other.0))
    }
}

impl PartialOrd for Float{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Distribution<Float> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Float {
        Float(rng.gen())
    }
}