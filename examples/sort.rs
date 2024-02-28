// this example creates an array with 10 key-value (u32,f32) pairs and sorts them on the gpu
use std::num::NonZeroU32;

use wgpu_sort::{utils::{download_buffer, guess_workgroup_size, upload_to_buffer}, GPUSorter};


#[pollster::main]
async fn main(){
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
    let subgroup_size = guess_workgroup_size(&device, &queue).await.expect("could not find a valid subgroup size");
    println!("using subgroup size {subgroup_size}");
    let sorter = GPUSorter::new(&device, subgroup_size);

    let n = 10;
    let sort_buffers = sorter.create_sort_buffers(&device, NonZeroU32::new(n).unwrap());


    let keys_scrambled: Vec<u32> = (0..n).rev().collect();

    let values_scrambled:Vec<f32> = keys_scrambled.iter().map(|v|1./(*v as f32)).collect();


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
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

    println!("before: {:?}",keys_scrambled.iter().zip(values_scrambled.iter()).collect::<Vec<(_,_)>>());

    // sorter.sort(&mut encoder, &sort_buffers);
    sorter.sort(&mut encoder,&queue,&sort_buffers,None);

    // wait for sorter to finish
    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));

    // keys buffer has padding at the end
    // so we only download the "valid" data
    let keys_sorted:Vec<u32> = download_buffer::<u32>(
        &sort_buffers.keys(),
        &device,
        &queue,
        0..sort_buffers.keys_valid_size(),
    )
    .await;
    let value_sorted = download_buffer::<f32>(
        &sort_buffers.values(),
        &device,
        &queue,
        ..,
    )
    .await;

    println!("after: {:?}",keys_sorted.iter().zip(value_sorted.iter()).collect::<Vec<(_,_)>>());
}

