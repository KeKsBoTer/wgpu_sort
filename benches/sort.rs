use std::{num::NonZeroU32, time::Duration};

use wgpu_sort::{utils::{download_buffer, guess_workgroup_size}, GPUSorter, SortBuffers};

struct SortStuff{
    device:wgpu::Device,
    queue:wgpu::Queue,
    query_set:wgpu::QuerySet,
    query_buffer:wgpu::Buffer,
}

async fn setup()-> SortStuff{
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits{
                    max_buffer_size:1<<30,
                    max_storage_buffer_binding_size:1<<30,
                    ..Default::default()
                },
                label: None,
            },
            None,
        )
        .await
        .unwrap();

    let capacity = 2;
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("time stamp query set"),
        ty: wgpu::QueryType::Timestamp,
        count: capacity,
    });


    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("query set buffer"),
        size: capacity as u64 * std::mem::size_of::<u64>() as u64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    return SortStuff{device,queue,query_set,query_buffer}

}

async fn sort(context:&SortStuff,sorter:&GPUSorter,buffers:&SortBuffers,n:u32,iters:u32) -> Duration {

    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
    });

    encoder.write_timestamp(&context.query_set, 0);

    for _ in 0..iters{
        sorter.sort(&mut encoder,&context.queue,buffers,Some(n));
    }

    encoder.write_timestamp(&context.query_set, 1);
    encoder.resolve_query_set(
        &context.query_set,
        0..2,
        &context.query_buffer,
        0,
    );
    let idx = context.queue.submit([encoder.finish()]);
    context.device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));

    let timestamps : Vec<u64> = pollster::block_on(download_buffer(&context.query_buffer, &context.device, &context.queue, ..));
    let diff_ticks = timestamps[1] - timestamps[0];
    let period = context.queue.get_timestamp_period();
    let diff_time = Duration::from_nanos((diff_ticks as f32 * period / iters as f32) as u64);

    return diff_time;
}



#[pollster::main]
async fn main() {

    let context = setup().await;

    let subgroup_size = guess_workgroup_size(&context.device, &context.queue).await.expect("could not find a valid subgroup size");

    let sorter = GPUSorter::new(&context.device, subgroup_size);


    for n in [10_000,100_000,1_000_000,8_000_000,20_000_000]{
        let buffers = sorter.create_sort_buffers(&context.device, NonZeroU32::new(n).unwrap());
        let d = sort(&context,&sorter, &buffers,n,10000).await;
        println!("{n}: {d:?}");
    }
  }
  
