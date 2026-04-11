use std::{
    num::NonZeroU32,
    ops::{Deref, RangeBounds},
};

use wgpu::util::DeviceExt;

use crate::GPUSorter;

#[doc(hidden)]
/// only used for testing
/// temporally used for guessing subgroup size
pub fn upload_to_buffer<T: bytemuck::Pod>(
    encoder: &mut wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    values: &[T],
) {
    let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Staging buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, buffer, 0, staging_buffer.size());
}

#[doc(hidden)]
/// only used for testing
/// temporally used for guessing subgroup size
pub async fn download_buffer<T: Clone + bytemuck::Pod>(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    range: impl RangeBounds<wgpu::BufferAddress>,
) -> Vec<T> {
    // copy buffer data
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Download buffer"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &download_buffer, 0, buffer.size());
    queue.submit([encoder.finish()]);

    // download buffer
    let buffer_slice = download_buffer.slice(range);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    return bytemuck::cast_slice(data.deref()).to_vec();
}

async fn test_sort(sorter: &GPUSorter, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    // simply runs a small sort and check if the sorting result is correct
    let n = 8192; // means that 2 workgroups are needed for sorting
    let scrambled_data: Vec<f32> = (0..n).rev().map(|x| x as f32).collect();
    let sorted_data: Vec<f32> = (0..n).map(|x| x as f32).collect();

    let sort_buffers = sorter.create_sort_buffers(device, NonZeroU32::new(n).unwrap());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });
    upload_to_buffer(
        &mut encoder,
        &sort_buffers.keys(),
        device,
        scrambled_data.as_slice(),
    );

    sorter.sort(&mut encoder, queue, &sort_buffers, None);
    let idx = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();

    let sorted = download_buffer::<f32>(
        &sort_buffers.keys(),
        device,
        queue,
        0..sort_buffers.keys_valid_size(),
    )
    .await;
    return sorted
        .into_iter()
        .zip(sorted_data.into_iter())
        .all(|(a, b)| a == b);
}

/// Function guesses the best subgroup size by testing the sorter with
/// subgroup sizes 1,8,16,32,64,128 and returning the largest subgroup size that worked.
pub async fn guess_workgroup_size(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<u32> {
    let mut cur_sorter: GPUSorter;

    log::debug!("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");

    let mut best = None;
    for subgroup_size in [1, 8, 16, 32, 64, 128] {
        log::debug!("Checking sorting with subgroupsize {}", subgroup_size);

        cur_sorter = GPUSorter::new(device, subgroup_size);
        let sort_success = test_sort(&cur_sorter, device, queue).await;

        log::debug!("{} worked: {}", subgroup_size, sort_success);

        if !sort_success {
            break;
        } else {
            best = Some(subgroup_size)
        }
    }
    return best;
}
