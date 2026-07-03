// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include <algorithm>

bool contains_element(uint32_t* arr, uint32_t size, uint32_t val) {
    return std::find(arr, arr + size, val) != arr + size;
}

void kernel_main() {
    // Compile-time args
    constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t index_total_size = get_compile_time_arg_val(1);
    constexpr uint32_t index_size = get_compile_time_arg_val(2);
    constexpr bool is_last_dim = get_compile_time_arg_val(3) == 1;
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr auto index_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Run-time args
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t index_buffer_address = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);
    uint32_t end_row = get_arg_val<uint32_t>(3);
    uint32_t num_rows_in_dim = get_arg_val<uint32_t>(4);
    uint32_t dim_size = get_arg_val<uint32_t>(5);

    // Derived
    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t index_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t fill_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t onepage = 1;

    const auto s0 = TensorAccessor(input_args, input_buffer_address);
    const auto s1 = TensorAccessor(index_args, index_buffer_address);

    Noc noc;
    CircularBuffer cb_src(src_cb_id);
    CircularBuffer cb_index(index_cb_id);

    // Read the entire index tensor into L1
    cb_index.reserve_back(onepage);
    uint32_t index_addr = cb_index.get_write_ptr();
    noc.async_read(s1, CoreLocalMem<uint32_t>(index_addr), index_total_size, {.page_id = 0}, {});
    noc.async_read_barrier();
    cb_index.push_back(onepage);
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_addr);

    // Read input tensor pages
    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        // Performance optimization: use pre-filled page instead of input page
        bool use_filled_page = false;
        if constexpr (!is_last_dim) {
            uint32_t dim_index = (row_id / num_rows_in_dim) % dim_size;
            use_filled_page = contains_element(index_ptr, index_size, dim_index);
        }

        if (!use_filled_page) {
            // Read input page
            cb_src.reserve_back(onepage);
            uint32_t input_addr = cb_src.get_write_ptr();
            noc.async_read(s0, CoreLocalMem<uint32_t>(input_addr), input_page_size, {.page_id = row_id}, {});
            noc.async_read_barrier();
            cb_src.push_back(onepage);
        }
    }
}
