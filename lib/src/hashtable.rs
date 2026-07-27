// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::compress::MetablockData;
use crate::constants::*;
use bounded_utils::safe_x86_64;
use bounded_utils::{BoundedSlice, BoundedU32, BoundedU8, BoundedUsize};
use hugepage_buffer::BoxedHugePageArray;
use std::arch::x86_64::*;
use zerocopy::FromZeroes;

const LOG_TABLE_SIZE: usize = 16;
const PREFETCH_OFFSET: usize = 4;
const LEN_MULT: i32 = 129;
const GAIN_OFF: i32 = 177;
const DIST_SHIFT: i32 = 5;
const GAIN_FOR_LAZY: i32 = 77;
const LD_MAXDIFF: i32 = 3;
const LD_OFF: i32 = 150;

const INTERIOR_MARGIN: usize = 32;

#[inline]
fn fill_entry_inner<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const TABLE_SIZE_MINUS_ONE: usize,
>(
    pos: usize,
    secondary_hash: u8,
    table: &mut HashTableEntry<ENTRY_SIZE, TABLE_SIZE_MINUS_ONE>,
    ridx: &mut BoundedU8<ENTRY_SIZE>,
) {
    let idx = if let Some(idx) = ridx.sub::<ENTRY_SIZE_MINUS_ONE, 1>() {
        idx
    } else {
        for i in 0..ENTRY_SIZE {
            table.pos[i] = (pos as u32).wrapping_sub(WSIZE as u32);
        }
        BoundedU8::constant::<0>()
    };
    *ridx = idx.mod_add(1).add::<ENTRY_SIZE, 1>();
    *BoundedSlice::new_from_equal_array_mut(&mut table.pos).get_mut(idx.into()) = pos as u32;
    table.secondary_hash[usize::from(idx.get())] = secondary_hash;

    // debug_assert!(
    //     pos < ENTRY_SIZE,
    //     "pos was {} but ENTRY_SIZE is {}",
    //     pos,
    //     ENTRY_SIZE
    // );
}

#[derive(Clone, Copy, FromZeroes)]
#[repr(C, align(32))]
struct HashTableEntry<const ENTRY_SIZE: usize, const TABLE_SIZE_MINUS_ONE: usize> {
    pos: [u32; ENTRY_SIZE],
    /// Secondary hash of the same data, used to limit conflicts
    secondary_hash: [u8; ENTRY_SIZE],
}

#[inline]
#[target_feature(enable = "avx,avx2")]
fn longest_match(data: &[u8], pos1: u32, pos2: usize) -> usize {
    let pos1 = pos1 as usize;
    let max = (data.len() - pos2.max(pos1) - INTERIOR_MARGIN).min(MAX_COPY_LEN);

    let d1 = &data[pos1..pos1 + max];
    let d2 = &data[pos2..pos2 + max];

    let mut i = 0;
    while i + 64 <= max {
        // The compiler knows i + 64 <= d1.len() because d1.len() == max.
        // So the slicing below has no bounds checks.
        let slice1 = BoundedSlice::<_, 64>::new(&d1[i..i + 64]).unwrap();
        let slice2 = BoundedSlice::<_, 64>::new(&d2[i..i + 64]).unwrap();

        let data1a = safe_x86_64::_mm256_load(slice1, BoundedUsize::<0>::MAX);
        let data2a = safe_x86_64::_mm256_load(slice2, BoundedUsize::<0>::MAX);
        let data1b = safe_x86_64::_mm256_load(slice1, BoundedUsize::<32>::MAX);
        let data2b = safe_x86_64::_mm256_load(slice2, BoundedUsize::<32>::MAX);

        let maska = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data1a, data2a)) as u32);
        let maskb = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data1b, data2b)) as u32);
        if maska != 0 {
            return i + maska.trailing_zeros() as usize;
        }
        if maskb != 0 {
            return i + 32 + maskb.trailing_zeros() as usize;
        }
        i += 64;
    }
    while i < max {
        if d1[i] != d2[i] {
            return i;
        }
        i += 1;
    }
    max
}

#[inline]
fn gain_from_len_and_dist<const USE_LAST_DISTANCES: bool>(
    len: u32,
    dist: u32,
    last_distances: [u32; 2],
) -> i32 {
    let distance_penalty = (dist.checked_ilog2().unwrap_or(0) << DIST_SHIFT) as i32 + GAIN_OFF;
    LEN_MULT * len as i32
        - if USE_LAST_DISTANCES
            && last_distances
                .into_iter()
                .any(|ld| (ld as i32 - dist as i32).abs() < LD_MAXDIFF)
        {
            LD_OFF
        } else {
            distance_penalty
        }
}

#[inline]
#[target_feature(enable = "avx,avx2")]
fn _mm256_ilog2_epi32(x: __m256i) -> __m256i {
    let float = _mm256_castps_si256(_mm256_cvtepi32_ps(x));
    _mm256_sub_epi32(_mm256_srli_epi32::<23>(float), _mm256_set1_epi32(127))
}

#[cfg(feature = "hash_stats")]
thread_local! {
    pub static TOTAL_SEARCH_STEPS: std::cell::Cell<u64> = std::cell::Cell::new(0);
    pub static FALSE_POSITIVES: std::cell::Cell<u64> = std::cell::Cell::new(0);
    pub static SKIPPED_WITH_SECOND_HASH: std::cell::Cell<u64> = std::cell::Cell::new(0);
    pub static MATCH_FOUND: std::cell::Cell<u64> = std::cell::Cell::new(0);
}

#[cfg(feature = "hash_stats")]
pub fn print_stats() {
    let total = TOTAL_SEARCH_STEPS.with(|c| c.get());
    let false_pos = FALSE_POSITIVES.with(|c| c.get());
    let matches = MATCH_FOUND.with(|c| c.get());
    let skipped_with_second_hash = SKIPPED_WITH_SECOND_HASH.with(|c| c.get());

    println!("\n--- Match Finder Stats ---");
    println!("Total hash matches (iterations): {}", total);
    println!("False Positives (Hash match, but 0 length): {}", false_pos);
    println!(
        "Skipped conflicts using second hash: {}",
        skipped_with_second_hash
    );
    println!("Matches Found (> 0 length): {}", matches);
    if total > 0 {
        println!(
            "False Positive Rate: {:.2}%",
            (false_pos as f64 / total as f64) * 100.0
        );
    }
}

#[cfg(not(feature = "hash_stats"))]
pub fn print_stats() {}

#[inline]
#[target_feature(enable = "avx,avx2")]
fn table_search<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const TABLE_SIZE_MINUS_ONE: usize,
    const USE_LAST_DISTANCES: bool,
>(
    data: &[u8],
    pos: usize,
    secondary_hash: u8,
    table: &mut HashTableEntry<ENTRY_SIZE, TABLE_SIZE_MINUS_ONE>,
    last_distances: [u32; 2],
) -> (u32, u32, i32) {
    let mut best_distance = 0;
    let mut best_len = 0;
    let mut best_gain = 0;

    let mut mask = if ENTRY_SIZE <= 16 {
        let target_sec = _mm_set1_epi8(secondary_hash as i8);
        let sec_bytes = safe_x86_64::_mm_load_u8_array(&table.secondary_hash);
        let matches = _mm_cmpeq_epi8(target_sec, sec_bytes);
        let mask16 = _mm_movemask_epi8(matches) as u32;
        mask16 & ((1u32 << ENTRY_SIZE) - 1)
    } else {
        let target_sec = _mm256_set1_epi8(secondary_hash as i8);
        let sec_bytes = safe_x86_64::_mm256_load_u8_array(&table.secondary_hash);
        let matches = _mm256_cmpeq_epi8(target_sec, sec_bytes);
        _mm256_movemask_epi8(matches) as u32
    };

    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        mask &= mask - 1;

        #[cfg(feature = "hash_stats")]
        TOTAL_SEARCH_STEPS.with(|c| c.set(c.get() + 1));

        let hpos = table.pos[i];
        // This means that the table entry has not been filled yet or it's too old.
        if pos as u32 <= hpos || pos as u32 - hpos > WSIZE as u32 {
            continue;
        }

        let dist = pos as u32 - hpos;
        let len = longest_match(data, hpos, pos) as u32;

        #[cfg(feature = "hash_stats")]
        {
            if len == 0 {
                FALSE_POSITIVES.with(|c| c.set(c.get() + 1));
            } else {
                MATCH_FOUND.with(|c| c.set(c.get() + 1));
            }
        }

        let gain = gain_from_len_and_dist::<USE_LAST_DISTANCES>(len, dist, last_distances);
        if gain > best_gain {
            best_gain = gain;
            best_len = len;
            best_distance = dist;
        }
    }
    (best_distance, best_len, best_gain)
}

const PRECOMPUTE_SIZE: usize = 16;

#[inline]
#[target_feature(enable = "sse2,ssse3,sse4.1,avx,avx2")]
fn compute_hashes_at(
    data_slice: &BoundedSlice<u8, INTERIOR_MARGIN>,
    hashes1: &mut [BoundedU32<{ TABLE_SIZE - 1 }>; PRECOMPUTE_SIZE],
    hashes2: &mut [BoundedU8<255>; 16],
) {
    const _: () = assert!(PRECOMPUTE_SIZE == 16);
    let hash_mul_1 = _mm256_set1_epi32(0x1E35A7BD);
    let hash_mul_2 = _mm256_set1_epi32(0x5BD1E995);
    let d08 = safe_x86_64::_mm256_load(data_slice, BoundedUsize::<0>::MAX);
    let d0 = _mm256_permute4x64_epi64::<0b01000100>(d08);
    let d8 = _mm256_permute4x64_epi64::<0b10011001>(d08);

    #[rustfmt::skip]
    let shufmask = _mm256_setr_epi8(
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7,
        5, 6, 7, 8,
        6, 7, 8, 9,
        7, 8, 9, 10,
    );

    let data0 = _mm256_shuffle_epi8(d0, shufmask);
    let data1 = _mm256_shuffle_epi8(d8, shufmask);

    const SHIFT: i32 = 32 - LOG_TABLE_SIZE as i32;

    let h1_0 = _mm256_srli_epi32::<SHIFT>(_mm256_mullo_epi32(data0, hash_mul_1));
    let h1_1 = _mm256_srli_epi32::<SHIFT>(_mm256_mullo_epi32(data1, hash_mul_1));

    let h2_0 = _mm256_srli_epi32::<24>(_mm256_mullo_epi32(data0, hash_mul_2));
    let h2_1 = _mm256_srli_epi32::<24>(_mm256_mullo_epi32(data1, hash_mul_2));

    {
        let hashes = BoundedSlice::new_from_equal_array_mut(hashes1);
        safe_x86_64::_mm256_store_masked_u32(hashes, BoundedUsize::<0>::MAX, h1_0);
        safe_x86_64::_mm256_store_masked_u32(hashes, BoundedUsize::<8>::MAX, h1_1);
    }
    {
        // Combine the two 256-bit registers (each containing 8 u32 hashes in the low byte of each lane)
        // into a single 128-bit register containing all 16 u8 hashes.
        let h2_0_128 = _mm256_castsi256_si128(_mm256_packus_epi32(h2_0, h2_0));
        let h2_1_128 = _mm256_castsi256_si128(_mm256_packus_epi32(h2_1, h2_1));
        let h2_packed = _mm_packus_epi16(h2_0_128, h2_1_128);

        let hashes = BoundedSlice::new_from_equal_array_mut(hashes2);
        safe_x86_64::_mm_store_masked_u8(hashes, BoundedUsize::<0>::MAX, h2_packed);
    }

    if cfg!(debug_assertions) {
        for (i, h) in hashes1.iter().enumerate() {
            let data = u32::from_le_bytes(
                *data_slice.get_array(BoundedUsize::<PRECOMPUTE_SIZE>::new(i).unwrap()),
            );
            let basic_hash = data.wrapping_mul(0x1E35A7BD) >> (32 - LOG_TABLE_SIZE);
            debug_assert_eq!(h.get(), basic_hash);
        }
    }

    if cfg!(debug_assertions) {
        for (i, h) in hashes2.iter().enumerate() {
            let data = u32::from_le_bytes(
                *data_slice.get_array(BoundedUsize::<PRECOMPUTE_SIZE>::new(i).unwrap()),
            );
            let basic_hash = (data.wrapping_mul(0x5BD1E995) >> 24) as u8;
            debug_assert_eq!(h.get(), basic_hash);
        }
    }
}

const TABLE_SIZE: usize = 1 << LOG_TABLE_SIZE;
const TABLE_SIZE_MINUS_ONE: usize = TABLE_SIZE - 1;

pub struct HashTable<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const ENTRY_SIZE_MINUS_EIGHT: usize,
> {
    table: BoxedHugePageArray<HashTableEntry<ENTRY_SIZE, TABLE_SIZE_MINUS_ONE>, TABLE_SIZE>,
    replacement_idx: BoxedHugePageArray<BoundedU8<ENTRY_SIZE>, TABLE_SIZE>,
}

impl<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
    > HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>
{
    pub fn new() -> Self {
        HashTable {
            table: BoxedHugePageArray::new_zeroed(),
            replacement_idx: BoxedHugePageArray::new_zeroed(),
        }
    }

    pub fn clear(&mut self) {
        self.replacement_idx.fill(BoundedU8::constant::<0>());
    }

    pub fn shift_back(&mut self, amount: u32) {
        for entry in self.table.iter_mut() {
            for i in 0..ENTRY_SIZE {
                entry.pos[i] = entry.pos[i].saturating_sub(amount);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "sse")]
    fn prefetch_pos(&self, pos: BoundedUsize<{ TABLE_SIZE - 1 }>) {
        let entry = BoundedSlice::new_from_equal_array(&self.table).get(pos);
        let ridx = BoundedSlice::new_from_equal_array(&self.replacement_idx).get(pos);
        safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(entry);
        safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(ridx);
    }

    /// Returns the number of bytes that were written to the output. Updates the hash table with
    /// strings starting at all of those bytes, if within the margin.
    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
    #[inline(never)]
    fn parse_and_emit_interior<const MIN_GAIN_FOR_GREEDY: i32, const USE_LAST_DISTANCES: bool>(
        &mut self,
        data: &[u8],
        start: usize,
        count: usize,
        metablock_data: &mut MetablockData,
    ) -> usize {
        let end_upper_bound = data.len().saturating_sub(INTERIOR_MARGIN - 1);
        let end = end_upper_bound.min(count + start);
        if end <= start {
            return 0;
        }

        let mut primary_hashes = [BoundedU32::constant::<0>(); PRECOMPUTE_SIZE];
        let mut secondary_hashes = [BoundedU8::constant::<0>(); 16];

        let mut last_dist = 0;
        let mut last_len = 0;
        let mut last_gain = 0;
        let mut last_lit = 0;
        let mut has_lazy = false;

        let mut last_distances = [0; 2];

        let mut skip = 0;
        for pos in start..end {
            let data_slice =
                BoundedSlice::<_, { INTERIOR_MARGIN }>::new_at_offset(data, pos).unwrap();

            let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
            if po.get() == 0 {
                compute_hashes_at(data_slice, &mut primary_hashes, &mut secondary_hashes);
            }

            self.prefetch_pos(
                (*BoundedSlice::new_from_equal_array(&primary_hashes)
                    .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>()))
                .into(),
            );

            let hash = (*BoundedSlice::new_from_equal_array(&primary_hashes).get(po)).into();
            let secondary_hash =
                (*BoundedSlice::new_from_equal_array(&secondary_hashes).get(po)).get();
            let table = BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
            let replacement_idx =
                BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx).get_mut(hash);

            if skip == 0 {
                let (dist, len, gain) = if replacement_idx.get() == 0 {
                    (0, 0, 0)
                } else {
                    table_search::<
                        ENTRY_SIZE,
                        ENTRY_SIZE_MINUS_ONE,
                        TABLE_SIZE_MINUS_ONE,
                        USE_LAST_DISTANCES,
                    >(data, pos, secondary_hash, table, last_distances)
                };
                let lit = *data_slice.get(BoundedUsize::<0>::MAX);

                let (lit_params, copy_params) = if has_lazy && gain <= last_gain + GAIN_FOR_LAZY {
                    let val = ((0, false), (last_len, last_dist, true));
                    skip = last_len - 2;
                    has_lazy = false;
                    val
                } else if gain > MIN_GAIN_FOR_GREEDY {
                    let val = ((last_lit, has_lazy), (len, dist, true));
                    skip = len - 1;
                    has_lazy = false;
                    val
                } else if len >= 4 {
                    let val = ((last_lit, has_lazy), (0, 0, false));
                    last_lit = lit;
                    last_dist = dist;
                    last_len = len;
                    last_gain = gain;
                    has_lazy = true;
                    val
                } else {
                    debug_assert!(!has_lazy);
                    ((lit, true), (0, 0, false))
                };
                metablock_data.add_literal(lit_params.0, lit_params.1);
                metablock_data.add_copy(copy_params.0, copy_params.1, copy_params.2);
                if USE_LAST_DISTANCES {
                    last_distances = if copy_params.2 {
                        [copy_params.1, last_distances[0]]
                    } else {
                        last_distances
                    };
                }
            } else {
                skip -= 1;
            }
            fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, TABLE_SIZE_MINUS_ONE>(
                pos,
                secondary_hash,
                table,
                replacement_idx,
            );
        }

        if has_lazy {
            metablock_data.add_copy(last_len, last_dist, true);
            skip = last_len - 1;
        }

        // Populate the hash table with the remaining copied bytes.
        let skip_end = end_upper_bound.min(end + skip as usize);
        for pos in end..skip_end {
            let data_slice =
                BoundedSlice::<_, { INTERIOR_MARGIN }>::new_at_offset(data, pos).unwrap();

            let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
            if po.get() == 0 {
                compute_hashes_at(data_slice, &mut primary_hashes, &mut secondary_hashes);
            }

            self.prefetch_pos(
                (*BoundedSlice::new_from_equal_array(&primary_hashes)
                    .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>()))
                .into(),
            );

            let hash = (*BoundedSlice::new_from_equal_array(&primary_hashes).get(po)).into();
            let secondary_hash =
                (*BoundedSlice::new_from_equal_array(&secondary_hashes).get(po)).get();
            let table = BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
            let replacement_idx =
                BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx).get_mut(hash);
            fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, TABLE_SIZE_MINUS_ONE>(
                pos,
                secondary_hash,
                table,
                replacement_idx,
            );
        }
        end + skip as usize - start
    }

    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
    pub fn parse_and_emit_metablock<
        const FAST_MATCHING: bool,
        const MIN_GAIN_FOR_GREEDY: i32,
        const USE_LAST_DISTANCES: bool,
    >(
        &mut self,
        data: &[u8],
        start: usize,
        count: usize,
        metablock_data: &mut MetablockData,
    ) -> usize {
        if FAST_MATCHING {
            return self.parse_and_emit_metablock_fast::<USE_LAST_DISTANCES>(
                data,
                start,
                count,
                metablock_data,
            );
        }
        // TODO(veluca): for some reason, not enabling target features on this function results in
        // slightly faster code.
        let mut bpos = start;
        bpos += self.parse_and_emit_interior::<MIN_GAIN_FOR_GREEDY, USE_LAST_DISTANCES>(
            data,
            bpos,
            (bpos + count)
                .min(data.len().saturating_sub(INTERIOR_MARGIN))
                .saturating_sub(bpos),
            metablock_data,
        );
        while bpos < start + count {
            metablock_data.add_literal(data[bpos], true);
            bpos += 1;
        }
        bpos - start
    }

    /// Returns the number of bytes that were written to the output. Updates the hash table with
    /// strings starting at all of those bytes, if within the margin.
    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
    #[inline(never)]
    fn parse_and_emit_interior_fast<const USE_LAST_DISTANCES: bool>(
        &mut self,
        data: &[u8],
        start: usize,
        count: usize,
        metablock_data: &mut MetablockData,
    ) -> usize {
        let end_upper_bound = data.len().saturating_sub(INTERIOR_MARGIN - 1);
        let end = end_upper_bound.min(count + start);
        if end <= start {
            return 0;
        }

        let mut primary_hashes = [BoundedU32::constant::<0>(); PRECOMPUTE_SIZE];
        let mut secondary_hashes = [BoundedU8::constant::<0>(); 16];

        let mut last_pc = 1;

        let mut last_distances = [0; 2];

        let mut pos = start;
        while pos < end {
            let pc = (pos - start) / (PRECOMPUTE_SIZE / 2);
            if pc != last_pc {
                let data_slice = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(
                    data,
                    pc * (PRECOMPUTE_SIZE / 2) + start,
                )
                .unwrap();

                compute_hashes_at(data_slice, &mut primary_hashes, &mut secondary_hashes);
            }
            last_pc = pc;
            let data_slice = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(data, pos).unwrap();

            let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
            self.prefetch_pos(
                (*BoundedSlice::new_from_equal_array(&primary_hashes)
                    .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>()))
                .into(),
            );

            let hash = (*BoundedSlice::new_from_equal_array(&primary_hashes).get(po)).into();
            let secondary_hash =
                (*BoundedSlice::new_from_equal_array(&secondary_hashes).get(po)).get();
            let table = BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
            let replacement_idx =
                BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx).get_mut(hash);

            let (dist, len, _gain) = if replacement_idx.get() == 0 {
                (0, 0, 0)
            } else {
                table_search::<
                    ENTRY_SIZE,
                    ENTRY_SIZE_MINUS_ONE,
                    TABLE_SIZE_MINUS_ONE,
                    USE_LAST_DISTANCES,
                >(data, pos, secondary_hash, table, last_distances)
            };
            fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, TABLE_SIZE_MINUS_ONE>(
                pos,
                secondary_hash,
                table,
                replacement_idx,
            );
            let lit = *data_slice.get(BoundedUsize::<1>::constant::<0>());
            let (lit_params, copy_params) = if len >= 4 {
                const _: () = assert!(PREFETCH_OFFSET <= 4);
                for i in 1..PREFETCH_OFFSET {
                    let hash = primary_hashes[po.get() + i].into();
                    let secondary_hash = secondary_hashes[po.get() + i].get();
                    let table =
                        BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
                    let replacement_idx =
                        BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx)
                            .get_mut(hash);
                    fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, TABLE_SIZE_MINUS_ONE>(
                        pos + i,
                        secondary_hash,
                        table,
                        replacement_idx,
                    );
                }
                pos += len as usize;
                ((0, false), (len, dist, true))
            } else {
                pos += 1;
                ((lit, true), (0, 0, false))
            };
            metablock_data.add_literal(lit_params.0, lit_params.1);
            metablock_data.add_copy(copy_params.0, copy_params.1, copy_params.2);
            if USE_LAST_DISTANCES {
                last_distances = if copy_params.2 {
                    [copy_params.1, last_distances[0]]
                } else {
                    last_distances
                };
            }
        }

        pos - start
    }

    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
    pub fn parse_and_emit_metablock_fast<const USE_LAST_DISTANCES: bool>(
        &mut self,
        data: &[u8],
        start: usize,
        count: usize,
        metablock_data: &mut MetablockData,
    ) -> usize {
        let mut bpos = start;
        bpos += self.parse_and_emit_interior_fast::<USE_LAST_DISTANCES>(
            data,
            bpos,
            (bpos + count)
                .min(data.len().saturating_sub(INTERIOR_MARGIN))
                .saturating_sub(bpos),
            metablock_data,
        );
        while bpos < start + count {
            metablock_data.add_literal(data[bpos], true);
            bpos += 1;
        }
        bpos - start
    }
}

#[cfg(test)]
mod test {
    use super::_mm256_ilog2_epi32;
    use crate::constants::*;
    use safe_arch_macro::safe_arch_entrypoint;
    use std::arch::x86_64::{_mm256_extract_epi32, _mm256_set1_epi32};

    #[test]
    #[safe_arch_entrypoint("avx", "avx2")]
    fn test_ilog2() {
        let step = if cfg!(miri) { 1_000_000 } else { 1 };

        for i in (1..WSIZE).step_by(step) {
            let simd =
                _mm256_extract_epi32::<0>(_mm256_ilog2_epi32(_mm256_set1_epi32(i as i32))) as u32;
            assert_eq!(simd, i.ilog2());
        }
    }
}
