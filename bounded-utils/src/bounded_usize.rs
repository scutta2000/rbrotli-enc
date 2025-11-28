// Copyright 2025 Google LLC
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

// use crate::make_bounded_type::make_bounded_type;
use crate::make_bounded_type::CheckPow2MinusOne;
use crate::utils::CheckBound;
use std::ops::{BitAnd, BitOr};

use zerocopy::{AsBytes, FromZeroes};

make_bounded_type!(
    $,
    BoundedUsize,
    bounded_usize_array,
    usize
);
make_bounded_type!($, BoundedU8, bounded_u8_array, u8);
make_bounded_type!($, BoundedU32, bounded_u32_array, u32);

impl BoundedUsize<255> {
    pub fn from_u8(val: u8) -> BoundedUsize<255> {
        BoundedUsize(val as usize)
    }
}

impl<const BOUND: usize> From<BoundedU8<BOUND>> for BoundedUsize<BOUND> {
    fn from(value: BoundedU8<BOUND>) -> Self {
        BoundedUsize(value.0 as usize)
    }
}

impl<const BOUND: usize> From<BoundedU32<BOUND>> for BoundedUsize<BOUND> {
    fn from(value: BoundedU32<BOUND>) -> Self {
        const _: () = assert!(std::mem::size_of::<u32>() <= std::mem::size_of::<usize>());
        BoundedUsize(value.0 as usize)
    }
}
