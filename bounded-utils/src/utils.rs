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

pub struct CheckBound<const N: usize, const M: usize, const ADD: usize>;
impl<const N: usize, const M: usize, const ADD: usize> CheckBound<N, M, ADD> {
    pub const CHECK_GT: () = assert!((N as u128) > (M as u128 + ADD as u128));
    pub const CHECK_GE: () = assert!((N as u128) >= (M as u128 + ADD as u128));
}
