import "radix_sort"

let main (n : []i32) : []i32 =
    radix_sort_int i32.num_bits i32.get_bit n
