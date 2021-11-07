import "radix_sort"

-- RUN a big test with:
-- $ futhark opencl radix_sort_futhark.fut
-- $ echo "[5,4,3,2,1,0,-1,-2]" | ./radix_sort_futhark -t /dev/stderr -r 10 > /dev/null
let main (n : []i32) : []i32 =
    radix_sort_int i32.num_bits i32.get_bit n
