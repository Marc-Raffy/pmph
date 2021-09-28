-- Primes: Naive Version
-- ==
-- compiled input { 30i64 } output { [2i64,3i64,5i64,7i64,11i64,13i64,17i64,19i64,23i64,29i64] }
-- compiled input { 300000i64 } auto output

-- run with: futhark bench --backend=opencl primes-adhoc.fut -r 1
-- or with
-- echo "300000i32" | ./primes-adhoc -t /dev/stderr > /dev/null

-- Find the first n primes
let primes (n:i64) : []i64 =
  let (acc, _) =
    loop (acc,c) = ([],2i64)
      while c < n do
      --while c < n+1 do
        let c2 = if n / c < c then n else c*c
        --let c2 = i32.min (c * c) (n+1)
        let is = map (+c) (iota(c2-c))
        let fs = map (\i ->
                        let xs = map (\p -> if i%p==0 then 1 else 0) acc
                        in reduce (+) 0 xs
                     ) is
        -- apply the sieve
        let new = filter (\i -> 0i32 == fs[i-c]) is
        in (concat acc new, c2)
  in acc

-- Return the number of primes less than n
let main (n:i64) : []i64 =
  primes n
