-- Primes: Naive Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 } auto output

let primesHelp [np1] (sq : i64) (a : *[np1]i32) : [np1]i32 =
  let n = np1 - 1 in
  loop(a) for i < (sq-1) do 
        let i    = i + 2 in
        if a[i] == 0i32 then a
        else
          let m    = (n / i) - 1
          let inds = map (\k -> (k+2)*i) (iota m)
          let vals = replicate m 0
          let a'  = scatter a inds vals
          in  a'

-- Run with $echo "1000000" | ./primes-naive -t /dev/stderr -r 10 > /dev/null
let main (n : i64) : []i64 =
  let np1 = n+1
  let a = map (\i -> if i==0 || i==1 then 0 else 1) (iota np1)
  let sq= i64.f64 (f64.sqrt (f64.i64 n))
  let fl= primesHelp sq a
  in  filter (\i -> fl[i]!=0) (iota np1)

