-- ASSIGNMENT 2: Flat-Parallel implementation of Sparse Matrix-Vector Multiplication
-- ==
-- compiled input { 
--   [0i64, 1i64, 0i64, 1i64, 2i64, 1i64, 2i64, 3i64, 2i64, 3i64, 3i64]
--   [2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, -1.0f32, -1.0f32, 2.0f32, 3.0f32]
--   [2i64, 3i64, 3i64, 2i64, 1i64]
--   [2.0f32, 1.0f32, 0.0f32, 3.0f32]
-- } 
-- output { [3.0f32, 0.0f32, -4.0f32, 6.0f32, 9.0f32] }

------------------------
--- Sgm Scan Helpers ---
------------------------

-- segmented scan with (+) on floats:
let sgmSumF32 [n] (flg : [n]bool) (arr : [n]f32) : [n]f32 =
  let flgs_vals = 
    scan ( \ (f1, x1) (f2,x2) -> 
            let f = f1 || f2 in
            if f2 then (f, x2)
            else (f, x1 + x2) )
         (false, 0.0f32) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals



-----------------------------------------------------
-- Please implement the function below, currently dummy,
-- which is supposed to implement sparse-matrix vector
-- multiplication. Note that the shp array contains the 
-- sizes of each row of the matrix.
-----------------------------------------------------
---  Dense Matrix:                                ---
---  [ 2.0, -1.0,  0.0, 0.0]                      ---
---  [-1.0,  2.0, -1.0, 0.0]                      ---
---  [ 0.0, -1.0,  2.0,-1.0]                      ---
---  [ 0.0,  0.0, -1.0, 2.0]                      ---
---  [ 0.0,  0.0,  0.0, 3.0]                      ---
---                                               ---
---  In the sparse, nested-parallel format it is  ---
---  represented as a list of lists named mat     ---
---  [ [(0,2.0),  (1,-1.0)],                      ---
---    [(0,-1.0), (1, 2.0), (2,-1.0)],            ---
---    [(1,-1.0), (2, 2.0), (3,-1.0)],            ---
---    [(2,-1.0), (3, 2.0)],                      ---
---    [(3,3.0)]                                  ---
---  ]                                            ---
---                                               ---
--- The nested-parallel code is something like:   ---
--- map (\ row ->                                 ---
---         let prods =                           ---
---               map (\(i,x) -> x*vct[i]) row    ---
---         in  reduce (+) 0 prods                ---
---     ) mat                                     ---
---                                               ---
--- mat is the flattened data of the matrix above,---
---  while shp holds the sizes of each row, i.e., ---
---                                               ---
--- mat_val = 
---       [ (0,2.0),(1,-1.0),(0,-1.0),(1, 2.0),   ---
---         (2,-1.0),(1,-1.0),(2, 2.0),(3,-1.0),  ---
---         (2,-1.0),(3, 2.0),(3,3.0)             ---
---       ]                                       ---
--- mat_shp = [2,3,3,2,1]                         ---
---                                                  
--- The vector is dense and matches the number of ---
---   columns of the matrix                       ---
---   e.g., x = [2.0, 1.0, 0.0, 3.0] (transposed) ---
---                                               ---
--- YOUR TASK is to implement the function below  ---
--- such that it consists of only flat-parallel   ---
--- operations and is semantically equivalent to  ---
--- the nested parallel program described above   ---
--- See also Section 3.2.4 ``Sparse-Matrix Vector ---
--- Multiplication'' in lecture notes, page 40-41.---
--- You may use in your implementation the        ---
--- `sgmSumF32` function provided above.          ---
--- You may also take a look at the sequential    ---
--- implementation in file spMVmult-seq.fut.      ---
--- If the futhark-opencl compiler complains about---
---  unsafe code try wrapping the offending       ---
---  expression with keyword `unsafe`.            ---
---                                               ---
--- Necessary steps for the flat-parallel implem: ---
--- 1. you need to compute the flag array from shp---
--- 2. you need to multiply all elements of the   --- 
---    matrix with their corresponding vector     ---
---    element                                    ---
--- 3. you need to sum up the products above      ---
---    across each row of the matrix. This can    ---
---    be achieved with a segmented scan and then ---
---    with a map that extracts the last element  ---
---    of the segment.
-----------------------------------------------------
let mkFlagArray 't [m] (aoa_shp: [m]i32) (zero: t) (aoa_val : [m]t) : []i64 = 
  let shp_rot = map (\i->if i==0 then 0 else aoa_shp[i-1]) (iota m)
  let shp_scn = scan (+) 0 shp_rot 
  let aoa_len = shp_scn[m-1]+ aoa_shp[m-1] 
  let shp_ind = map2 (\shp ind -> if shp ==0 then i64.i32 -1 else ind) aoa_shp shp_scn 
  in scatter (replicate (i64.i32 aoa_len) zero) shp_ind aoa_val


let spMatVctMult [num_elms] [vct_len] [num_rows] 
                 (mat_val : [num_elms](i64,f32))
                 (mat_shp : [num_rows]i64)
                 (vct : [vct_len]f32) : [num_rows]f32 =

  let shp_sc   = scan (+) 0 mat_shp
    
  in  replicate num_rows 0.0f32
  
-- One may run with for example:
-- $ futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./spMVmult-seq -t /dev/stderr > /dev/null
let main [n] [m] 
         (mat_inds : [n]i64) (mat_vals : [n]f32) 
         (shp : [m]i64) (vct : []f32) : [m]f32 =
  spMatVctMult (zip mat_inds mat_vals) shp vct
