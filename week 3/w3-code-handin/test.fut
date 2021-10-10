-- compiled input { 
--   [1i32, 2i32]
--   [3i32, 4i32]
-- } 
-- output { 
--   [1i32, 5i32]
--   [9i32, 15i32]
-- } 

let main (A: []i32) : []i32 =
    map2(\i j -> scan (**) 0 A)