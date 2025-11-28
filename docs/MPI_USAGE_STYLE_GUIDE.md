# MPI Usage Style Guide

**Date**: 2025-11-28
**Topic**: MPI function naming conventions in Geodynamo.jl

---

## Current Pattern

The codebase currently uses **fully qualified names** for all MPI functions:

```julia
using MPI

# Usage:
MPI.Comm_size(comm)
MPI.Allreduce(data, MPI.SUM, comm)
MPI.Barrier(comm)
```

## Usage Statistics

Based on current codebase analysis:
```
72 uses - MPI.Allreduce
63 uses - MPI.SUM
42 uses - MPI.Wtime
22 uses - MPI.Comm_size
13 uses - MPI.Barrier
 7 uses - MPI.Comm_rank
 7 uses - MPI.COMM_WORLD
 5 uses - MPI.MAX
 2 uses - MPI.MIN
```

---

## Option 1: Keep Current Pattern (RECOMMENDED)

### Advantages ✅

1. **Explicitness**: Immediately obvious functions are from MPI
2. **No namespace pollution**: No risk of name conflicts
3. **Consistency**: Matches current codebase style
4. **Julia best practice**: Recommended by Julia style guide
5. **Searchability**: Easy to find all MPI calls with grep

### Example
```julia
using MPI

function my_function()
    comm = get_comm()
    if MPI.Comm_size(comm) > 1
        MPI.Allreduce!(data, MPI.SUM, comm)
    end
end
```

### Recommendation
✅ **Keep current pattern** - No changes needed

---

## Option 2: Import Specific Symbols

### Implementation
```julia
using MPI: Comm_size, Comm_rank, Allreduce, Allreduce!,
           SUM, MAX, MIN, COMM_WORLD, Barrier, Wtime
```

### Advantages
- Shorter function calls
- Still explicit (imported symbols are listed)
- Common in some Julia codebases

### Disadvantages
- Need to maintain import list
- Less obvious where functions come from (need to check imports)
- Inconsistent with current codebase
- May conflict with future additions

### Example
```julia
using MPI: Comm_size, Allreduce!, SUM

function my_function()
    comm = get_comm()
    if Comm_size(comm) > 1  # Shorter, but less obvious
        Allreduce!(data, SUM, comm)
    end
end
```

---

## Option 3: Import All MPI (NOT RECOMMENDED)

### Implementation
```julia
using MPI
import MPI: *  # Import everything
```

### Why NOT recommended
- ❌ Pollutes namespace with 100+ symbols
- ❌ High risk of name conflicts
- ❌ Unclear where functions come from
- ❌ Against Julia style guide
- ❌ Makes code harder to understand

---

## Comparison

### Readability Comparison

**Current (Qualified):**
```julia
if MPI.Comm_size(comm) > 1
    MPI.Allreduce!(profile_real, MPI.SUM, comm)
    MPI.Allreduce!(profile_imag, MPI.SUM, comm)
end
```

**With Imports:**
```julia
if Comm_size(comm) > 1
    Allreduce!(profile_real, SUM, comm)
    Allreduce!(profile_imag, SUM, comm)
end
```

**Line length savings**: ~8 characters per MPI call

---

## Recommendation

### For Geodynamo.jl: **Keep Current Pattern** ✅

**Reasons:**
1. **Consistency**: 270+ existing MPI calls use this pattern
2. **Best practice**: Aligns with Julia style guide
3. **Clarity**: Immediately obvious these are MPI functions
4. **Maintainability**: Easy to search and refactor
5. **No conflicts**: `Allreduce` won't conflict with future packages

### If You Prefer Shorter Names

If you strongly prefer shorter names, here's the minimal change:

**Add to `src/Geodynamo.jl` after `using MPI`:**
```julia
using MPI: Comm_size, Comm_rank, Allreduce, Allreduce!,
           SUM, MAX, MIN, COMM_WORLD, Barrier, Wtime,
           Request, Isend, Irecv, Waitall
```

Then update all files to use the imported names.

**Effort**: Would need to update ~270 lines across the codebase.

---

## Other Codebases

### How other Julia HPC codes handle this:

**CliMA (Climate Modeling Alliance):**
```julia
using MPI  # Uses MPI.Comm_size, MPI.Allreduce, etc.
```
✅ Uses qualified names

**Trixi.jl (PDE solver):**
```julia
using MPI: MPI  # Import as module
# Uses MPI.Comm_size, MPI.Allreduce
```
✅ Uses qualified names

**PencilArrays.jl:**
```julia
using MPI
# Uses MPI.Comm_size, MPI.COMM_WORLD
```
✅ Uses qualified names

**Consensus**: Most Julia HPC packages use qualified `MPI.*` names.

---

## Decision Matrix

| Criterion | Keep Current | Import Symbols |
|-----------|--------------|----------------|
| Clarity | ✅ Excellent | ⚠️ Good |
| Consistency | ✅ Matches codebase | ❌ New pattern |
| Best Practice | ✅ Yes | ⚠️ Acceptable |
| Line Length | ⚠️ Longer | ✅ Shorter |
| Maintenance | ✅ Easy | ⚠️ Need import list |
| Conflicts | ✅ None | ⚠️ Possible |
| Effort | ✅ Zero | ❌ High (~270 lines) |

---

## Conclusion

### Recommendation: **No Change Needed** ✅

The current pattern of using `MPI.Comm_size`, `MPI.Allreduce`, etc. is:
- ✅ **Correct** - Widely used in Julia community
- ✅ **Clear** - No ambiguity about function origin
- ✅ **Consistent** - Matches entire codebase style
- ✅ **Maintainable** - Easy to search and refactor

### If Shorter Names Desired

If you decide shorter names would significantly improve readability:

1. Add specific imports to `src/Geodynamo.jl`
2. Update all MPI calls across codebase (~270 occurrences)
3. Verify no name conflicts
4. Update this guide

But this is **optional** - the current code is already following best practices.

---

## Example Files to Update (if changing)

If you decide to import symbols, these files would need updates:

```
src/velocity.jl          - 16 MPI calls
src/scalar_field_common.jl - 8 MPI calls
src/thermal.jl           - 10 MPI calls
src/compositional.jl     - 10 MPI calls
src/magnetic.jl          - 6 MPI calls
src/simulation.jl        - 8 MPI calls
src/timestep.jl          - 45 MPI calls
src/shtnskit_field_functions.jl - 4 MPI calls
src/outputs_writer.jl    - 5 MPI calls
src/optimizations.jl     - 12 MPI calls
```

Total: ~270 lines to update across 10 files.

---

**Author**: Claude Code Analysis
**Date**: 2025-11-28
**Status**: Current pattern is recommended
