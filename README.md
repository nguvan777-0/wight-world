# wight-world

wight-world is a neuroevolution engine that executes natural selection purely through image filters. Instead of modeling discrete virtual bodies, the organism here is a *wight*—a single living pixel made entirely of floating-point logic. The wight is exactly its weights.

> *Life thrives at the boundary of states. Inject an energy gradient into a static tensor, and the math will organize to transform it.*

This project abandons object-oriented loops in favor of **hardware-native natural selection**. The entire ecosystem is a single contiguous tensor where the laws of physics and biology are reduced to simultaneous 2D convolutions.

Apple designed the Apple Neural Engine (ANE) for FaceID and computational photography. By formatting the laws of biology—energy transformation, spatial translation, and tensor mitosis—exclusively as spatial matrix operations, we hijack that exact hardware to execute natural selection natively. No loops, no lists, just single-pass tensor updates.

Because there are no arrays of objects to iterate through, time complexity is a flat **`O(1)`** wall-clock per tick, completely decoupled from population size. Space complexity is locked at **`O(W × H)`**. The ANE evaluates the entire ecosystem in a single hardware dispatch whether there are 5 wights alive or 64,000.

## Substrate

1. **The Tensor World:** A 3D matrix of size `[Width, Height, Channels]`. Empty space is `0.0`. A piece of food is a positive number in the `CH_FOOD` channel.
2. **The Wight:** A single pixel—an 18-channel depth column at a specific coordinate (1 energy state, 1 age, 1 energy drain accumulator, and 15 neural weights). A wight has no discrete body or positional state—it is exactly its brain.
4. **Evolution without Code:**
   - Survival: If a coordinate's `Energy` channel hits zero, we multiply its depth column by `0`. It vanishes.
   - Reproduction: If a coordinate acquires enough `Energy`, it copies its brain weights into an adjacent coordinate.
   - Mutation: During the copy, random noise (`numpy.random`) is injected into the weights.

## Let there be wight

Requires macOS Apple Silicon and Python 3.11+. The first run will evaluate the matrix graph and compile an ANE-optimized `.mlpackage` to the local `build/` directory. Subsequent boots load instantly from this hardware cache.

```bash
uv run python world.py
```

Or, bypass the UI and execute headlessly for a specific number of ticks to crunch the tensor math at hardware speed:

```bash
uv run python world.py --headless --ticks 1200
```

```text
Running simulation headless for 1200 ticks...
────────────────────────────────────────────────────────────────────────
    tick    pop  e_avg  e_max  a_avg  a_max  d_avg  d_max  elapsed
────────────────────────────────────────────────────────────────────────
       0     24     50     50      0      1      0      1  0.0s
     500   1588     62     79     98    501     97    508  0.4s
          ↳ population boom  24 → 1588
          ↳ metabolism surging  0 → 97
    1000   1778     60     79    176   1001    175    996  0.8s
          ↳ longevity unlocked — max age > 1,000
          ↳ metabolism surging  97 → 175
    1199   1719     60     79    221   1200    222   1151  1.0s
────────────────────────────────────────────────────────────────────────
1,200 ticks  1.0s  1,240 t/s
```



## Speciation & Strategy Space

Because the organism *is* strictly its mathematical intent, the HUD allows you to watch the population's weights diverge and speciate in real-time.

1. **Weight Heatmaps:** Tracks the 10th to 90th percentile of the population's brain matrices. Unoptimized wights render as static **Green** `[0.0]`. Over thousands of generations, survival pressure forces the distributions to stretch into heavy **Blue** (inhibitory) and **Red** (excitatory) extremes.
2. **Live Subject PCA:** A real-time Principal Component Analysis (computed via fast SVD) compresses the 15-channel brains into a 2D scatter plot.

When the matrix boots, the population initializes as a single noisy cloud. As natural selection filters the world tensor, you can watch the math tear itself apart into distinct, autonomous islands.

These clusters represent ecological niches forming within the latent space—such as an island of high-mobility scavengers branching away from a dense cluster of stationary lurkers. You are viewing speciation as a byproduct of mathematical optimization.

## Layout

Because the physics, biology, and environment are just layers of the same tensor, the entire engine is beautifully compressed down to a single Python script:

`world.py` — The unified Core ML graph, tensor setup, runtime, and emergent Pygame renderer.

## License

BSD 3-Clause