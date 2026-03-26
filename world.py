import os
import sys
import time
import argparse
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
except ImportError:
    print("Warning: coremltools not found. The model cannot be built.")

# --- Configuration ---
W_GRID, H_GRID = 64, 64
RENDER_SCALE = 12
W_PX, H_PX = W_GRID * RENDER_SCALE, H_GRID * RENDER_SCALE
HUD_WIDTH = 380

CH_FOOD = 1
CH_ENERGY = 1
CH_AGE = 1
CH_ENERGY_DRAIN = 1
CH_WEIGHTS = 15 # 3 senses * 5 intentions = 15 parameters
CH_ORG = CH_ENERGY + CH_AGE + CH_ENERGY_DRAIN + CH_WEIGHTS
CH_TOTAL = CH_FOOD + CH_ORG

MODEL_PATH = "build/ane_wight_world.mlpackage"

# --- Directional Kernels for Discrete Shifting ---
# We represent 5 discrete choices: 0:Stay, 1:North, 2:South, 3:East, 4:West
# To process movement, every pixel "pulls" the state of wights wanting to enter it.
def create_pull_kernels():
    k_stay = np.zeros((1, 1, 3, 3), dtype=np.float32); k_stay[0,0,1,1] = 1.0
    # Intent 1: Move North -> We pull from the cell BELOW us (Bottom)
    k_pull_from_B = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_B[0,0,2,1] = 1.0
    # Intent 2: Move South -> We pull from the cell ABOVE us (Top)
    k_pull_from_T = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_T[0,0,0,1] = 1.0
    # Intent 3: Move East -> We pull from the cell to our LEFT
    k_pull_from_L = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_L[0,0,1,0] = 1.0
    # Intent 4: Move West -> We pull from the cell to our RIGHT
    k_pull_from_R = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_R[0,0,1,2] = 1.0
    return [k_stay, k_pull_from_B, k_pull_from_T, k_pull_from_L, k_pull_from_R]

def mb_circular_pad(x):
    """
    Manually pads a tensor (1, C, H, W) circularly by 1 pixel using ANE-friendly slice and concat.
    """
    # Pad Width (axis 3)
    left_pad = mb.slice_by_index(x=x, begin=[0,0,0,W_GRID-1], end=[1,0,H_GRID,W_GRID],
                                 begin_mask=[True,True,True,False], end_mask=[True,True,True,False])
    right_pad = mb.slice_by_index(x=x, begin=[0,0,0,0], end=[1,0,H_GRID,1],
                                  begin_mask=[True,True,True,False], end_mask=[True,True,True,False])
    padded_w = mb.concat(values=[left_pad, x, right_pad], axis=3)

    # Pad Height (axis 2) Note: padded_w is now W_GRID+2
    top_pad = mb.slice_by_index(x=padded_w, begin=[0,0,H_GRID-1,0], end=[1,0,H_GRID,0],
                                begin_mask=[True,True,False,True], end_mask=[True,True,False,True])
    bottom_pad = mb.slice_by_index(x=padded_w, begin=[0,0,0,0], end=[1,0,1,0],
                                   begin_mask=[True,True,False,True], end_mask=[True,True,False,True])

    return mb.concat(values=[top_pad, padded_w, bottom_pad], axis=2)

def build_evolution_engine():
    """Builds the hardware evolution engine: one forward pass advances the entire evolutionary ecosystem on the ANE."""
    kernels = create_pull_kernels()
    pull_k_org = [np.repeat(k, CH_ORG, axis=0) for k in kernels]     # Shift the 16 organism channels
    pull_k_int = [np.repeat(k, 5, axis=0) for k in kernels]          # Shift the 5 intention output channels

    # Simple uniform blur for food sensing
    k_blur = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, CH_TOTAL, H_GRID, W_GRID)),
        mb.TensorSpec(shape=(1, CH_WEIGHTS, H_GRID, W_GRID))
    ])
    def world_step(world, mutation):
        # 1. SLICE WORLD CHANNELS
        food_layer = mb.slice_by_index(x=world, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        org_layer  = mb.slice_by_index(x=world, begin=[0,1,0,0], end=[1,CH_TOTAL,H_GRID,W_GRID])
        energy     = mb.slice_by_index(x=org_layer, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        # Age and Energy_Drain are ignored by the brain but carried by the organism
        weights    = mb.slice_by_index(x=org_layer, begin=[0,3,0,0], end=[1,CH_ORG,H_GRID,W_GRID])

        # 2. SENSING
        # Use our manual Torus wrap using slice/concat so Core ML compiles happily
        pad_food = mb_circular_pad(food_layer)
        blur_food = mb.conv(x=pad_food, weight=k_blur, pad_type="valid")
        senses = [food_layer, blur_food, energy] # 3 Inputs to the neural net

        # 3. ORGANISM BRAIN (Map 3 Senses -> 5 Intentions)
        intent_channels = []
        for out_idx in range(5):
            net = None
            for in_idx in range(3):
                w_idx = (out_idx * 3) + in_idx
                # Per-pixel weights acting as a dense layer
                w_ch = mb.slice_by_index(x=weights, begin=[0,w_idx,0,0], end=[1,w_idx+1,H_GRID,W_GRID])
                term = mb.mul(x=senses[in_idx], y=w_ch)
                if net is None: net = term
                else: net = mb.add(x=net, y=term)

            intent_channels.append(net)

        # 4. DISCRETE ACTION SELECTION (1-HOT)
        intent_scores = mb.concat(values=intent_channels, axis=1) # Shape: (1, 5, H, W)

        # Use reduce_argmax to break ties and prevent FP16 duplicates
        best_idx = mb.reduce_argmax(x=intent_scores, axis=1) # Shape: (1, H, W)
        best_idx = mb.cast(x=best_idx, dtype="fp32")
        best_idx_expanded = mb.expand_dims(x=best_idx, axes=[1]) # Shape: (1, 1, H, W)

        onehot_channels = []
        for d in range(5):
            d_val = np.array([[[[d]]]], dtype=np.float32)
            c = mb.cast(x=mb.equal(x=best_idx_expanded, y=d_val), dtype="fp32")
            onehot_channels.append(c)

        raw_intent_1hot = mb.concat(values=onehot_channels, axis=1) # Shape: (1, 5, H, W)

        # Prevent completely empty background cells from generating empty-cell movement intentions
        # that secretly steal priority in the tensor collision ladder and delete living organisms!
        has_energy = mb.cast(x=mb.greater(x=energy, y=np.float32(0.0)), dtype="fp32")
        intent_1hot = mb.mul(x=raw_intent_1hot, y=has_energy)

        # 5. SHIFT-AND-MASK TO RESOLVE MOVEMENT
        # Mode "circular" creates a Torus geometry: organisms walking off the right edge
        # wrap into the left edge.
        pad_org = mb_circular_pad(org_layer)
        pad_int = mb_circular_pad(intent_1hot)

        cands_org = []
        cands_valid = []

        for d in range(5):
            # Pull the neighbor's state and intention into this pixel
            sh_org = mb.conv(x=pad_org, weight=pull_k_org[d], pad_type="valid", groups=CH_ORG)
            sh_int = mb.conv(x=pad_int, weight=pull_k_int[d], pad_type="valid", groups=5)
            # Did the neighbor we pulled actually *choose* to move in direction 'd'?
            v = mb.slice_by_index(x=sh_int, begin=[0,d,0,0], end=[1,d+1,H_GRID,W_GRID])

            cands_org.append(sh_org)
            cands_valid.append(v)

        # Collision resolution: Priority ladder (Stay > N > S > E > W)
        final_org = None
        cum_w = None
        for d in range(5):
            # W marks if this candidate is the single winner permitted to enter the cell
            if d == 0:
                w = cands_valid[0]
                cum_w = w
            else:
                w = mb.mul(x=cands_valid[d], y=mb.sub(x=np.float32(1.0), y=cum_w))
                cum_w = mb.add(x=cum_w, y=w)
            # Add winner into the new state (Losers are multiplied by 0.0)
            term = mb.mul(x=cands_org[d], y=w)
            if final_org is None: final_org = term
            else: final_org = mb.add(x=final_org, y=term)

        # 6. SURVIVAL & METABOLISM
        shifted_energy  = mb.slice_by_index(x=final_org, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        shifted_age     = mb.slice_by_index(x=final_org, begin=[0,1,0,0], end=[1,2,H_GRID,W_GRID])
        shifted_drain   = mb.slice_by_index(x=final_org, begin=[0,2,0,0], end=[1,3,H_GRID,W_GRID])
        shifted_weights = mb.slice_by_index(x=final_org, begin=[0,3,0,0], end=[1,CH_ORG,H_GRID,W_GRID])

        # Burn energy every tick
        tick_drain = np.float32(0.01)
        post_move_energy = mb.sub(x=shifted_energy, y=tick_drain)
        is_there = mb.cast(x=mb.greater(x=post_move_energy, y=np.float32(0.0)), dtype="fp32")

        new_age = mb.add(x=shifted_age, y=is_there)
        new_drain = mb.add(x=shifted_drain, y=mb.mul(x=is_there, y=tick_drain))

        # If alive, eat the food we are standing on
        can_eat = mb.mul(x=food_layer, y=is_there)
        food_eaten = mb.clip(x=can_eat, alpha=np.float32(0.0), beta=np.float32(0.9))

        # Update biological energy
        new_energy = mb.add(x=post_move_energy, y=food_eaten)
        new_energy = mb.clip(x=new_energy, alpha=np.float32(0.0), beta=np.float32(1.0))
        new_food = mb.sub(x=food_layer, y=food_eaten)

        # Basic death mask before reproduction
        is_alive_now = mb.cast(x=mb.greater(x=new_energy, y=np.float32(0.0)), dtype="fp32")
        base_energy = mb.mul(x=new_energy, y=is_alive_now)
        base_age    = mb.mul(x=new_age, y=is_alive_now)
        base_drain  = mb.mul(x=new_drain, y=is_alive_now)
        base_weights = mb.mul(x=shifted_weights, y=is_alive_now)

        # 7. MITOSIS (Reproduction & Mutation)
        # Parent decides to reproduce if energy > 0.8
        can_reproduce = mb.cast(x=mb.greater(x=base_energy, y=np.float32(0.8)), dtype="fp32")

        # Parent loses half energy if reproducing
        cost = mb.mul(x=base_energy, y=mb.mul(x=can_reproduce, y=np.float32(0.5)))
        parent_energy = mb.sub(x=base_energy, y=cost)

        # Offspring payload: gets the cost energy, parent's weights + random mutation noise
        mutated_weights = mb.add(x=base_weights, y=mutation)
        offspring_w_send = mb.mul(x=mutated_weights, y=can_reproduce)
        offspring_e_send = cost

        # Offspring has 0 age and 0 drain
        offspring_zero = mb.mul(x=cost, y=np.float32(0.0))
        offspring_a_send = offspring_zero
        offspring_d_send = offspring_zero

        # Place offspring in the adjacent cell
        offspring_org_send = mb.concat(values=[offspring_e_send, offspring_a_send, offspring_d_send, offspring_w_send], axis=1)
        pad_offspring = mb_circular_pad(offspring_org_send)
        inbound_offspring = mb.conv(x=pad_offspring, weight=pull_k_org[4], pad_type="valid", groups=CH_ORG)

        # Offspring only survives if it lands on a cell that parent_energy currently evaluates as empty
        is_empty = mb.cast(x=mb.equal(x=parent_energy, y=np.float32(0.0)), dtype="fp32")
        landed_offspring = mb.mul(x=inbound_offspring, y=is_empty)

        landed_e = mb.slice_by_index(x=landed_offspring, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        landed_a = mb.slice_by_index(x=landed_offspring, begin=[0,1,0,0], end=[1,2,H_GRID,W_GRID])
        landed_d = mb.slice_by_index(x=landed_offspring, begin=[0,2,0,0], end=[1,3,H_GRID,W_GRID])
        landed_w = mb.slice_by_index(x=landed_offspring, begin=[0,3,0,0], end=[1,CH_ORG,H_GRID,W_GRID])

        # Combine parent and landed offspring
        final_energy = mb.add(x=parent_energy, y=landed_e)

        # Ensure parent weights are zeroed if dead
        parent_alive = mb.cast(x=mb.greater(x=parent_energy, y=np.float32(0.0)), dtype="fp32")
        kept_a = mb.mul(x=base_age, y=parent_alive)
        kept_d = mb.mul(x=base_drain, y=parent_alive)
        kept_w = mb.mul(x=base_weights, y=parent_alive)

        final_age = mb.add(x=kept_a, y=landed_a)
        final_drain = mb.add(x=kept_d, y=landed_d)
        final_weights = mb.add(x=kept_w, y=landed_w)

        # Reconstruct org channels
        next_org = mb.concat(values=[final_energy, final_age, final_drain, final_weights], axis=1)
        next_world = mb.concat(values=[new_food, next_org], axis=1)

        return next_world

    print(f"Compiling hardware evolution engine ({CH_TOTAL} channels) for ANE...")
    return ct.convert(
        world_step,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13
    )

def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model = build_evolution_engine()
        model.save(MODEL_PATH)
    return ct.models.MLModel(MODEL_PATH)

# --- Runtime & Pygame ---
def init_world():
    world = np.zeros((1, CH_TOTAL, H_GRID, W_GRID), dtype=np.float32)
    world[0, 0] = np.random.rand(H_GRID, W_GRID) * 0.3    # Wild food
    return world

def drop_organism(world, gx, gy, lineage_id=None):
    """Spawns an organism with fully random brain weights."""
    world[0, 1, gy, gx] = 1.0 # Max Energy
    world[0, 2, gy, gx] = 0.0 # Age
    world[0, 3, gy, gx] = 0.0 # Drain
    weights = np.random.randn(CH_WEIGHTS) * 4.0
    if lineage_id is not None and lineage_id < 12:
        # Boost this specific weight so it dominates and locks in the founder's species color!
        weights[lineage_id] = 20.0
    world[0, 4:, gy, gx] = weights

def main():
    parser = argparse.ArgumentParser(description="wight-world neuroevolution engine")
    parser.add_argument("--headless", action="store_true", help="Run in terminal without UI")
    parser.add_argument("--ticks", type=int, default=None, help="Number of ticks to run (default: infinite)")
    args = parser.parse_args()

    is_headless = args.headless
    max_ticks = args.ticks

    if not is_headless:
        pygame.init()
        screen = pygame.display.set_mode((W_PX + HUD_WIDTH, H_PX))
        pygame.display.set_caption("wight-world")

        # Load fonts
        try:
            # specifically request crisp programming fonts before falling back
            font_family = "menlo,monaco,consolas,unifont,monospace"
            font = pygame.font.SysFont(font_family, 14, bold=True)
            font_sm = pygame.font.SysFont(font_family, 12, bold=True)
            font_lg = pygame.font.SysFont(font_family, 18, bold=True)
        except:
            font = pygame.font.SysFont(None, 14, bold=True)
            font_sm = pygame.font.SysFont(None, 12, bold=True)
            font_lg = pygame.font.SysFont(None, 18, bold=True)

        clock = pygame.time.Clock()

        # UI State
        import collections
        lineage_history = collections.deque(maxlen=280)

        _LINEAGE_COLORS = [
            (255,   0,   0), (255, 128,   0), (255, 255,   0), (128, 255,   0), # Red, Orange, Yellow, Chartreuse
            (  0, 255,   0), (  0, 255, 128), (  0, 255, 255), (  0, 128, 255), # Green, Spring Green, Cyan, Sky Blue
            (  0,   0, 255), (128,   0, 255), (255,   0, 255), (255,   0, 128), # Blue, Purple, Magenta, Rose
        ]

    model = get_model()
    world = init_world()

    # Spawn 12 starting founders, seeding the 12 rainbow lineages
    for i in range(12):
        drop_organism(world, np.random.randint(W_GRID), np.random.randint(H_GRID), lineage_id=i)

    if is_headless:
        duration_str = f"{max_ticks:,} ticks" if max_ticks else "forever"
        print(f"\nRunning simulation headless for {duration_str}...")
        print("─" * 72)
        print(f"{'tick':>8} {'pop':>6} {'e_avg':>6} {'e_max':>6} {'a_avg':>6} {'a_max':>6} {'d_avg':>6} {'d_max':>6}  elapsed")
        print("─" * 72)
        t0 = time.time()

        _prev = {'pop': None, 'd_avg': None}
        _flags = set()

        i = 0
        while max_ticks is None or i < max_ticks:
            mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
            out = model.predict({"world": world, "mutation": mutation})
            world = list(out.values())[0]

            world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
            world[0, 0] = np.clip(world[0,0], 0.0, 1.0)

            if i % 500 == 0 or (max_ticks and i == max_ticks - 1):
                orgs = world[0, 1]
                ages = world[0, 2]
                drains = world[0, 3]
                mask = orgs > 0
                pop = mask.sum()
                elapsed = time.time() - t0
                if pop > 0:
                    max_e = int(orgs[mask].max() * 100)
                    avg_e = int(orgs[mask].mean() * 100)
                    max_age = int(ages[mask].max())
                    avg_age = int(ages[mask].mean())
                    max_drain = int(drains[mask].max() * 100)
                    avg_drain = int(drains[mask].mean() * 100)
                    print(f"{i:8d} {pop:6d} {avg_e:6d} {max_e:6d} {avg_age:6d} {max_age:6d} {avg_drain:6d} {max_drain:6d}  {elapsed:.1f}s")

                    notes = []
                    if _prev['pop'] is not None:
                        if pop < _prev['pop'] * 0.5:
                            notes.append(f"population crash  {_prev['pop']} -> {pop}")
                        elif pop > _prev['pop'] * 2.0:
                            notes.append(f"population boom  {_prev['pop']} -> {pop}")

                    if max_age >= 1000 and 'age_1k' not in _flags:
                        notes.append("longevity unlocked - max age > 1,000")
                        _flags.add('age_1k')
                    elif max_age >= 5000 and 'age_5k' not in _flags:
                        notes.append("immortality - max age > 5,000")
                        _flags.add('age_5k')

                    if _prev['d_avg'] is not None:
                        if avg_drain > _prev['d_avg'] * 1.5 and avg_drain > 50:
                            notes.append(f"metabolism surging  {_prev['d_avg']} -> {avg_drain}")
                        elif avg_drain < _prev['d_avg'] * 0.6 and _prev['d_avg'] > 50:
                            notes.append(f"efficiency breakthrough  {_prev['d_avg']} -> {avg_drain}")

                    for note in notes:
                        print(f"          ↳ {note}")

                    _prev['pop'] = pop
                    _prev['d_avg'] = avg_drain
                else:
                    print(f"{i:8d}      EXTINCT")
                    break

            i += 1

        t1 = time.time()
        fps = i / max(1e-6, t1 - t0)
        print("─" * 72)
        print(f"{i:,} ticks  {t1-t0:.1f}s  {fps:,.0f} t/s\n")
        return

    running = True
    tick_count = 0
    print("\nSimulation Started!")
    print(" - Clicking spawns an organism.")

    ui_prev = {'pop': None, 'd_avg': None}
    ui_flags = set()
    ui_events = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gx = np.clip(event.pos[0] // RENDER_SCALE, 0, W_GRID - 1)
                gy = np.clip(event.pos[1] // RENDER_SCALE, 0, H_GRID - 1)
                # Drop an organism with random genes
                drop_organism(world, gx, gy)
                # Drop some large food nearby
                world[0, 0, max(0,gy-2):min(H_GRID,gy+2), max(0,gx-2):min(W_GRID,gx+2)] += 1.0

        # Run strict 1-tick pass on Neural Engine
        mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
        out = model.predict({"world": world, "mutation": mutation})
        world = list(out.values())[0]

        # Background physics: global food regeneration bounds
        world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
        world[0, 0] = np.clip(world[0,0], 0.0, 1.0)

        # Render visual channels
        t = world[0]

        # 1. Background Space & Food Surface
        screen.fill((15, 15, 20)) # Dark void
        rgba = np.zeros((H_GRID, W_GRID, 3), dtype=np.uint8)
        rgba[..., 1] = np.clip(t[0] * 70.0, 0, 255) # Dim green for food algae
        rgba = np.transpose(rgba, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rgba)
        surf_scaled = pygame.transform.scale(surf, (W_PX, H_PX))
        screen.blit(surf_scaled, (0, 0))

        # 2. Draw Organisms dynamically
        orgs = t[1]
        ages = t[2]
        drains = t[3]
        weights = t[4:]
        y_idx, x_idx = np.nonzero(orgs > 0)
        pop = len(y_idx)

        # Track lineages for this frame
        current_lineages = collections.defaultdict(int)

        for y, x in zip(y_idx, x_idx):
            energy = orgs[y, x]
            w = weights[:, y, x]

            # Simple analytical lineage clustering
            # Since mitosis rarely flips the dominant weight,
            # argmax over the first 12 genes creates stable hereditary species.
            lid = int(np.argmax(w[:12]))
            current_lineages[lid] += 1

            c = _LINEAGE_COLORS[lid]

            # Position & pulsating size based on energy
            cx = x * RENDER_SCALE + RENDER_SCALE // 2
            cy = y * RENDER_SCALE + RENDER_SCALE // 2
            r = int((0.5 + energy * 0.5) * RENDER_SCALE * 0.9)

            # Body & Core Outline
            pygame.draw.circle(screen, c, (cx, cy), r)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r, max(1, r//3))

            # Orientation heuristic: sweeping white line mapped to continuous trait 0
            ax = cx + int(np.cos(w[0] * np.pi) * r * 1.8)
            ay = cy + int(np.sin(w[0] * np.pi) * r * 1.8)
            pygame.draw.line(screen, (255, 255, 255), (cx, cy), (ax, ay), max(1, RENDER_SCALE//8))

        lineage_history.append(dict(current_lineages))

        # 3. Draw Side HUD
        px = W_PX
        # Background track for HUD
        pygame.draw.rect(screen, (16, 16, 28), (px, 0, HUD_WIDTH, H_PX))
        pygame.draw.line(screen, (50, 50, 80), (px, 0), (px, H_PX), 1)

        hud_x = px + 10
        stats_y = 10

        def txt(s, f=None, color=(180, 180, 200)):
            nonlocal stats_y
            screen.blit((f or font).render(str(s), True, color), (px + 10, stats_y))
            stats_y += (f or font).get_height() + 2

        def sep():
            nonlocal stats_y
            pygame.draw.line(screen, (40, 40, 60), (px + 8, stats_y + 2), (px + HUD_WIDTH - 8, stats_y + 2), 1)
            stats_y += 8

        # Title
        title_surf = font_lg.render("WIGHT-WORLD", True, (220, 220, 255))
        screen.blit(title_surf, (px + 10, stats_y))
        screen.blit(font_sm.render("DISCRETE ANE", True, (100, 110, 140)), (px + 15 + title_surf.get_width(), stats_y + 4))
        stats_y += font_lg.get_height() + 2
        sep()

        # Stats
        txt(f"tick {tick_count:,}   [ANE]  SPACE=cycle", font_sm, (120, 120, 150))
        sep()

        total_food = int(t[0].sum() * 100)

        if pop > 0:
            min_energy = int(orgs[y_idx, x_idx].min() * 100)
            avg_energy = int(orgs[y_idx, x_idx].mean() * 100)
            max_energy = int(orgs[y_idx, x_idx].max() * 100)
            std_energy = int(orgs[y_idx, x_idx].std() * 100)

            min_age = int(ages[y_idx, x_idx].min())
            avg_age = int(ages[y_idx, x_idx].mean())
            max_age = int(ages[y_idx, x_idx].max())
            std_age = int(ages[y_idx, x_idx].std())

            min_drain = int(drains[y_idx, x_idx].min() * 100)
            avg_drain = int(drains[y_idx, x_idx].mean() * 100)
            max_drain = int(drains[y_idx, x_idx].max() * 100)
            std_drain = int(drains[y_idx, x_idx].std() * 100)

            biomass = int(orgs[y_idx, x_idx].sum() * 100)

            txt(f"POPULATION  {pop:<7,}   BIOMASS  {biomass:,}", font_sm, (180, 220, 180))
            txt(f"WORLD FOOD  {total_food:<7,}", font_sm, (140, 190, 140))
            stats_y += 4

            # Event Tracking
            if tick_count % 120 == 0:
                if ui_prev['pop'] is not None:
                    if pop < ui_prev['pop'] * 0.5:
                        ui_events.append(f"[{tick_count}] population crash  {ui_prev['pop']} -> {pop}")
                    elif pop > ui_prev['pop'] * 2.0:
                        ui_events.append(f"[{tick_count}] population boom  {ui_prev['pop']} -> {pop}")

                if max_age >= 1000 and 'age_1k' not in ui_flags:
                    ui_events.append(f"[{tick_count}] longevity unlocked - max age > 1,000")
                    ui_flags.add('age_1k')
                elif max_age >= 5000 and 'age_5k' not in ui_flags:
                    ui_events.append(f"[{tick_count}] immortality - max age > 5,000")
                    ui_flags.add('age_5k')

                if ui_prev['d_avg'] is not None:
                    if avg_drain > ui_prev['d_avg'] * 1.5 and avg_drain > 50:
                        ui_events.append(f"[{tick_count}] metabolism surging  {ui_prev['d_avg']} -> {avg_drain}")
                    elif avg_drain < ui_prev['d_avg'] * 0.6 and ui_prev['d_avg'] > 50:
                        ui_events.append(f"[{tick_count}] efficiency breakthrough  {ui_prev['d_avg']} -> {avg_drain}")

                ui_prev['pop'] = pop
                ui_prev['d_avg'] = avg_drain
                ui_events = ui_events[-6:] # Keep last 6

            txt("            MIN     AVG     MAX     STD", font_sm, (100, 110, 140))
            txt(f"Energy      {min_energy:<7} {avg_energy:<7} {max_energy:<7} {std_energy}", font_sm, (180, 180, 200))
            txt(f"Age         {min_age:<7} {avg_age:<7} {max_age:<7} {std_age}", font_sm, (180, 180, 200))
            txt(f"Metabolism  {min_drain:<7} {avg_drain:<7} {max_drain:<7} {std_drain}", font_sm, (180, 180, 200))
            sep()
        else:
            txt(f"POPULATION  0         BIOMASS  0", font_sm, (180, 220, 180))
            txt(f"WORLD FOOD  {total_food:<7,}", font_sm, (140, 190, 140))
            sep()

        # LINEAGES Over Time (Rainbow Stacked Area Chart)
        txt("LINEAGES over time", font, (200, 180, 140))
        rx, ry, rw, rh = px + 8, stats_y, 280, 100

        pygame.draw.rect(screen, (20, 20, 25), (rx, ry, rw, rh))
        if len(lineage_history) > 1:
            all_ids = sorted({aid for frame in lineage_history for aid in frame})
            T = len(lineage_history)
            for t_idx in range(T):
                frame = lineage_history[t_idx]
                total = max(1, sum(frame.values()))
                lx = rx + int(t_idx * rw / T)
                lx1 = rx + int((t_idx + 1) * rw / T)
                bot = ry + rh

                # Draw vertical sliver for each species
                for aid in all_ids:
                    count = frame.get(aid, 0)
                    if count == 0: continue
                    h = int(count / total * rh)
                    if h <= 0: continue
                    top = bot - h
                    pygame.draw.rect(screen, _LINEAGE_COLORS[aid], (lx, top, max(1, lx1 - lx), h))
                    bot = top

            # Top boundary line for total style
            pts = [(rx + i * rw // T, ry + rh - int(sum(lineage_history[i].values()) / max(1, max(sum(f.values()) for f in lineage_history)) * rh)) for i in range(T)]
            pygame.draw.lines(screen, (120, 50, 15), False, pts, 1)

        stats_y += rh + 10

        # Top Living Lineages Legend
        if current_lineages:
            lx_iter = hud_x
            for aid, cnt in sorted(current_lineages.items(), key=lambda kv: -kv[1])[:10]:
                color = _LINEAGE_COLORS[aid]
                pygame.draw.circle(screen, color, (lx_iter + 4, stats_y + 8), 5)
                text_surf = font.render(f"{cnt}", True, color)
                screen.blit(text_surf, (lx_iter + 14, stats_y))
                lx_iter += text_surf.get_width() + 24

                if lx_iter > hud_x + rw - 35:
                    lx_iter = hud_x
                    stats_y += 20

        stats_y += 20

        # Strategy Space (Live PCA over Neural Weights)
        if pop > 3:
            # Flatten population weights to (pop, 15)
            W_pop = weights[:, y_idx, x_idx].T
            W_cen = W_pop - W_pop.mean(axis=0)
            try:
                # Top 2 components via SVD
                u, s, vh = np.linalg.svd(W_cen, full_matrices=False)
                proj = np.dot(W_cen, vh[:2].T)

                screen.blit(font.render("STRATEGY SPACE  (W_wight PCA)", True, (160, 180, 220)), (hud_x, stats_y)); stats_y += 15
                pca_h = 130
                pygame.draw.rect(screen, (20, 20, 25), (hud_x, stats_y, rw, pca_h))
                pygame.draw.rect(screen, (40, 40, 50), (hud_x, stats_y, rw, pca_h), 1)

                p_min = proj.min(axis=0)
                p_max = proj.max(axis=0)
                span = p_max - p_min
                span[span < 1e-6] = 1.0 # prevent div zero

                for i in range(pop):
                    lid = int(np.argmax(W_pop[i, :12]))
                    x_c = int(hud_x + 5 + (proj[i, 0] - p_min[0]) / span[0] * (rw - 10))
                    y_c = int(stats_y + 5 + (proj[i, 1] - p_min[1]) / span[1] * (pca_h - 10))
                    # Draw lineage color dots into the strategy space
                    screen.set_at((x_c, y_c), _LINEAGE_COLORS[lid])
                    screen.set_at((x_c+1, y_c), _LINEAGE_COLORS[lid])
                    screen.set_at((x_c, y_c+1), _LINEAGE_COLORS[lid])
                    screen.set_at((x_c+1, y_c+1), _LINEAGE_COLORS[lid])
            except:
                pass
            stats_y += pca_h + 20

        # Trait Map
        if pop > 0:
            screen.blit(font.render("TRAITS (median | p10-p90 band)", True, (160, 180, 220)), (hud_x, stats_y)); stats_y += 15
            W_pop = weights[:, y_idx, x_idx].T # (pop, 15)
            if pop > 1:
                p10 = np.percentile(W_pop, 10, axis=0)
                med = np.median(W_pop, axis=0)
                p90 = np.percentile(W_pop, 90, axis=0)
            else:
                p10 = med = p90 = W_pop[0]

            trait_names = [
                "Stay:Food", "Stay:Scent", "Stay:nrg",
                "N:Food", "N:Scent", "N:nrg",
                "S:Food", "S:Scent", "S:nrg",
                "E:Food", "E:Scent", "E:nrg",
                "W:Food", "W:Scent", "W:nrg"
            ]

            # Map values from [-8, 8] to pixel width
            v_min, v_max = -8.0, 8.0

            def _lerp_color(t_val, lo=(60, 100, 200), mid=(60, 200, 120), hi=(220, 80, 60)):
                if t_val < 0.5:
                    s = t_val * 2
                    return tuple(int(lo[j] + (mid[j] - lo[j]) * s) for j in range(3))
                s = (t_val - 0.5) * 2
                return tuple(int(mid[j] + (hi[j] - mid[j]) * s) for j in range(3))

            for i, name in enumerate(trait_names):
                # Text
                screen.blit(font_sm.render(f"{name:<11}", True, (140, 140, 160)), (hud_x, stats_y))

                # Bars
                bar_x = hud_x + 90
                bar_w = 180
                row_h = 13

                # Background track for visualization
                pygame.draw.rect(screen, (20, 20, 32), (bar_x, stats_y + 1, bar_w, row_h - 2))

                # Zero line
                z_x = bar_x + int((0 - v_min) / (v_max - v_min) * bar_w)
                pygame.draw.line(screen, (40, 40, 50), (z_x, stats_y), (z_x, stats_y + row_h))

                def sx(v):
                    return max(0, min(bar_w, int((v - v_min) / (v_max - v_min) * bar_w)))

                x10, xmed, x90 = sx(p10[i]), sx(med[i]), sx(p90[i])

                n_med_norm = np.clip((med[i] - v_min) / (v_max - v_min), 0, 1)
                color = _lerp_color(n_med_norm)
                dim = tuple(c // 4 for c in color)

                # Background p10-p90 band (dim rainbow tinted)
                if x90 > x10:
                    pygame.draw.rect(screen, dim, (bar_x + x10, stats_y + 1, x90 - x10, row_h - 2))

                # Median marker
                pygame.draw.rect(screen, color, (bar_x + xmed - 1, stats_y, 3, row_h))

                stats_y += row_h + 1

        # Event History Transparency Window
        if ui_events:
            ev_w = 380
            ev_h = 40 + len(ui_events) * (font_sm.get_height() + 2)
            s = pygame.Surface((ev_w, ev_h), pygame.SRCALPHA)
            s.fill((16, 16, 24, 210)) # Semi-transparent dark
            pygame.draw.rect(s, (40, 40, 60, 255), s.get_rect(), 1)
            screen.blit(s, (10, 10))

            screen.blit(font.render("LIVE EVENTS", True, (200, 180, 140)), (20, 18))
            e_y = 35 + font.get_height() // 2
            line_spacing = font_sm.get_height() + 2
            for ev in ui_events:
                screen.blit(font_sm.render(ev, True, (160, 180, 200)), (20, e_y))
                e_y += line_spacing

        pygame.display.flip()

        clock.tick(60)
        tick_count += 1
        pygame.display.set_caption(f"wight-world - ANE Matrix Evolution | {clock.get_fps():.0f} FPS")

        if max_ticks is not None and tick_count >= max_ticks:
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()