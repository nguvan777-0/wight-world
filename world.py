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
    print("Error: coremltools is required.")
    sys.exit(1)

# --- Configuration ---
W_GRID, H_GRID = 64, 64
RENDER_SCALE = 10
LOG_WIDTH = 260
W_PX, H_PX = W_GRID * RENDER_SCALE, H_GRID * RENDER_SCALE
HUD_WIDTH = 340

CH_FOOD = 1
CH_ENERGY = 1
CH_AGE = 1
CH_ENERGY_DRAIN = 1
CH_WEIGHTS = 15 # 3 senses * 5 intentions = 15 parameters
CH_ORG = CH_ENERGY + CH_AGE + CH_ENERGY_DRAIN + CH_WEIGHTS
CH_TOTAL = CH_FOOD + CH_ORG

MODEL_PATH = "build/ane_wight_world.mlpackage"

WEIGHT_NAMES = [
    "Stay:Food", "Stay:Scent", "Stay:nrg",
    "N:Food", "N:Scent", "N:nrg",
    "S:Food", "S:Scent", "S:nrg",
    "E:Food", "E:Scent", "E:nrg",
    "W:Food", "W:Scent", "W:nrg"
]

def get_lerp_color(t_val, lo=(60, 100, 200), mid=(60, 200, 120), hi=(220, 80, 60)):
    t_val = max(0.0, min(1.0, t_val))
    if t_val < 0.5:
        s = t_val * 2
        return tuple(int(lo[j] + (mid[j] - lo[j]) * s) for j in range(3))
    s = (t_val - 0.5) * 2
    return tuple(int(mid[j] + (hi[j] - mid[j]) * s) for j in range(3))

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
        # Boost this specific weight so it dominates and locks in the founder's lineage color
        weights[lineage_id] = 20.0
    world[0, 4:, gy, gx] = weights

def evaluate_milestones(pop, avg_age, max_age, avg_drain, max_weight_abs, total_food, lineage_counts, prev_state, flags):
    """
    Evaluates ecological statistics to generate emergence events.
    Shared identically between Headless and UI rendering.
    Returns:
        events: List of string descriptions of events that just occurred.
    """
    events = []

    # Initialize missing state keys
    for key in ['pop', 'd_avg', 'food', 'dom']:
        if key not in prev_state:
            prev_state[key] = None
    if 'lineages' not in prev_state:
        prev_state['lineages'] = {}

    if prev_state['pop'] is not None:
        if pop < prev_state['pop'] * 0.5:
            events.append(f"population crash  {prev_state['pop']} -> {pop}")
        elif pop > prev_state['pop'] * 2.0:
            events.append(f"population boom  {prev_state['pop']} -> {pop}")

        # Near-Extinction
        if pop < 50 and prev_state['pop'] >= 50:
            events.append(f"ecosystem collapse threshold: population critical (< 50)")

    if prev_state['food'] is not None:
        if total_food < 100 and prev_state['food'] >= 100:
            events.append("widespread famine: food resources exhausted")

    # Resource Saturation
    if total_food > W_GRID * H_GRID * 0.80 * 100 and 'food_sat' not in flags:
        events.append("resource saturation: global food availability has reached peak capacity")
        flags.add('food_sat')
    elif total_food < W_GRID * H_GRID * 0.50 * 100 and 'food_sat' in flags:
        flags.remove('food_sat')

    # Critical density
    if pop > W_GRID * H_GRID * 0.30 and 'overcrowded' not in flags:
        events.append("critical density warning: ecosystem is becoming severely overcrowded")
        flags.add('overcrowded')
    elif pop < W_GRID * H_GRID * 0.20 and 'overcrowded' in flags:
        flags.remove('overcrowded')

    # Speciation & Survival Tracking
    if pop > 0 and len(lineage_counts) > 0:
        # Dominance Shift
        if pop > 100:
            dom_id = max(lineage_counts, key=lineage_counts.get)
            dom_pct = lineage_counts[dom_id] / pop

            if len(lineage_counts) == 1 and len(prev_state['lineages']) > 1 and dom_pct == 1.0:
                events.append(f"global fixation: lineage {dom_id} has achieved total monoculture")
                prev_state['dom'] = dom_id
            elif dom_pct >= 0.60 and prev_state['dom'] != dom_id:
                events.append(f"lineage {dom_id} has become the dominant lineage ({int(dom_pct*100)}% of population)")
                prev_state['dom'] = dom_id

        # Establish lineages that are viable enough to be tracked
        for lid, count in lineage_counts.items():
            if count >= 1:
                flags.add(f"est_{lid}")

            # Bottleneck Recovery Tracking
            if count < 5 and count > 0:
                flags.add(f"endangered_{lid}")
            elif count > 100 and f"endangered_{lid}" in flags:
                events.append(f"bottleneck recovery: lineage {lid} has resurged from a critical population low")
                flags.remove(f"endangered_{lid}")

    if max_age >= 1000 and 'age_1k' not in flags:
        events.append("longevity unlocked - max age > 1,000")
        flags.add('age_1k')
    elif max_age >= 5000 and 'age_5k' not in flags:
        events.append("immortality - max age > 5,000")
        flags.add('age_5k')

    if avg_age > 1000 and 'stagnation' not in flags:
        events.append("low-turnover ecosystem: average population age has surpassed 1,000")
        flags.add('stagnation')
    elif avg_age < 500 and 'stagnation' in flags:
        flags.remove('stagnation')

    if max_weight_abs > 50.0 and 'extreme_brain' not in flags:
        events.append("phenotypic divergence: extreme neural specialization detected")
        flags.add('extreme_brain')

    if prev_state['d_avg'] is not None:
        if avg_drain > prev_state['d_avg'] * 1.5 and avg_drain > 50:
            events.append(f"metabolism surging  {prev_state['d_avg']} -> {avg_drain}")
        elif avg_drain < prev_state['d_avg'] * 0.6 and prev_state['d_avg'] > 50:
            events.append(f"efficiency breakthrough  {prev_state['d_avg']} -> {avg_drain}")

    # Update state history in place
    prev_state['pop'] = pop
    prev_state['d_avg'] = avg_drain
    prev_state['food'] = total_food
    prev_state['lineages'] = dict(lineage_counts)

    return events
def _save_worker(filepath, world_copy, tick, seed_str, rng_copy, ui_prev_copy, ui_events_copy, history_copy, flags_copy):
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(
        filepath,
        world=world_copy,
        tick=np.array([tick]),
        seed=np.array([seed_str]),
        rng_state=np.array(rng_copy, dtype=object),
        ui_prev=np.array([ui_prev_copy], dtype=object),
        ui_events=np.array(ui_events_copy, dtype=object),
        lineage_history=np.array(history_copy, dtype=object),
        ui_flags=np.array(list(flags_copy), dtype=object)
    )

def save_state(filepath, world, tick, seed_str, rng_state, ui_prev, ui_events, lineage_history, ui_flags):
    import threading
    import copy
    world_copy = world.copy()
    rng_copy = copy.deepcopy(rng_state)
    ui_prev_copy = copy.deepcopy(ui_prev)
    ui_events_copy = list(ui_events)
    history_copy = list(lineage_history)
    flags_copy = set(ui_flags)

    t = threading.Thread(target=_save_worker, args=(
        filepath, world_copy, tick, seed_str, rng_copy, ui_prev_copy, ui_events_copy, history_copy, flags_copy
    ))
    t.start()

def load_state(filepath):
    data = np.load(filepath, allow_pickle=True)
    world = data['world']
    tick = int(data['tick'][0])
    seed_str = str(data['seed'][0])
    np.random.set_state(tuple(data['rng_state']))
    ui_prev = data['ui_prev'][0]
    ui_events = list(data['ui_events'])

    import collections
    lineage_history = collections.deque(list(data['lineage_history']), maxlen=280)
    ui_flags = set(data['ui_flags'])

    return world, tick, seed_str, ui_prev, ui_events, lineage_history, ui_flags


def main():
    parser = argparse.ArgumentParser(description="wight-world neuroevolution engine")
    parser.add_argument("--headless", action="store_true", help="Run in terminal without UI")
    parser.add_argument("--ticks", type=int, default=None, help="Number of ticks to run (default: infinite)")
    parser.add_argument("--interval", type=int, default=500, help="Tick interval for logging in headless mode (default: 500)")
    parser.add_argument("--seed", type=str, default=None, help="Alphanumeric seed for deterministic generation")
    parser.add_argument("--load", type=str, default=None, help="Path to .npz file to load state from")
    args = parser.parse_args()

    is_headless = args.headless
    max_ticks = args.ticks
    interval = args.interval

    import hashlib
    import random
    import string

    if args.load is not None:
        world, start_tick, seed_str, state_prev, state_events, state_history, state_flags = load_state(args.load)
        print(f"\n[ wight-world | Resuming State: {args.load} | Seed: '{seed_str}' ]")
        import collections
    else:
        seed_str = args.seed
        if seed_str is None:
            # Generate a fun readable 6-character random seed string with mixed case
            seed_str = "".join(random.choices(string.ascii_letters, k=6))

        # Deterministically hash the string to a 32-bit int for numpy
        seed_int = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed_int)
        print(f"\n[ wight-world | Seed: '{seed_str}' ]")
        start_tick = 0
        state_prev = {'pop': None, 'd_avg': None}
        state_flags = set([f"est_{i}" for i in range(12)])
        state_events = []
        import collections
        state_history = collections.deque(maxlen=280)

    if not is_headless:
        pygame.init()
        screen = pygame.display.set_mode((LOG_WIDTH + W_PX + HUD_WIDTH, H_PX), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("wight-world")

        # Load fonts
        try:
            # specifically request crisp programming fonts before falling back
            font_family = "sf mono,menlo,monaco,consolas,unifont,monospace"
            font = pygame.font.SysFont(font_family, 14, bold=False)
            font_sm = pygame.font.SysFont(font_family, 12, bold=False)
            font_lg = pygame.font.SysFont(font_family, 18, bold=False)
        except:
            font = pygame.font.SysFont(None, 14, bold=False)
            font_sm = pygame.font.SysFont(None, 12, bold=False)
            font_lg = pygame.font.SysFont(None, 18, bold=False)

        clock = pygame.time.Clock()

        # UI State
        _LINEAGE_COLORS = [
            (255,   0,   0), (255, 128,   0), (255, 255,   0), (128, 255,   0), # Red, Orange, Yellow, Chartreuse
            (  0, 255,   0), (  0, 255, 128), (  0, 255, 255), (  0, 128, 255), # Green, Spring Green, Cyan, Sky Blue
            (  0,   0, 255), (128,   0, 255), (255,   0, 255), (255,   0, 128), # Blue, Purple, Magenta, Rose
        ]

    model = get_model()

    if args.load is None:
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

        _prev = state_prev
        _flags = state_flags

        i = start_tick
        target_tick = start_tick + max_ticks if max_ticks else None
        try:
            while target_tick is None or i < target_tick:
                mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
                out = model.predict({"world": world, "mutation": mutation})
                world = list(out.values())[0]

                world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
                world[0, 0] = np.clip(world[0,0], 0.0, 1.0)

                if i % interval == 0 or (target_tick and i == target_tick - 1):
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

                        total_food = int(world[0, 0].sum() * 100)
                        alive_weights = world[0, 4:16][:, mask]
                        lineages = np.argmax(alive_weights, axis=0) # shape (pop,)
                        unique_lids, counts = np.unique(lineages, return_counts=True)
                        lineage_counts = dict(zip([int(k) for k in unique_lids], [int(c) for c in counts]))

                        if _prev.get('lineages'):
                            for lid in _prev['lineages']:
                                if lid not in lineage_counts:
                                    print(f"          ↳ lineage {lid} has gone extinct")

                        notes = evaluate_milestones(pop, avg_age, max_age, avg_drain, float(np.abs(alive_weights).max()), total_food, lineage_counts, _prev, _flags)
                        for note in notes:
                            print(f"          ↳ {note}")

                    else:
                        if _prev.get('lineages'):
                            for lid in _prev['lineages']:
                                print(f"          ↳ lineage {lid} has gone extinct")
                            _prev['lineages'] = {}
                        print(f"{i:8d}      EXTINCT")
                        break

                # Headless runs at maximum computational speed, so a 5,000 tick interval is roughly ~2-4 seconds.
                if i > 0 and i % 5000 == 0:
                    import collections
                    save_state(f"saves/wight-world_{seed_str}.npz", world, i, seed_str, np.random.get_state(), _prev, [], collections.deque(), _flags)

                i += 1
        except KeyboardInterrupt:
            print("\nHeadless run interrupted by user.")

        t1 = time.time()
        fps = i / max(1e-6, t1 - t0)
        print("─" * 72)
        print(f"{i:,} ticks  {t1-t0:.1f}s  {fps:,.0f} t/s\n")
        return

    running = True
    tick_count = start_tick
    paused = False
    speed_mode = 1
    print("\nSimulation Started!")
    print(" - Clicking spawns an organism.")
    print(" - S takes a screenshot.")
    print(" - SPACE pauses/resumes.")
    print(" - Keys 1-5 adjust speed (1x, 5x, 20x, 100x, MAX).")

    ui_prev = state_prev
    ui_flags = state_flags
    ui_events = state_events
    lineage_history = state_history
    ui_events_scroll = 0
    last_inspected_wight = None

    last_event_tick = tick_count
    last_autosave_tick = start_tick
    flash_r = 0
    flash_s = 0

    while running:
        if flash_r > 0: flash_r -= 1
        if flash_s > 0: flash_s -= 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or getattr(event, 'unicode', '') == ' ':
                    paused = not paused
                elif event.key in (pygame.K_1, pygame.K_KP1) or getattr(event, 'unicode', '') == '1':
                    speed_mode = 1
                    paused = False
                elif event.key in (pygame.K_2, pygame.K_KP2) or getattr(event, 'unicode', '') == '2':
                    speed_mode = 5
                    paused = False
                elif event.key in (pygame.K_3, pygame.K_KP3) or getattr(event, 'unicode', '') == '3':
                    speed_mode = 20
                    paused = False
                elif event.key in (pygame.K_4, pygame.K_KP4) or getattr(event, 'unicode', '') == '4':
                    speed_mode = 100
                    paused = False
                elif event.key in (pygame.K_5, pygame.K_KP5) or getattr(event, 'unicode', '') == '5':
                    speed_mode = 'MAX'
                    paused = False
                elif event.key == pygame.K_r or getattr(event, 'unicode', '').lower() == 'r':
                    flash_r = 15
                    world = init_world()
                    for i in range(12):
                        drop_organism(world, np.random.randint(W_GRID), np.random.randint(H_GRID), lineage_id=i)
                    tick_count = 0
                    last_event_tick = 0
                    ui_prev = {'pop': None, 'd_avg': None}
                    ui_flags.clear()
                    ui_events.clear()
                    ui_events_scroll = 0
                    lineage_history.clear()
                    last_inspected_wight = None
                elif event.key == pygame.K_s or getattr(event, 'unicode', '').lower() == 's':
                    flash_s = 15
                    import os
                    os.makedirs("screenshots", exist_ok=True)
                    filename = f"screenshots/wight-world_{seed_str}_{tick_count}.png"
                    pygame.image.save(screen, filename)
                    print(f"Saved screenshot: {filename}")
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx < LOG_WIDTH and my < H_PX - 320: # Scrolling in the Live Events area
                    ui_events_scroll += event.y
                    # Will clamp scroll bounds during rendering
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if LOG_WIDTH <= event.pos[0] < LOG_WIDTH + W_PX:
                    gx = np.clip((event.pos[0] - LOG_WIDTH) // RENDER_SCALE, 0, W_GRID - 1)
                    gy = np.clip(event.pos[1] // RENDER_SCALE, 0, H_GRID - 1)
                    # Drop an organism with random genes
                    drop_organism(world, gx, gy)
                    # Drop some large food nearby
                    world[0, 0, max(0,gy-2):min(H_GRID,gy+2), max(0,gx-2):min(W_GRID,gx+2)] += 1.0

        # Hover tracking for Wight Inspector
        mx, my = pygame.mouse.get_pos()
        hover_gx, hover_gy = None, None
        if LOG_WIDTH <= mx < LOG_WIDTH + W_PX and 0 <= my < H_PX:
            hover_gx = np.clip((mx - LOG_WIDTH) // RENDER_SCALE, 0, W_GRID - 1)
            hover_gy = np.clip(my // RENDER_SCALE, 0, H_GRID - 1)

        if not paused:
            if speed_mode == 'MAX':
                t_start = time.time()
                # Process heavily for ~33ms before yielding back to the render loop
                while time.time() - t_start < 0.033:
                    mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
                    out = model.predict({"world": world, "mutation": mutation})
                    world = list(out.values())[0]
                    world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
                    world[0, 0] = np.clip(world[0,0], 0.0, 1.0)
                    tick_count += 1
            else:
                for _ in range(speed_mode):
                    mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
                    out = model.predict({"world": world, "mutation": mutation})
                    world = list(out.values())[0]
                    world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
                    world[0, 0] = np.clip(world[0,0], 0.0, 1.0)
                    tick_count += 1

        # Adaptive auto-save interval based on game speed to ensure ~1 save per second
        if speed_mode == 1: save_interval = 60
        elif speed_mode == 5: save_interval = 300
        elif speed_mode == 20: save_interval = 1200
        else: save_interval = 5000

        if tick_count > 0 and tick_count - last_autosave_tick >= save_interval:
            last_autosave_tick = tick_count
            save_state(f"saves/wight-world_{seed_str}.npz", world, tick_count, seed_str, np.random.get_state(), ui_prev, ui_events, lineage_history, ui_flags)

        # Render visual channels
        t = world[0]

        # 1. Background Space & Food Surface
        screen.fill((15, 15, 20)) # Dark void
        rgba = np.zeros((H_GRID, W_GRID, 3), dtype=np.uint8)
        rgba[..., 1] = np.clip(t[0] * 70.0, 0, 255) # Dim green for food algae
        rgba = np.transpose(rgba, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rgba)
        surf_scaled = pygame.transform.scale(surf, (W_PX, H_PX))
        # Clear the left dock
        pygame.draw.rect(screen, (16, 16, 24), (0, 0, LOG_WIDTH, H_PX))
        pygame.draw.line(screen, (50, 50, 80), (LOG_WIDTH - 1, 0), (LOG_WIDTH - 1, H_PX), 1)
        # Blit main matrix
        screen.blit(surf_scaled, (LOG_WIDTH, 0))

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
            # argmax over the first 12 weights establishes stable lineages.
            lid = int(np.argmax(w[:12]))
            current_lineages[lid] += 1

            c = _LINEAGE_COLORS[lid]

            # Position & pulsating size based on energy
            cx = LOG_WIDTH + x * RENDER_SCALE + RENDER_SCALE // 2
            cy = y * RENDER_SCALE + RENDER_SCALE // 2
            r = int((0.5 + energy * 0.5) * RENDER_SCALE * 0.9)

            # Body & Core Outline
            pygame.draw.circle(screen, c, (cx, cy), r)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r, max(1, r//3))

            # Orientation heuristic: sweeping white line mapped to neural weight 0
            ax = cx + int(np.cos(w[0] * np.pi) * r * 1.8)
            ay = cy + int(np.sin(w[0] * np.pi) * r * 1.8)
            pygame.draw.line(screen, (255, 255, 255), (cx, cy), (ax, ay), max(1, RENDER_SCALE//8))

        if not paused:
            # Check for newly extinct lineages using the UI render history
            if len(lineage_history) > 0:
                prev_lineages = lineage_history[-1]
                for lid in prev_lineages:
                    if lid not in current_lineages:
                        ui_events.append(f"[{tick_count}] lineage {lid} has gone extinct")

                if len(ui_events) > 200:
                    ui_events = ui_events[-200:]

            lineage_history.append(dict(current_lineages))

        # 3. Draw Side HUD
        px = LOG_WIDTH + W_PX
        # Background track for HUD
        pygame.draw.rect(screen, (16, 16, 28), (px, 0, HUD_WIDTH, H_PX))
        pygame.draw.line(screen, (50, 50, 80), (px, 0), (px, H_PX), 1)

        hud_x = px + 10
        stats_y = 10

        def txt(s, f=None, color=(230, 230, 240)):
            nonlocal stats_y
            screen.blit((f or font).render(str(s), True, color), (px + 10, stats_y))
            stats_y += (f or font).get_height() + 2

        def trow(vals, f=None, color=(230, 230, 240)):
            nonlocal stats_y
            cx = px + 10
            col_widths = [100, 60, 60, 60, 60]
            f = f or font_sm
            for i, v in enumerate(vals):
                screen.blit(f.render(str(v), True, color), (cx, stats_y))
                cx += col_widths[i]
            stats_y += f.get_height() + 2

        def sep():
            nonlocal stats_y
            pygame.draw.line(screen, (40, 40, 60), (px + 8, stats_y + 2), (px + HUD_WIDTH - 8, stats_y + 2), 1)
            stats_y += 8

        # Title
        title_surf = font_lg.render("WIGHT-WORLD", True, (255, 255, 255))
        screen.blit(title_surf, (px + 10, stats_y))

        seed_surf = font_sm.render(f"s:{seed_str}", True, (230, 230, 245))
        tick_surf = font_sm.render(f"t:{tick_count:,}", True, (230, 230, 245))
        ane_surf  = font_sm.render("[ANE]", True, (230, 230, 245))

        total_w = seed_surf.get_width() + tick_surf.get_width() + ane_surf.get_width()
        avail_space = HUD_WIDTH - 20 - title_surf.get_width()

        # 3 gaps (between Title-S, S-T, and T-ANE)
        gap = max(6, (avail_space - total_w) // 3)

        # Vertically center with title using heights
        y_pos = stats_y + max(0, (title_surf.get_height() - font_sm.get_height()) // 2)

        cx = px + 10 + title_surf.get_width() + gap
        screen.blit(seed_surf, (cx, y_pos))

        cx += seed_surf.get_width() + gap
        screen.blit(tick_surf, (cx, y_pos))

        cx += tick_surf.get_width() + gap
        screen.blit(ane_surf, (cx, y_pos))

        stats_y += font_lg.get_height() + 2
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

            c_g = (200, 255, 200)
            screen.blit(font_sm.render(f"POP: {pop:,}", True, c_g), (px + 10, stats_y))
            screen.blit(font_sm.render(f"BIO: {biomass:,}", True, c_g), (px + 120, stats_y))
            screen.blit(font_sm.render(f"FOOD: {total_food:,}", True, c_g), (px + 230, stats_y))
            stats_y += font_sm.get_height() + 8

            # Event Tracking
            if tick_count // 120 > last_event_tick // 120:
                last_event_tick = tick_count

                new_events = evaluate_milestones(
                    pop, avg_age, max_age, avg_drain, float(np.abs(weights[:, y_idx, x_idx]).max()), total_food, dict(current_lineages), ui_prev, ui_flags
                )
                for ev in new_events:
                    ui_events.append(f"[{tick_count}] {ev}")

                ui_events = ui_events[-200:] # Keep last 200 in raw history

            trow(["", "MIN", "AVG", "MAX", "STD"], font_sm, (170, 180, 210))
            trow(["Energy", min_energy, avg_energy, max_energy, std_energy], font_sm, (230, 230, 245))
            trow(["Age", min_age, avg_age, max_age, std_age], font_sm, (230, 230, 245))
            trow(["Metabolism", min_drain, avg_drain, max_drain, std_drain], font_sm, (230, 230, 245))
            sep()
        else:
            c_g = (200, 255, 200)
            screen.blit(font_sm.render(f"POP: 0", True, c_g), (px + 10, stats_y))
            screen.blit(font_sm.render(f"BIO: 0", True, c_g), (px + 120, stats_y))
            screen.blit(font_sm.render(f"FOOD: {total_food:,}", True, c_g), (px + 230, stats_y))
            stats_y += font_sm.get_height() + 8

            # Event Tracking: Trigger final extinction events if pop just hit 0
            if not paused and ui_prev['pop'] is not None and ui_prev['pop'] > 0:
                if len(lineage_history) > 0:
                    prev_lineages = lineage_history[-1]
                    for lid in prev_lineages:
                        ui_events.append(f"[{tick_count}] lineage {lid} has gone extinct")

                    if prev_lineages:
                        ui_events.append(f"[{tick_count}] global extinction: population has reached zero")

                ui_events = ui_events[-200:]
                ui_prev['pop'] = 0

            sep()

        # LINEAGES Over Time (Rainbow Stacked Area Chart)
        txt("LINEAGES over time", font, (255, 210, 120))
        rx, ry, rw, rh = px + 8, stats_y, HUD_WIDTH - 16, 95

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

                # Draw vertical sliver for each lineage
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

        stats_y += rh + 5

        # Top Living Lineages Legend
        if current_lineages:
            lx_iter = hud_x
            for aid, cnt in sorted(current_lineages.items(), key=lambda kv: -kv[1])[:10]:
                color = _LINEAGE_COLORS[aid]
                pygame.draw.circle(screen, color, (lx_iter + 4, stats_y + 6), 5)
                text_surf = font.render(f"{cnt}", True, color)
                screen.blit(text_surf, (lx_iter + 14, stats_y))
                lx_iter += text_surf.get_width() + 24

                if lx_iter > hud_x + rw - 35:
                    lx_iter = hud_x
                    stats_y += 18

            stats_y += 18
        else:
            stats_y += 18

        # Strategy Space (Live PCA over Neural Weights)
        if pop > 3:
            # Flatten population weights to (pop, 15)
            W_pop = weights[:, y_idx, x_idx].T
            W_cen = W_pop - W_pop.mean(axis=0)
            try:
                # Top 2 components via SVD
                u, s, vh = np.linalg.svd(W_cen, full_matrices=False)
                proj = np.dot(W_cen, vh[:2].T)

                screen.blit(font.render("STRATEGY SPACE  (W_wight PCA)", True, (220, 230, 255)), (hud_x, stats_y)); stats_y += 15
                pca_h = 115
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
            stats_y += pca_h + 10

        # Neural Weights Map
        if pop > 0:
            screen.blit(font.render("NEURAL WEIGHTS (median | p10-p90 band)", True, (220, 230, 255)), (hud_x, stats_y)); stats_y += 15
            W_pop = weights[:, y_idx, x_idx].T # (pop, 15)
            if pop > 1:
                p10 = np.percentile(W_pop, 10, axis=0)
                med = np.median(W_pop, axis=0)
                p90 = np.percentile(W_pop, 90, axis=0)
            else:
                p10 = med = p90 = W_pop[0]

            # Map values from [-8, 8] to pixel width
            v_min, v_max = -8.0, 8.0

            # Compute row height based on remaining space
            available_h = H_PX - stats_y - 25
            row_interval = max(11.0, available_h / 15.0)
            row_h = int(row_interval) - 1 # Leaves 1px pad between rows

            for i, name in enumerate(WEIGHT_NAMES):
                y_baseline = stats_y + int(i * row_interval)

                # Text
                screen.blit(font_sm.render(f"{name:<11}", True, (200, 200, 220)), (hud_x, y_baseline))

                # Bars
                bar_x = hud_x + 90
                bar_w = 260

                # Background track for visualization
                pygame.draw.rect(screen, (20, 20, 32), (bar_x, y_baseline + 1, bar_w, row_h - 2))

                # Zero line
                z_x = bar_x + int((0 - v_min) / (v_max - v_min) * bar_w)
                pygame.draw.line(screen, (40, 40, 50), (z_x, y_baseline), (z_x, y_baseline + row_h))

                def sx(v):
                    return max(0, min(bar_w, int((v - v_min) / (v_max - v_min) * bar_w)))

                x10, xmed, x90 = sx(p10[i]), sx(med[i]), sx(p90[i])

                n_med_norm = np.clip((med[i] - v_min) / (v_max - v_min), 0, 1)
                color = get_lerp_color(n_med_norm)
                dim = tuple(c // 4 for c in color)

                # Background p10-p90 band (dim rainbow tinted)
                if x90 > x10:
                    pygame.draw.rect(screen, dim, (bar_x + x10, y_baseline + 1, x90 - x10, row_h - 2))

                # Median marker
                pygame.draw.rect(screen, color, (bar_x + xmed - 1, y_baseline, 3, row_h))

            stats_y = H_PX - 20

        # Footer Controls
        footer_y = H_PX - 20
        pygame.draw.line(screen, (40, 40, 60), (px + 8, footer_y - 5), (px + HUD_WIDTH - 8, footer_y - 5), 1)

        c_x = px + 10
        controls = [
            (f"SPC:{'▶ ' if paused else '▌▌'}", paused, (255, 100, 100)), # Red
            ("R:↻", flash_r > 0, (255, 165, 0)), # Orange
            ("S:img", flash_s > 0, (255, 255, 0)), # Yellow
            ("1:1x", not paused and speed_mode == 1, (100, 255, 100)), # Green
            ("2:5x", not paused and speed_mode == 5, (100, 255, 255)), # Cyan
            ("3:20x", not paused and speed_mode == 20, (100, 150, 255)), # Blue
            ("4:100x", not paused and speed_mode == 100, (180, 100, 255)), # Purple
            ("5:MAX", not paused and speed_mode == 'MAX', (255, 100, 255)) # Pink
        ]

        for text, is_active, active_color in controls:
            color = active_color if is_active else (120, 120, 130)
            surf = font_sm.render(text, True, color)
            screen.blit(surf, (c_x, footer_y))
            c_x += surf.get_width() + 8

        # 4. Left Dock - Live Events
        # Clip to prevent event ticker text from bleeding into the matrix
        screen.set_clip(pygame.Rect(0, 0, LOG_WIDTH - 2, H_PX))

        box_x = 10
        box_y = 10
        screen.blit(font.render("LIVE EVENTS", True, (255, 210, 120)), (box_x, box_y))

        # Word Wrap Logic
        wrapped_lines = []
        max_width = LOG_WIDTH - 20
        for ev in ui_events:
            # Color coding keywords
            ev_lower = ev.lower()
            if any(w in ev_lower for w in ["longevity", "immortality"]):
                c = (255, 215, 0) # Gold
            elif any(w in ev_lower for w in ["crash", "extinct", "famine", "collapse"]):
                c = (255, 80, 80) # Red
            elif any(w in ev_lower for w in ["boom", "surging", "breakthrough"]):
                c = (100, 255, 100) # Green
            elif "dominant" in ev_lower:
                c = (100, 180, 255) # Blue
            else:
                c = (220, 230, 250) # Standard

            words = ev.split(' ')
            current_line = []
            for word in words:
                current_line.append(word)
                if font_sm.size(' '.join(current_line))[0] > max_width:
                    current_line.pop()
                    if current_line:
                        wrapped_lines.append((' '.join(current_line), c))
                    current_line = [word]
            if current_line:
                wrapped_lines.append((' '.join(current_line), c))

        line_spacing = font_sm.get_height() + 4
        # Calculate max lines that fit above the inspector panel
        max_lines_fit = (H_PX - 320 - box_y - 25) // line_spacing

        max_scroll = max(0, len(wrapped_lines) - max_lines_fit)
        ui_events_scroll = max(0, min(max_scroll, ui_events_scroll))

        # ui_events_scroll > 0 means scrolled UP to see older history
        start_idx = max_scroll - ui_events_scroll

        e_y = box_y + 25
        visible_lines = wrapped_lines[start_idx:start_idx + max_lines_fit]
        for line_text, line_color in visible_lines:
            screen.blit(font_sm.render(line_text, True, line_color), (box_x, e_y))
            e_y += line_spacing

        # Wight Inspector
        insp_y = H_PX - 320
        pygame.draw.line(screen, (50, 50, 80), (10, insp_y - 15), (LOG_WIDTH - 10, insp_y - 15), 1)
        screen.blit(font.render("WIGHT INSPECTOR", True, (255, 210, 120)), (10, insp_y))

        # Remove clip explicitly before Matrix/Tracking logic that draws into main screen
        screen.set_clip(None)

        is_hovering_live_cell = False
        if hover_gx is not None and hover_gy is not None:
            # Highlight hovered cell in matrix
            pygame.draw.rect(screen, (255, 255, 255), (LOG_WIDTH + hover_gx * RENDER_SCALE, hover_gy * RENDER_SCALE, RENDER_SCALE, RENDER_SCALE), 1)

            # If hovered over a valid organism, snapshot it
            e_val = orgs[hover_gy, hover_gx]
            if e_val > 0.0:
                is_hovering_live_cell = True
                last_inspected_wight = {
                    'x': int(hover_gx),
                    'y': int(hover_gy),
                    'e': float(e_val),
                    'a': float(ages[hover_gy, hover_gx]),
                    'd': float(drains[hover_gy, hover_gx]),
                    'w': weights[:, hover_gy, hover_gx].copy(),
                    'status': 'alive'
                }

        # Track the previously inspected wight if we aren't hovering a new one
        if not is_hovering_live_cell and last_inspected_wight is not None:
            diff = weights - last_inspected_wight['w'][:, None, None]
            dist = np.sum(diff**2, axis=0) # shape (H_GRID, W_GRID)
            # Only consider living cells
            dist[orgs == 0] = float('inf')

            min_dist = np.min(dist)
            # Threshold ensures we track the exact parent, not a mutated child
            if min_dist < 0.1:
                min_idx = np.argmin(dist)
                ny, nx = np.unravel_index(min_idx, dist.shape)
                last_inspected_wight = {
                    'x': int(nx),
                    'y': int(ny),
                    'e': float(orgs[ny, nx]),
                    'a': float(ages[ny, nx]),
                    'd': float(drains[ny, nx]),
                    'w': weights[:, ny, nx].copy(),
                    'status': 'alive'
                }
            else:
                last_inspected_wight['status'] = 'DEAD' # The tracked wight died

        if last_inspected_wight is not None:
            wight = last_inspected_wight

            # Draw tracking crosshair on the matrix
            if wight.get('status') != 'DEAD':
                pygame.draw.rect(screen, (255, 100, 100), (LOG_WIDTH + wight['x'] * RENDER_SCALE, wight['y'] * RENDER_SCALE, RENDER_SCALE, RENDER_SCALE), 2)

            w_val = wight['w']
            lid = int(np.argmax(w_val[:12]))
            c = _LINEAGE_COLORS[lid]

            # Big portrait
            pw, py = 45, insp_y + 45
            pygame.draw.circle(screen, c, (pw, py), 30)
            pygame.draw.circle(screen, (255, 255, 255), (pw, py), 30, 2)
            ax = pw + int(np.cos(w_val[0] * np.pi) * 30 * 1.0)
            ay = py + int(np.sin(w_val[0] * np.pi) * 30 * 1.0)
            pygame.draw.line(screen, (255, 255, 255), (pw, py), (ax, ay), 2)

            # Stats Grid (Show DEAD tag if deceased)
            spec_str = f"Lineage {lid}"
            stat_color = (220, 230, 250)
            if wight.get('status') == 'DEAD':
                spec_str += " [DEAD]"
                c = (255, 80, 80)
                stat_color = (150, 150, 160)
                # Draw a cross over the portrait if dead
                pygame.draw.line(screen, (255, 50, 50), (pw-20, py-20), (pw+20, py+20), 3)
                pygame.draw.line(screen, (255, 50, 50), (pw-20, py+20), (pw+20, py-20), 3)

            tx1, tx2 = 85, 175
            screen.blit(font.render(spec_str, True, c), (tx1, insp_y + 10))

            # Column 1
            screen.blit(font_sm.render(f"Pos: ({wight['x']}, {wight['y']})", True, stat_color), (tx1, insp_y + 30))
            screen.blit(font_sm.render(f"Age: {int(wight['a'])}", True, stat_color), (tx1, insp_y + 45))
            # Column 2
            screen.blit(font_sm.render(f"Nrg: {wight['e']:.2f}", True, stat_color), (tx2, insp_y + 30))
            screen.blit(font_sm.render(f"Met: {wight['d']:.2f}", True, stat_color), (tx2, insp_y + 45))

            # Brain Bar Graph - Matrix Layout (Food, Scent, Nrg across N,S,E,W,Stay)
            # Headers
            w_y = insp_y + 75
            head_c = (150, 150, 180)
            screen.blit(font_sm.render("Food", True, head_c), (55, w_y))
            screen.blit(font_sm.render("Scent", True, head_c), (105, w_y))
            screen.blit(font_sm.render("Nrg", True, head_c), (155, w_y))

            dirs = ["Stay", "N", "S", "E", "W"]
            for row_idx, d_name in enumerate(dirs):
                r_y = w_y + 20 + (row_idx * 16)
                screen.blit(font_sm.render(f"{d_name:>4}:", True, (200, 200, 220)), (10, r_y))

                # Each direction has 3 components (Food, Scent, nrg)
                for col_idx in range(3):
                    w_idx = row_idx * 3 + col_idx
                    val = w_val[w_idx]
                    percent = (val + 8) / 16.0

                    bar_w = int(percent * 35)
                    bar_w = max(0, min(35, bar_w))
                    bx = 55 + (col_idx * 50)

                    bg_rect = (bx, r_y + 2, 35, 8)
                    pygame.draw.rect(screen, (40, 40, 50), bg_rect) # background groove

                    bar_color = get_lerp_color(percent)
                    pygame.draw.rect(screen, bar_color, (bx, r_y + 2, bar_w, 8))

                    # Optional: tiny value overlay
                    # tiny_v = int(val)
                    # screen.blit(font_sm.render(f"{tiny_v}", True, (255,255,255)), (bx+2, r_y-1))

            # Draw an evaluation sparkline or extra context if alive
            if wight.get('status') != 'DEAD':
                fav_dir_idx = int(np.argmax([np.sum(w_val[i*3:(i+1)*3]) for i in range(5)]))
                fav_dir = dirs[fav_dir_idx]
                eval_str = f"Intent: {fav_dir}"
                screen.blit(font_sm.render(eval_str, True, (120, 255, 120)), (10, r_y + 25))
        else:
            screen.blit(font_sm.render("Hover over matrix to inspect.", True, (120, 120, 130)), (10, insp_y + 25))

        pygame.display.flip()

        clock.tick(60)
        pygame.display.set_caption(f"wight-world - ANE Matrix Evolution | {clock.get_fps():.0f} FPS")

        if max_ticks is not None and tick_count >= max_ticks:
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()