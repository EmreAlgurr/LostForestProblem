import random
import numpy as np
import matplotlib.pyplot as plt

# --- Path representation and simulation for a square forest ---
class SegmentPath:
    def __init__(self, segments):
        """
        segments: list of (length, turn_angle) tuples
        length: distance to walk before turning
        turn_angle: radians relative to current heading
        """
        self.segments = segments

    def simulate(self, P0, alpha0, step=0.1, S=20):
        """
        Simulate a walk from P0=(x,y), heading alpha0,
        in a square of side S centered at the origin.
        Returns total distance until exiting the square.
        """
        x, y = P0
        heading = alpha0
        d = 0.0
        seg_index = 0
        seg_traveled = 0.0
        max_d = sum(length for length, _ in self.segments) + np.sqrt(2)*S

        half = S / 2
        # walk until we leave the square or exceed max_d
        while abs(x) <= half and abs(y) <= half and d <= max_d:
            if seg_index < len(self.segments):
                length, turn = self.segments[seg_index]
                # step forward
                x += step * np.cos(heading)
                y += step * np.sin(heading)
                seg_traveled += step
                d += step
                # segment complete: apply turn
                if seg_traveled >= length:
                    heading += turn
                    seg_index += 1
                    seg_traveled = 0.0
            else:
                # no more segments: go straight
                x += step * np.cos(heading)
                y += step * np.sin(heading)
                d += step

        return d  # exit distance


# --- Genetic Algorithm components ---
def make_individual(num_segments, max_len, max_turn):
    """Random list of (length, turn_angle)"""
    return [(random.uniform(0, max_len),
             random.uniform(-max_turn, max_turn))
            for _ in range(num_segments)]

def random_start_square(S):
    """Uniform random starting point inside square of side S"""
    half = S / 2
    return (random.uniform(-half, half),
            random.uniform(-half, half))

def evaluate(ind, sample_points=20, sample_angles=8, S=20):
    """
    Fitness = negative worst-case exit distance over
    random start positions and headings.
    """
    path = SegmentPath(ind)
    worst = 0.0
    for _ in range(sample_points):
        P0 = random_start_square(S)
        for alpha0 in np.linspace(0, 2*np.pi, sample_angles, endpoint=False):
            d = path.simulate(P0, alpha0, S=S)
            if d > worst:
                worst = d
    return -worst  # we maximize fitness by minimizing worst-case distance

def tournament_select(pop, fitnesses, k=3):
    aspirants = random.sample(list(zip(pop, fitnesses)), k)
    return max(aspirants, key=lambda x: x[1])[0]

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return (p1[:pt] + p2[pt:], p2[:pt] + p1[pt:])

def mutate(ind, max_len, max_turn, mut_rate=0.2):
    for i in range(len(ind)):
        if random.random() < mut_rate:
            length, turn = ind[i]
            length += random.gauss(0, max_len * 0.1)
            turn   += random.gauss(0, max_turn * 0.1)
            # clamp to bounds
            length = max(0, min(length, max_len))
            turn   = max(-max_turn, min(turn, max_turn))
            ind[i] = (length, turn)

def run_evolution(pop_size=50, num_segments=6, max_len=20, max_turn=np.pi/2,
                  generations=30, cross_rate=0.7, mut_rate=0.2, S=20):
    # initialize population
    pop = [make_individual(num_segments, max_len, max_turn) for _ in range(pop_size)]
    history = []

    for gen in range(generations):
        fitnesses = [evaluate(ind, S=S) for ind in pop]
        best_idx = np.argmax(fitnesses)
        best_dist = -fitnesses[best_idx]
        history.append(best_dist)
        print(f"Gen {gen+1}/{generations}: Best worst‐case dist = {best_dist:.2f}")

        # create next generation
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)
            if random.random() < cross_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            mutate(c1, max_len, max_turn, mut_rate)
            mutate(c2, max_len, max_turn, mut_rate)
            new_pop.extend([c1, c2])
        pop = new_pop[:pop_size]

    return history, pop[np.argmax(fitnesses)]


# --- Main execution ---
if __name__ == "__main__":
    # Run evolution
    history, best_ind = run_evolution()

    # Plot fitness progress
    plt.figure(figsize=(6,4))
    plt.plot(history, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best worst-case exit distance")
    plt.title("Evolution Progress (Square Forest)")
    plt.grid(True)
    plt.show()

    # Visualize best strategy from several unknown starts
    S = 20
    path = SegmentPath(best_ind)
    half = S / 2
    t = np.linspace(0, 2*np.pi, 200)
    plt.figure(figsize=(6,6))
    # draw square boundary
    xsq = [-half, half, half, -half, -half]
    ysq = [-half, -half, half, half, -half]
    plt.plot(xsq, ysq, 'orange', lw=2, label='Square boundary')
    # simulate from various starts and headings
    for _ in range(5):
        P0 = random_start_square(S)
        for alpha0 in [0, np.pi/4, np.pi/2]:
            xs, ys = [], []
            x, y = P0
            heading = alpha0
            seg_i = seg_t = d = 0.0
            max_d = sum(l for l,_ in best_ind) + np.sqrt(2)*S
            while abs(x) <= half and abs(y) <= half and d <= max_d:
                xs.append(x); ys.append(y)
                if seg_i < len(best_ind):
                    length, turn = best_ind[int(seg_i)]
                    x += 0.1*np.cos(heading); y += 0.1*np.sin(heading)
                    d += 0.1; seg_t += 0.1
                    if seg_t >= length:
                        heading += turn
                        seg_i += 1
                        seg_t = 0.0
                else:
                    x += 0.1*np.cos(heading); y += 0.1*np.sin(heading)
                    d += 0.1
            plt.plot(xs, ys)
    plt.gca().set_aspect('equal', 'box')
    plt.title("Best Path under Random Unknown Starts/Headings")
    plt.legend()
    plt.show()