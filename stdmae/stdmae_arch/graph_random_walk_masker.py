import numpy as np


class GraphRandomWalkMasker:
    def __init__(self, adj: np.ndarray, mask_ratio: float,
                 walk_len: int = 4, num_walks: int = 8,
                 p: float = 1.0, q: float = 1.0, seed: int = 0):
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
        self.A = (adj > 0).astype(np.float32)
        self.N = adj.shape[0]
        self.deg = self.A.sum(1)
        self.nei = [np.where(self.A[i] > 0)[0] for i in range(self.N)]
        self.r = float(mask_ratio)
        self.walk_len = int(walk_len)
        self.num_walks = int(num_walks)
        self.p, self.q = float(p), float(q)
        self.rng = np.random.default_rng(seed)

    def _one_walk(self, start: int, rng):
        path_nodes = [start]
        path_edges = []
        prev = -1
        cur = start
        for _ in range(self.walk_len):
            nbrs = self.nei[cur]
            if len(nbrs) == 0:
                break
            if prev < 0:
                nxt = int(rng.choice(nbrs))
            else:
                w = []
                for v in nbrs:
                    if v == prev:
                        w.append(1.0 / self.p)
                    elif self.A[prev, v] > 0:
                        w.append(1.0)
                    else:
                        w.append(1.0 / self.q)
                w = np.asarray(w, dtype=np.float32)
                w_sum = w.sum() + 1e-12
                w = w / w_sum
                nxt = int(rng.choice(nbrs, p=w))
            path_edges.append((cur, nxt))
            prev, cur = cur, nxt
            path_nodes.append(cur)
        return path_nodes, path_edges

    def sample(self, seed: int = None):
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)
        need = max(1, int(round(self.N * self.r)))
        visited = set()
        edges = []
        prob = (self.deg + 1e-6) / (self.deg.sum() + 1e-6)
        while len(visited) < need:
            start = int(rng.choice(self.N, p=prob))
            nodes, es = self._one_walk(start, rng)
            visited.update(nodes)
            edges.extend(es)
            for _ in range(self.num_walks - 1):
                if len(nodes) == 0:
                    break
                s2 = int(rng.choice(nodes))
                n2, e2 = self._one_walk(s2, rng)
                visited.update(n2)
                edges.extend(e2)
                if len(visited) >= need:
                    break
        mask_nodes = np.array(sorted(list(visited))[:need], dtype=np.int64)
        walk_edges = np.array(edges, dtype=np.int64) if len(edges) > 0 else np.zeros((0, 2), dtype=np.int64)
        return mask_nodes, walk_edges
