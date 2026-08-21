"""
Clutter (false measurement) generator.

Matches Musicki & Evans (2002) Section 4:
  - Surveillance region: 1000m (x) x 400m (y)
  - Background clutter density: 1.0e-4 / scan / m^2  (Poisson)
  - Two high-density patches with 7x the background density:
      patch 1: (x in [330, 490], y in [203, 303])
      patch 2: (x in [718, 840], y in [100, 200])
"""
import numpy as np


class ClutterModel:
    def __init__(
        self,
        region=(0, 1000, 0, 400),
        base_density=1.0e-4,
        patch_multiplier=7.0,
        patches=((330, 490, 203, 303), (718, 840, 100, 200)),
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = region
        self.base_density = base_density
        self.patch_multiplier = patch_multiplier
        self.patches = patches  # list of (x_min, x_max, y_min, y_max)

        self.region_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.patch_areas = [
            (px1 - px0) * (py1 - py0) for (px0, px1, py0, py1) in patches
        ]

    def _expected_counts(self):
        """Expected number of clutter points in background vs. each patch,
        accounting for patch areas being carved out of the background."""
        patch_total_area = sum(self.patch_areas)
        bg_area = self.region_area - patch_total_area
        bg_count = self.base_density * bg_area
        patch_counts = [
            self.base_density * self.patch_multiplier * a for a in self.patch_areas
        ]
        return bg_count, patch_counts

    def generate(self, rng):
        """Return list of (x, y) clutter measurements for one scan."""
        clutter = []
        bg_count, patch_counts = self._expected_counts()

        # Background clutter (rejection-sample out of patch regions)
        n_bg = rng.poisson(bg_count)
        added = 0
        while added < n_bg:
            x = rng.uniform(self.x_min, self.x_max)
            y = rng.uniform(self.y_min, self.y_max)
            if not self._in_any_patch(x, y):
                clutter.append((x, y))
                added += 1

        # Patch clutter
        for (px0, px1, py0, py1), pc in zip(self.patches, patch_counts):
            n_p = rng.poisson(pc)
            for _ in range(n_p):
                x = rng.uniform(px0, px1)
                y = rng.uniform(py0, py1)
                clutter.append((x, y))

        return [np.array(c) for c in clutter]

    def _in_any_patch(self, x, y):
        for (px0, px1, py0, py1) in self.patches:
            if px0 <= x <= px1 and py0 <= y <= py1:
                return True
        return False

    def density_at(self, x, y):
        """Local clutter density at a point (used for JIPDA's a-priori clutter est.)."""
        if self._in_any_patch(x, y):
            return self.base_density * self.patch_multiplier
        return self.base_density
