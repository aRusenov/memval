from dataclasses import dataclass
from typing import Dict

@dataclass
class OverlapConfig:
    total_length:          int   = 30
    shared_fraction:       float = 0.40
    shared_position:       float = 0.33
    zone_fraction:         float = 0.50   # fraction of shared corridor covered by odour zone
    zone_offset:           float = 0.25   # start of zone within shared segment (normalised)
    encounter_similarity:  float = 0.0

    def resolve(self) -> Dict[str, int]:
        """
        Return the resolved index boundaries for the shared corridor and odour zone.
        Ensures zone is strictly within the shared corridor boundaries.
        
        Returns:
            Dict[str, int] containing keys:
                - shared_start
                - shared_end
                - zone_start
                - zone_end
        """
        L = self.total_length
        # Length of shared corridor
        sl = max(1, min(L, round(self.shared_fraction * L)))
        
        # Start index of shared corridor (clamp to ensure it fits)
        max_start = L - sl
        ss = max(0, min(max_start, round(self.shared_position * max_start)))
        se = ss + sl
        
        # Length of odour zone (must be at least 1 and fit inside shared corridor)
        zl = max(1, min(sl, round(self.zone_fraction * sl)))
        
        # Start index of odour zone within shared corridor (clamp to stay inside shared corridor)
        max_zone_offset = sl - zl
        zo_idx = max(0, min(max_zone_offset, round(self.zone_offset * max_zone_offset)))
        zs = ss + zo_idx
        ze = zs + zl
        
        return {
            "shared_start": ss,
            "shared_end": se,
            "zone_start": zs,
            "zone_end": ze
        }
