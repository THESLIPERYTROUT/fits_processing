"""
FilterChain — assembles and runs the ordered filter pipeline.

Each filter is a pure function with signature (np.ndarray, FilterParams) -> np.ndarray.
FilterChain.from_config() wires them in the correct fixed order based on which
filters are enabled in the config.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from streakiller.config.schema import EnabledFilters, FilterParams
from streakiller.models.streak import FilterStageSnapshot

logger = logging.getLogger(__name__)

# Type alias for filter functions
FilterFn = Callable[[np.ndarray, FilterParams], np.ndarray]


class FilterChain:
    """
    Ordered list of filter steps.  Each step has a name and a pure function.

    Usage::

        chain = FilterChain.from_config(config.enabled_line_filters)
        final_lines, snapshots = chain.run(raw_lines, config.filter_params)
    """

    def __init__(self, steps: list[tuple[str, FilterFn]]) -> None:
        self._steps = steps

    @classmethod
    def from_config(cls, enabled: EnabledFilters) -> "FilterChain":
        """
        Build the chain from the enabled-filters config.

        Order is fixed: midpoint → angle → colinear → endpoint → length.
        This order matters — colinear merge should run before endpoint/length
        pruning to avoid prematurely discarding segments that would merge.
        """
        from streakiller.filters.midpoint import midpoint_filter
        from streakiller.filters.angle import angle_filter
        from streakiller.filters.colinear import colinear_merge
        from streakiller.filters.endpoint import endpoint_filter
        from streakiller.filters.length import length_filter

        steps: list[tuple[str, FilterFn]] = []
        if enabled.midpoint_filter:
            steps.append(("midpoint_filter", midpoint_filter))
        if enabled.line_angle:
            steps.append(("angle_filter", angle_filter))
        if enabled.colinear_filter:
            steps.append(("colinear_merge", colinear_merge))
        if enabled.endpoint_filter:
            steps.append(("endpoint_filter", endpoint_filter))
        if enabled.length_filter:
            steps.append(("length_filter", length_filter))

        return cls(steps)

    def run(
        self,
        lines: np.ndarray,
        params: FilterParams,
    ) -> tuple[np.ndarray, list[FilterStageSnapshot]]:
        """
        Apply each filter in order.

        Parameters
        ----------
        lines : ndarray, shape (N, 1, 4)
        params : FilterParams

        Returns
        -------
        (final_lines, snapshots)
            final_lines: shape (M, 1, 4)
            snapshots: one entry per filter stage, including line counts
        """
        current = lines
        snapshots: list[FilterStageSnapshot] = []

        for name, fn in self._steps:
            before = len(current)
            current = fn(current, params)
            after = len(current)
            snapshots.append(
                FilterStageSnapshot(
                    stage_name=name,
                    lines_before=before,
                    lines_after=after,
                    lines=current.copy(),
                )
            )
            logger.info("Filter %-20s  %d → %d lines", name, before, after)

        return current, snapshots

    @property
    def step_names(self) -> list[str]:
        return [name for name, _ in self._steps]
