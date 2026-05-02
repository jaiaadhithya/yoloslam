from dataclasses import dataclass
from enum import Enum


@dataclass
class LandingSMConfig:
    """Tunable thresholds (defaults match original hard-coded behavior)."""

    survey_duration_s: float = 6.0
    zone_score_evaluate_commit: float = 5.0
    zone_score_search_commit: float = 2.0
    approach_xy_tolerance_m: float = 1.0
    align_xy_tolerance_m: float = 0.15
    align_stable_s: float = 2.0
    descend_xy_drift_m: float = 0.5
    touchdown_altitude_m: float = 0.2
    target_lost_timeout_s: float = 3.0
    abort_resurvey_s: float = 2.0


class LandingState(str, Enum):
    SURVEY = "SURVEY"
    EVALUATE = "EVALUATE"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    DESCEND = "DESCEND"
    ABORT = "ABORT"
    LANDED = "LANDED"


class LandingStateMachine:
    def __init__(self, config: LandingSMConfig | None = None) -> None:
        self.cfg = config or LandingSMConfig()
        self.state = LandingState.SURVEY
        self.align_stable_seconds = 0.0
        self.target_lost_seconds = 0.0
        self.survey_elapsed = 0.0
        self.abort_elapsed = 0.0
        # Set each update: non-empty only when a transition occurred this timestep.
        self.transition_reason: str = ""

    def update(
        self,
        has_valid_zone: bool,
        zone_score: float,
        xy_error: float,
        altitude: float,
        dt: float,
        intrusion_detected: bool,
    ) -> LandingState:
        self.transition_reason = ""
        if not has_valid_zone:
            self.target_lost_seconds += dt
        else:
            self.target_lost_seconds = 0.0

        c = self.cfg
        if self.state == LandingState.SURVEY:
            self.survey_elapsed += dt
            if self.survey_elapsed >= c.survey_duration_s:
                self.state = LandingState.EVALUATE
                self.transition_reason = "survey_timer_elapsed"
            return self.state

        if self.state == LandingState.EVALUATE and has_valid_zone and zone_score > c.zone_score_evaluate_commit:
            self.state = LandingState.APPROACH
            self.transition_reason = "zone_commit_high_confidence"
        elif self.state == LandingState.EVALUATE:
            self.state = LandingState.SEARCH
            self.transition_reason = "zone_unavailable_or_low_confidence"
        elif self.state == LandingState.SEARCH and has_valid_zone and zone_score > c.zone_score_search_commit:
            self.state = LandingState.APPROACH
            self.transition_reason = "zone_commit_recovered"
        elif self.state == LandingState.APPROACH and xy_error < c.approach_xy_tolerance_m and has_valid_zone:
            self.state = LandingState.ALIGN
            self.transition_reason = "horizontal_convergence"
        elif self.state == LandingState.ALIGN:
            self.align_stable_seconds = self.align_stable_seconds + dt if xy_error < c.align_xy_tolerance_m else 0.0
            if self.align_stable_seconds >= c.align_stable_s:
                self.state = LandingState.DESCEND
                self.transition_reason = "alignment_stable"
        elif self.state == LandingState.DESCEND:
            if intrusion_detected:
                self.state = LandingState.ABORT
                self.transition_reason = "intrusion_detected"
            elif xy_error > c.descend_xy_drift_m:
                self.state = LandingState.ALIGN
                self.transition_reason = "horizontal_drift_during_descent"
            elif altitude < c.touchdown_altitude_m:
                self.state = LandingState.LANDED
                self.transition_reason = "touchdown"
        elif self.state == LandingState.ABORT:
            self.abort_elapsed += dt
            if self.abort_elapsed > c.abort_resurvey_s:
                self.abort_elapsed = 0.0
                self.state = LandingState.SURVEY
                self.transition_reason = "abort_complete_resurvey"

        if self.target_lost_seconds > c.target_lost_timeout_s and self.state not in (LandingState.LANDED, LandingState.SURVEY):
            self.state = LandingState.SEARCH
            self.transition_reason = "target_lost_timeout"

        return self.state
