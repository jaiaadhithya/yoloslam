from enum import Enum


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
    def __init__(self) -> None:
        self.state = LandingState.SURVEY
        self.align_stable_seconds = 0.0
        self.target_lost_seconds = 0.0
        self.survey_elapsed = 0.0
        self.abort_elapsed = 0.0

    def update(
        self,
        has_valid_zone: bool,
        zone_score: float,
        xy_error: float,
        altitude: float,
        dt: float,
        intrusion_detected: bool,
    ) -> LandingState:
        if not has_valid_zone:
            self.target_lost_seconds += dt
        else:
            self.target_lost_seconds = 0.0

        if self.state == LandingState.SURVEY:
            self.survey_elapsed += dt
            if self.survey_elapsed >= 6.0:
                self.state = LandingState.EVALUATE
            return self.state

        if self.state == LandingState.EVALUATE and has_valid_zone and zone_score > 5.0:
            self.state = LandingState.APPROACH
        elif self.state == LandingState.EVALUATE:
            self.state = LandingState.SEARCH
        elif self.state == LandingState.SEARCH and has_valid_zone and zone_score > 2.0:
            self.state = LandingState.APPROACH
        elif self.state == LandingState.APPROACH and xy_error < 1.0 and has_valid_zone:
            self.state = LandingState.ALIGN
        elif self.state == LandingState.ALIGN:
            self.align_stable_seconds = self.align_stable_seconds + dt if xy_error < 0.15 else 0.0
            if self.align_stable_seconds >= 2.0:
                self.state = LandingState.DESCEND
        elif self.state == LandingState.DESCEND:
            if intrusion_detected:
                self.state = LandingState.ABORT
            elif xy_error > 0.5:
                self.state = LandingState.ALIGN
            elif altitude < 0.2:
                self.state = LandingState.LANDED
        elif self.state == LandingState.ABORT:
            self.abort_elapsed += dt
            if self.abort_elapsed > 2.0:
                self.abort_elapsed = 0.0
                self.state = LandingState.SURVEY

        if self.target_lost_seconds > 3.0 and self.state not in (LandingState.LANDED, LandingState.SURVEY):
            self.state = LandingState.SEARCH

        return self.state
