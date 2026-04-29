from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    i_limit: float = 2.0


class PID:
    def __init__(self, gains: PIDGains) -> None:
        self.g = gains
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return self.g.kp * error
        self.integral += error * dt
        self.integral = max(-self.g.i_limit, min(self.g.i_limit, self.integral))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.g.kp * error + self.g.ki * self.integral + self.g.kd * derivative
