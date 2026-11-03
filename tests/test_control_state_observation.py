"""Regression for indistinguishable controller states with different futures."""
import unittest
from types import SimpleNamespace

from src.traffic_control.fairness.control_observation import controller_memory
from src.traffic_control.fairness.signals import SafeSignal


def signal():
    return SafeSignal(SimpleNamespace(config={
        "yellow_seconds": 3, "all_red_seconds": 2,
        "pedestrian_green_seconds": 12, "pedestrian_clearance_empty_seconds": 3,
        "vehicle_green_min_seconds": 12, "vehicle_green_actuated_max_seconds": 30},
        clearance=11))


class ControlMemoryTest(unittest.TestCase):
    def test_committed_target_distinguishes_otherwise_identical_transition(self):
        a, b = signal(), signal()
        a.stage = b.stage = "ALL_RED"
        a.since = b.since = 10
        a.target, b.target = "NS", "PED"
        self.assertEqual((a.stage, a.age(11)), (b.stage, b.age(11)))
        self.assertEqual(a.mask(11, (True, True, True)), b.mask(11, (True, True, True)))
        self.assertNotEqual(controller_memory(a, 11)[0], controller_memory(b, 11)[0])
        a.advance(12, False)
        b.advance(12, False)
        self.assertEqual((a.stage, b.stage), ("NS", "PED"))

    def test_clearance_memory_distinguishes_release_from_wait(self):
        a, b = signal(), signal()
        a.stage = b.stage = "PED_CLEARANCE"
        a.since = b.since = 0
        a.empty_since, b.empty_since = 8, 10
        self.assertNotEqual(controller_memory(a, 11)[0], controller_memory(b, 11)[0])
        a.advance(11, False)
        b.advance(11, False)
        self.assertEqual((a.stage, b.stage), ("READY", "PED_CLEARANCE"))

    def test_empty_timer_not_started_is_not_zero_age_timer(self):
        a, b = signal(), signal()
        a.stage = b.stage = "PED_CLEARANCE"
        b.empty_since = 11
        self.assertNotEqual(controller_memory(a, 11)[0], controller_memory(b, 11)[0])
        a.stage = "EW"
        a.empty_since = 1
        self.assertEqual(controller_memory(a, 11)[0][-2:], [0., 0.])


if __name__ == "__main__":
    unittest.main()
