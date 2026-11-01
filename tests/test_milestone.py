import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.traffic_control.cli import run
from src.traffic_control.connected import incoming_groups
from src.traffic_control.metrics import CrossingTracker, vehicle_summary
from src.traffic_control.ppo import PPOModel, gae
from src.traffic_control.scenario import build_network, clearance_seconds, executable, make_demand
from src.traffic_control.signals import Controller, SignalPlan, classify_links, load_signal_conflicts
from src.traffic_control.viewer import write_replay


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config" / "milestone.json").read_text())
try:
    SUMO_READY = bool(executable("sumo") and executable("netconvert")
                      and importlib.util.find_spec("traci"))
except RuntimeError:
    SUMO_READY = False


class MilestoneUnitTests(unittest.TestCase):
    def test_hand_checked_crossing_events(self):
        tracker = CrossingTracker({":C_c0"})
        tracker.observe(1, {"p1": (":C_w0", ":C_c0", 1)})  # first stopped interval began at t=0
        tracker.observe(2, {"p1": (":C_w0", ":C_c0", 2),
                            "p2": (":C_c0", "", 0)})        # immediate entry: zero wait
        tracker.observe(5, {"p1": (":C_c0", "", 0)})        # p1 waited 5 seconds
        tracker.observe(7, {"p3": (":C_w0", ":C_c0", 1)})  # still unserved at t=10
        summary = tracker.summary(10)[":C_c0"]
        self.assertEqual(summary["entered_count"], 2)
        self.assertEqual(summary["mean_wait_seconds"], 2.5)
        self.assertEqual(summary["p95_wait_seconds"], 5)
        self.assertEqual(summary["unserved_count"], 1)
        self.assertEqual(summary["unserved_elapsed_seconds"], [4])

    def test_entrance_queue_and_entry_trace(self):
        tracker = CrossingTracker({":C_c0"})
        # A moving walker reaches the measured 1.2 m entrance zone at t=3.
        tracker.observe(2, {"walking": (":C_w0", ":C_c0", 0, 1.8)})
        tracker.observe(3, {"walking": (":C_w0", ":C_c0", 0, 1.1)})
        tracker.observe(4, {"walking": (":C_w0", ":C_c0", 1, 1.04)})
        tracker.observe(9, {"walking": (":C_c0", ":C_w1", 0, None)})
        # Another walker queues on the immediate approach, farther from the
        # entrance; the first stopped interval is [5, 6].
        tracker.observe(6, {"queued": (":C_w0", ":C_c0", 1, 2.5)})
        tracker.observe(7, {"queued": (":C_w0", ":C_c0", 2, 2.5)})
        tracker.observe(10, {"direct": (":C_c0", ":C_w1", 0, None)})
        events = {event.person_id: event for event in tracker.events.values()}
        self.assertEqual((events["walking"].reached_at, events["walking"].entered_at,
                          events["walking"].wait, events["walking"].arrival_source),
                         (3, 9, 6, "entrance"))
        self.assertEqual((events["queued"].reached_at, events["queued"].entered_at,
                          events["queued"].arrival_source), (5, None, "queued_approach"))
        self.assertEqual((events["direct"].wait, events["direct"].arrival_source),
                         (0, "entry_without_approach"))
        summary = tracker.summary(12)[":C_c0"]
        self.assertEqual(summary["entered_count"], 2)
        self.assertEqual(summary["unserved_count"], 1)
        self.assertEqual(summary["unserved_elapsed_seconds"], [7])
        self.assertEqual(summary["entry_without_approach_count"], 1)

    def test_demand_seed_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            one = make_demand(Path(folder) / "a.xml", 38, CONFIG)
            two = make_demand(Path(folder) / "b.xml", 38, CONFIG)
            other = make_demand(Path(folder) / "c.xml", 39, CONFIG)
        self.assertEqual(one["sha256"], two["sha256"])
        self.assertNotEqual(one["sha256"], other["sha256"])

    def test_unfinished_and_uninserted_vehicle_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            tripinfo = Path(folder) / "tripinfo.xml"
            tripinfo.write_text('<tripinfos><tripinfo id="v0" timeLoss="7.5"/></tripinfos>')
            summary = vehicle_summary(tripinfo, {"v0", "v1", "v2"}, {"v0", "v1"}, {"v0"})
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["unfinished_count"], 1)
        self.assertEqual(summary["uninserted_count"], 1)
        self.assertEqual(summary["completed_total_time_loss_seconds"], 7.5)

    def test_legal_phase_sequences(self):
        plan = SignalPlan({"EW": {0, 1}, "NS": {2, 3}, "PED": {4, 5}}, 6, 18, CONFIG)
        for kind in ("fixed", "actuated", "policy", "pressure"):
            control = Controller(plan, kind)
            observed = []
            for second in range(1, 230):
                state = control.tick(second, pending_pedestrian=True, action=1)
                plan.assert_safe(state)
                observed.append(control.stage)
            self.assertIn("PED", observed)
            self.assertIn("PED_CLEARANCE", observed)
            transitions = control.history
            for (start, stage), (end, following) in zip(transitions, transitions[1:]):
                duration = end - start
                if stage in ("EW", "NS"):
                    self.assertGreaterEqual(duration, CONFIG["vehicle_green_min_seconds"])
                    self.assertEqual(following, stage + "_YELLOW")
                if stage == "PED_CLEARANCE":
                    self.assertGreaterEqual(duration, plan.clearance)
                    self.assertEqual(following, "EW")
                if stage == "PED":
                    self.assertEqual(following, "PED_CLEARANCE")
                    self.assertEqual(duration, CONFIG["pedestrian_green_seconds"])
                if stage.endswith("_YELLOW"):
                    self.assertEqual(duration, CONFIG["yellow_seconds"])
                    self.assertEqual(following, "ALL_RED")
                if stage == "ALL_RED":
                    self.assertEqual(duration, CONFIG["all_red_seconds"])

    def test_conflict_matrix_catches_same_group_foes(self):
        plan = SignalPlan({"EW": {0, 1}, "NS": {2}, "PED": {3}}, 4, 18,
                          CONFIG, {(0, 1)})
        with self.assertRaisesRegex(AssertionError, "foes"):
            plan.state("EW")
        with self.assertRaisesRegex(AssertionError, "foes"):
            plan.assert_safe("yGrr")
        self.assertEqual(plan.state("NS"), "rrGr")

    def test_clearance_formula_covers_slowest_design_speed(self):
        crossing_lengths = {":C_c0": 6.4, ":C_c1": 8.0}
        expected = math.ceil(8.0 / CONFIG["minimum_walking_speed_mps"]
                             + CONFIG["pedestrian_clearance_buffer_seconds"])
        self.assertEqual(clearance_seconds(crossing_lengths, CONFIG), expected)

    def test_clearance_extends_while_crossing_occupied(self):
        plan = SignalPlan({"EW": {0}, "NS": {1}, "PED": {2}}, 3, 18, CONFIG)
        control = Controller(plan, "fixed")
        second = 0
        while control.stage != "PED_CLEARANCE":
            second += 1
            control.tick(second, True)
        for _ in range(plan.clearance + 5):
            second += 1
            control.tick(second, False, crossing_occupied=True)
            self.assertEqual(control.stage, "PED_CLEARANCE")
        for _ in range(CONFIG["pedestrian_clearance_empty_seconds"] - 1):
            second += 1
            control.tick(second, False, crossing_occupied=False)
            self.assertEqual(control.stage, "PED_CLEARANCE")
        # A renewed crossing entry resets the empty-step guard.
        second += 1
        control.tick(second, False, crossing_occupied=True)
        self.assertEqual(control.stage, "PED_CLEARANCE")
        for _ in range(CONFIG["pedestrian_clearance_empty_seconds"] - 1):
            second += 1
            control.tick(second, False, crossing_occupied=False)
            self.assertEqual(control.stage, "PED_CLEARANCE")
        control.tick(second + 1, False, crossing_occupied=False)
        self.assertEqual(control.stage, "EW")

    def test_central_critic_cannot_change_local_action(self):
        settings = json.loads((Path("config") / "training.json").read_text())
        model = PPOModel("mappo", "pedestrian", ["C0", "C1", "C2"], settings, 38)
        model.critics["C0"][9] = 2.0
        local = [0.0] * 9
        one = model.select("C0", local, [0.0] * 27, False, training=True)
        global_changed = [0.0] * 27
        global_changed[9] = 1.0
        two = model.select("C0", local, global_changed, False, training=True)
        self.assertEqual(one[0], two[0])
        self.assertNotEqual(one[2], two[2])
        self.assertEqual(model.select("C0", local, global_changed, True, training=False)[2], 0.0)
        self.assertEqual(model.select("C0", local, [], True, training=False)[0],
                         model.select("C0", local, global_changed, True, training=False)[0])
        self.assertEqual(gae([1, 1], [0, 0], 1, 1), [2, 1])

    def test_mappo_and_ippo_use_same_local_reward_scope(self):
        settings = json.loads((Path("config") / "training.json").read_text())
        model = PPOModel("mappo", "vehicle", ["C0", "C1", "C2"], settings, 38)
        global_obs = [1.0] + [0.0] * 26
        obs = [1.0] + [0.0] * 8
        traces = {tls: [{"obs": obs, "global_obs": global_obs,
                         "action": 0, "old_logp": 0.0, "value": 0.0,
                         "actor_mask": False, "reward": reward}]
                  for tls, reward in (("C0", 1.0), ("C1", 100.0), ("C2", 0.0))}
        model.update(traces)
        self.assertGreater(model.critics["C0"][0], 0)
        self.assertGreater(model.critics["C1"][0], 10 * model.critics["C0"][0])
        self.assertEqual(model.critics["C2"][0], 0)


@unittest.skipUnless(SUMO_READY, "SUMO binaries and TraCI are required for integration smoke test")
class SumoSmokeTest(unittest.TestCase):
    def test_generated_turns_have_no_green_foes(self):
        import traci

        with tempfile.TemporaryDirectory() as folder:
            net = Path(folder) / "connected.net.xml"
            crossings = build_network(net, 3)
            traci.start([executable("sumo"), "--net-file", str(net),
                         "--end", "1", "--no-step-log", "true"])
            try:
                for tls in ("C0", "C1", "C2"):
                    local_crossings = {name for name in crossings if name.startswith(f":{tls}_c")}
                    groups = classify_links(traci.trafficlight.getControlledLinks(tls),
                                            local_crossings, incoming_groups(3)[tls])
                    foes = load_signal_conflicts(net, tls)
                    self.assertGreater(len(foes), 0)
                    plan = SignalPlan(groups,
                                      len(traci.trafficlight.getRedYellowGreenState(tls)),
                                      clearance_seconds({c: crossings[c] for c in local_crossings}, CONFIG),
                                      CONFIG, foes)
                    for stage in ("EW", "NS", "PED", "EW_YELLOW", "NS_YELLOW",
                                  "ALL_RED", "PED_CLEARANCE"):
                        plan.assert_safe(plan.state(stage))
            finally:
                traci.close()

    def test_live_position_trace_matches_crossing_events(self):
        import traci

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            net = directory / "single.net.xml"
            routes = directory / "demand.rou.xml"
            crossings = build_network(net)
            make_demand(routes, 38, CONFIG)
            traci.start([executable("sumo"), "--net-file", str(net),
                         "--route-files", str(routes), "--end", "300",
                         "--seed", "38", "--step-length", "1",
                         "--pedestrian.striping.dawdling", "0",
                         "--no-step-log", "true"])
            tracker = CrossingTracker(set(crossings))
            approaches = {}
            entries = {}
            try:
                ends = {crossing: (traci.lane.getShape(crossing + "_0")[0],
                                   traci.lane.getShape(crossing + "_0")[-1])
                        for crossing in crossings}
                for _ in range(300):
                    traci.simulationStep()
                    now = traci.simulation.getTime()
                    observations = {}
                    for ident in traci.person.getIDList():
                        road = traci.person.getRoadID(ident)
                        next_edge = traci.person.getNextEdge(ident)
                        stopped = traci.person.getWaitingTime(ident)
                        distance = None
                        if next_edge in crossings:
                            position = traci.person.getPosition(ident)
                            distance = min(math.dist(position, endpoint)
                                           for endpoint in ends[next_edge])
                            approaches.setdefault((ident, next_edge), []).append(
                                (now, distance, stopped))
                        if road in crossings:
                            entries.setdefault((ident, road), now)
                        observations[ident] = (road, next_edge, stopped, distance)
                    tracker.observe(now, observations)
            finally:
                traci.close()
        self.assertGreater(len(entries), 0)
        sampled_at_entrance = 0
        positive_wait = 0
        for key, event in tracker.events.items():
            if key in entries:
                self.assertEqual(event.entered_at, entries[key])
                self.assertGreaterEqual(event.wait, 0)
                positive_wait += event.wait > 0
            if event.arrival_source == "entrance":
                first_near = next(time for time, distance, _ in approaches[key]
                                  if distance <= tracker.approach_distance_metres)
                self.assertEqual(event.reached_at, first_near)
                sampled_at_entrance += 1
        self.assertGreater(sampled_at_entrance, 0)
        self.assertGreater(positive_wait, 0)

    def test_both_baselines_move_and_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            fixed = run("fixed", 38, path / "fixed", Path("config/milestone.json"))
            actuated = run("actuated", 38, path / "actuated", Path("config/milestone.json"))
            repeated = run("fixed", 38, path / "repeat", Path("config/milestone.json"))
            self.assertEqual(fixed, repeated)
            self.assertEqual(fixed["demand_sha256"], actuated["demand_sha256"])
            for result in (fixed, actuated):
                self.assertGreater(result["vehicles"]["completed_count"], 0)
                self.assertGreater(result["persons_seen_at_crossing"], 0)
                self.assertGreater(sum(c["entered_count"] for c in result["pedestrians_by_crossing"].values()), 0)
                self.assertIn("PED", [stage for _, stage in result["signal_history"]])
                vehicle = result["vehicles"]
                self.assertEqual(vehicle["scheduled_count"], vehicle["completed_count"]
                                 + vehicle["unfinished_count"] + vehicle["uninserted_count"])

    def test_connected_junctions_and_slow_walkers(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run("actuated", 39, Path(folder) / "three", Path("config/milestone.json"),
                         junctions=3, slow_pedestrians=True)
            self.assertEqual(result["junctions"], 3)
            self.assertEqual(len(result["pedestrians_by_crossing"]), 12)
            self.assertEqual(len(result["signal_history"]), 3)
            self.assertGreater(result["vehicles"]["completed_count"], 0)
            self.assertGreater(sum(x["entered_count"] for x in
                                   result["pedestrians_by_crossing"].values()), 0)
            for history in result["signal_history"].values():
                self.assertIn("PED", [stage for _, stage in history])

    def test_gui_phase_replay_reproduces_completed_trip_time_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            original = run("fixed", 40, source, Path("config/milestone.json"), junctions=3)
            replay_dir = Path(folder) / "replay"
            _, additional = write_replay(source, replay_dir)
            tripinfo = replay_dir / "tripinfo.xml"
            subprocess.run([
                executable("sumo"), "--net-file", str(source / "connected.net.xml"),
                "--route-files", str(source / "demand.rou.xml"),
                "--additional-files", str(additional),
                "--begin", "0", "--end", "600", "--step-length", "1",
                "--seed", "40", "--pedestrian.striping.dawdling", "0",
                "--time-to-teleport", "-1", "--tripinfo-output", str(tripinfo),
                "--no-step-log", "true",
            ], check=True)
            losses = [float(item.attrib["timeLoss"])
                      for item in ET.parse(tripinfo).iter("tripinfo")]
            self.assertEqual(len(losses), original["vehicles"]["completed_count"])
            self.assertAlmostEqual(statistics.mean(losses),
                                   original["vehicles"]["completed_mean_time_loss_seconds"])
            original_trips = {item.attrib["id"]: item.attrib for item in
                              ET.parse(source / "tripinfo.xml").iter("tripinfo")}
            replay_trips = {item.attrib["id"]: item.attrib for item in
                            ET.parse(tripinfo).iter("tripinfo")}
            self.assertEqual(replay_trips, original_trips)


if __name__ == "__main__":
    unittest.main()
