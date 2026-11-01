import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from src.traffic_control.connected import write_network
from src.traffic_control.singapore import (
    audit_counts, build_sumo_network, candidate_chains, clip_points,
    extract_count_member, inspect_count_zip, load_site, load_transfer_models, network_audit,
    osm_audit, prepare, run_geometry_smoke,
)
from src.traffic_control.ppo import PPOModel
from src.traffic_control.scenario import ROOT, executable


class SingaporeCaseTests(unittest.TestCase):
    def test_optional_layout_does_not_change_training_corridor_xml(self):
        # Frozen from a live v2 training episode; evaluated networks must use
        # the same default topology and coordinates as the trained policies.
        expected = {
            "connected.nod.xml": "61a868432a01d14124a669b37407c65e3f3384bc7babe9c2cc8c04fff3eb10e9",
            "connected.edg.xml": "584a6c95b533fed13977809a7e02ddc924c67c87cf4547df22401362088c2386",
            "connected.con.xml": "76498846ee254d3f948c93b1e5c7649ea378480c7d70f99dd1f66819cbd34413",
        }
        with tempfile.TemporaryDirectory() as folder:
            for path in write_network(Path(folder), 140, False, True):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 expected[path.name])

    def test_site_limits_window_and_disk_caps(self):
        with tempfile.TemporaryDirectory() as folder:
            site = Path(folder) / "site.json"
            site.write_text(json.dumps({"name": "pilot", "bbox_west_south_east_north":
                                        [103.774, 1.2938, 103.786, 1.303],
                                        "study_junctions": 3, "max_osm_bytes": 8_000_000,
                                        "max_lta_geojson_bytes": 22_000_000,
                                        "max_signal_link_metres": 450}), encoding="utf-8")
            self.assertEqual(load_site(site)["study_junctions"], 3)
            site.write_text(json.dumps({"name": "too large", "bbox_west_south_east_north":
                                        [103.6, 1.2, 104.1, 1.5],
                                        "study_junctions": 3, "max_osm_bytes": 8_000_000,
                                        "max_lta_geojson_bytes": 22_000_000,
                                        "max_signal_link_metres": 450}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_site(site)

    def test_osm_and_lta_clipping_are_conservative(self):
        with tempfile.TemporaryDirectory() as folder:
            osm = Path(folder) / "tiny.osm.xml"
            osm.write_text('''<osm><node id="1"><tag k="highway" v="traffic_signals"/></node>
                <node id="2"><tag k="highway" v="crossing"/></node>
                <way id="3"><tag k="highway" v="footway"/></way></osm>''', encoding="utf-8")
            self.assertEqual(osm_audit(osm), {"osm_signal_nodes": ["1"],
                                               "osm_crossing_features": 1,
                                               "osm_footway_features": 1})
        data = {"type": "FeatureCollection", "features": [
            {"geometry": {"type": "Point", "coordinates": [103.78, 1.30]}},
            {"geometry": {"type": "Point", "coordinates": [103.9, 1.30]}},
            {"geometry": {"type": "LineString", "coordinates": [[103.78, 1.30]]}},
        ]}
        self.assertEqual(len(clip_points(data, [103.77, 1.29, 103.79, 1.31])["features"]), 1)

    def test_signal_candidates_are_road_connected(self):
        net = ET.fromstring('''<net>
          <junction id="A" type="traffic_light" x="0" y="0"/>
          <junction id="B" type="traffic_light" x="170" y="0"/>
          <junction id="C" type="traffic_light" x="280" y="0"/>
          <junction id="X" type="priority" x="80" y="0"/>
          <edge id="AX" from="A" to="X"><lane length="80"/></edge>
          <edge id="XB" from="X" to="B"><lane length="90"/></edge>
          <edge id="BC" from="B" to="C"><lane length="110"/></edge>
          <edge id="walk" from="A" to="C"><lane allow="pedestrian" length="2"/></edge>
          <edge id=":A_c0" function="crossing"><lane allow="pedestrian" length="6"/></edge>
        </net>''')
        self.assertEqual(candidate_chains(net, 200), [
            {"signal_groups": [["A"], ["B"], ["C"]], "road_path_metres": 280.0,
             "site_centroids_m": [[0.0, 0.0], [170.0, 0.0], [280.0, 0.0]],
             "centroid_spacings_m": [170.0, 110.0]}])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "net.xml"
            path.write_text(ET.tostring(net, encoding="unicode"), encoding="utf-8")
            summary = network_audit(path, 200)
            self.assertEqual(summary["sumo_crossings"], 1)
            self.assertEqual(summary["sumo_signal_junctions"], 3)

    def test_count_intake_requires_provenance_and_valid_intervals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "counts.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["collection_date", "site_id", "start_s", "end_s",
                                 "mode", "vehicle_class", "movement_id", "count",
                                 "source", "source_reference"])
                writer.writerow(["2026-09-28", "J1", 0, 900, "vehicle", "car",
                                 "north:through", 21, "field_observation", "team-log-1"])
                writer.writerow(["2026-09-28", "J1", 0, 900, "pedestrian", "",
                                 "crossing:west", 8, "field_observation", "team-log-1"])
            result = audit_counts(path)
            self.assertEqual(result["totals"], {"vehicle": 21, "pedestrian": 8})
            self.assertTrue(result["observed_count_data"])
            content = path.read_text(encoding="utf-8").replace(",21,", ",-1,")
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_counts(path)

    def test_count_zip_intake_extracts_only_one_small_safe_member(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            archive = directory / "counts.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("junctions/J1.csv", "timestamp,count\n0,12\n")
                handle.writestr("../wrong.csv", "do not extract")
                handle.writestr("database.gdb/table.gdbtable", "binary")
            index = inspect_count_zip(archive, "J1")
            self.assertEqual(index["matching_count"], 1)
            self.assertEqual(index["extractable_tabular_count"], 1)
            saved = extract_count_member(archive, "junctions/J1.csv", directory / "out")
            self.assertEqual(saved.read_text(), "timestamp,count\n0,12\n")
            with self.assertRaises(FileExistsError):
                extract_count_member(archive, "junctions/J1.csv", directory / "out")
            with self.assertRaises(ValueError):
                extract_count_member(archive, "../wrong.csv", directory / "out")
            with self.assertRaises(ValueError):
                extract_count_member(archive, "database.gdb/table.gdbtable", directory / "out")

    def test_transfer_loader_selects_trained_replica_zero_policies(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            entries = []
            for method in ("ippo", "mappo"):
                for weight, objective in ((0, "vehicle"), (3, "pedestrian"),
                                          (10, "pedestrian")):
                    model = PPOModel(method, objective, ["C0", "C1", "C2"], {}, 1)
                    model.episodes_trained = 2
                    path = directory / f"{method}-w{weight}-r0.json"
                    model.save(path)
                    entries.append({"controller": f"{method}-w{weight}",
                                    "method": method, "objective": objective,
                                    "reward_weight": weight, "training_run": 0,
                                    "training_seed": 1101, "model_path": str(path)})
            (directory / "training_manifest.json").write_text(
                json.dumps({"version": 2, "models": entries}), encoding="utf-8")
            selected = load_transfer_models(directory)
            self.assertEqual([item["label"] for item in selected],
                             ["ippo-w0-r0", "ippo-w3-r0", "mappo-w0-r0", "mappo-w3-r0"])
            self.assertTrue(all(item["episodes_trained"] == 2 for item in selected))


try:
    SUMO_READY = bool(executable("sumo") and executable("netconvert")
                      and importlib.util.find_spec("traci")
                      and importlib.util.find_spec("sumolib"))
except RuntimeError:
    SUMO_READY = False


@unittest.skipUnless(SUMO_READY, "SUMO, TraCI and sumolib are required")
class SingaporeGeometrySmokeTests(unittest.TestCase):
    def test_snapshot_moves_both_modes_and_repeats(self):
        snapshot = ROOT / "scenario" / "singapore" / "nus-kent-ridge-2026-09-28.osm.xml.gz"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            net_file = output / "singapore.net.xml"
            build_sumo_network(snapshot, net_file)
            one = run_geometry_smoke(net_file, output, 38)
            two = run_geometry_smoke(net_file, output, 38)
            self.assertEqual(one, two)
            self.assertEqual(one["completed_vehicles"], 24)
            self.assertEqual(one["pedestrians_seen"], 12)
            self.assertGreater(one["pedestrians_entering_crossings"], 0)

    def test_derived_baselines_share_safe_demand_and_crossings(self):
        with tempfile.TemporaryDirectory() as folder:
            report = prepare(ROOT / "config" / "singapore.json", Path(folder),
                             derived_baselines=True)
            derived = report["derived_baselines"]
            self.assertEqual(derived["modeled_junction_spacings_m"], [161, 196])
            self.assertEqual(derived["candidate_site_centroids_m"], [
                [1415.52, 3757.59], [1301.21, 3871.48], [1175.2, 4021.76]])
            self.assertEqual(set(derived["controllers"]), {"fixed", "actuated", "pressure"})
            nodes = {node.get("id"): (float(node.get("x")), float(node.get("y")))
                     for node in ET.parse(Path(folder) / "derived" / "fixed" /
                                          "connected.nod.xml").getroot().findall("node")}
            for i, saved in enumerate(derived["candidate_site_centroids_m"]):
                self.assertEqual(nodes[f"C{i}"], tuple(saved))
                arm = tuple(nodes[f"N{i}"][j] - saved[j] for j in range(2))
                self.assertAlmostEqual(math.hypot(*arm), 110, places=5)
                self.assertAlmostEqual(nodes[f"S{i}"][0] - saved[0], -arm[0], places=5)
                self.assertAlmostEqual(nodes[f"S{i}"][1] - saved[1], -arm[1], places=5)
            for summary in derived["controllers"].values():
                self.assertEqual(summary["completed_vehicles"], 192)
                self.assertEqual(summary["protected_crossings"], 12)
                self.assertGreater(summary["crossing_entries"], 0)
                metrics = json.loads((Path(folder) / summary["metrics_file"]).read_text())
                self.assertEqual(len(metrics["signal_history"]), 3)
                self.assertEqual(sum(row["entered_count"] for row in
                                     metrics["pedestrians_by_crossing"].values()),
                                 summary["crossing_entries"])
                for history in metrics["signal_history"].values():
                    self.assertIn("PED", [stage for _, stage in history])
            self.assertTrue((Path(folder) / derived["phase_replays"]["fixed"]).is_file())


if __name__ == "__main__":
    unittest.main()
