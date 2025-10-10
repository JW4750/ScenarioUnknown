
# -*- coding: utf-8 -*-
"""
HighD scenario mining via tag-combination method + coverage metrics + danger detection.

Usage:
  python highd_scenario_mining.py --prefix 18

Outputs saved next to inputs:
  highd_tags.csv
  highd_scenarios.csv
  highd_scenario_counts.csv
  highd_unknown_danger_events.csv
  highd_unknown_danger_stats.csv
  highd_coverage_metrics.csv
  highd_report.html
"""
import argparse
import io, os, json, math, base64
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CFG = {
    "accel_window_s": 0.5,
    "lane_change_cooldown_frames": 10,
    "accel_threshold": 0.3,
    "decel_threshold": -0.3,
    "dv_approach_threshold": 1.0,
    "rear_target_window": 60.0,
    "merge_front_window": 70.0,
    "merge_rear_window": 60.0,
    "ttc_crit": 1.5,
    "thw_crit": 1.0,
    "dhw_crit": 5.0,
    "pet_crit": 1.0,
    "unknown_merge_window_frames": 5,
    "coverage_tag_n_values": [1,5],
    "fig_dpi": 110
}

def read_highd_files(prefix: str):
    tracks = pd.read_csv(f"/mnt/data/{prefix}_tracks.csv")
    tracks_meta = pd.read_csv(f"/mnt/data/{prefix}_tracksMeta.csv")
    rec_meta = pd.read_csv(f"/mnt/data/{prefix}_recordingMeta.csv")
    return tracks, tracks_meta, rec_meta

def prep_tracks(tracks: pd.DataFrame, rec_meta: pd.DataFrame) -> pd.DataFrame:
    tr = tracks.copy().sort_values(["id","frame"]).reset_index(drop=True)
    fr = float(rec_meta["frameRate"].iloc[0]) if "frameRate" in rec_meta.columns else 25.0
    tr["frameRate"] = fr
    tr["dt"] = 1.0/fr
    if "drivingDirection" not in tr.columns:
        tr["drivingDirection"] = 2
    sign = np.where(tr["drivingDirection"] == 1, -1.0, 1.0)
    tr["v_along"] = tr["xVelocity"] * sign
    tr["a_along"] = tr["xAcceleration"] * sign
    win = int(max(1, fr * CFG["accel_window_s"]))
    tr["a_smooth"] = tr.groupby("id")["a_along"].rolling(window=win, min_periods=1).mean().reset_index(level=0, drop=True)
    cond_acc = tr["a_smooth"] > CFG["accel_threshold"]
    cond_dec = tr["a_smooth"] < CFG["decel_threshold"]
    tr["ego_longitudinal"] = np.where(cond_acc,"accelerating", np.where(cond_dec,"decelerating","cruising"))
    tr["prev_laneId"] = tr.groupby("id")["laneId"].shift(1)
    tr["lane_change_event"] = (tr["laneId"] != tr["prev_laneId"]) & tr["prev_laneId"].notna()
    lane_delta = tr["laneId"] - tr["prev_laneId"]
    is_left = ((tr["drivingDirection"] == 2) & (lane_delta < 0)) | ((tr["drivingDirection"] == 1) & (lane_delta > 0))
    tr["lane_change_dir"] = np.where(tr["lane_change_event"], np.where(is_left,"left","right"), "")
    tr["ego_lateral"] = "following_lane"
    tr.loc[tr["lane_change_event"] & (tr["lane_change_dir"]=="left"), "ego_lateral"] = "changing_left"
    tr.loc[tr["lane_change_event"] & (tr["lane_change_dir"]=="right"), "ego_lateral"] = "changing_right"
    tr["leader_id"] = tr["precedingId"]
    tr["has_leader"] = tr["leader_id"] > 0
    leader_cols = ["frame","id","v_along","a_smooth","xVelocity","xAcceleration"]
    leaders = tr[leader_cols].rename(columns={"id":"leader_id","v_along":"leader_v_along","a_smooth":"leader_a_smooth","xVelocity":"leader_xVelocity","xAcceleration":"leader_xAcceleration"})
    tr = tr.merge(leaders, on=["frame","leader_id"], how="left")
    tr["leader_motion"] = np.where(tr["leader_a_smooth"] > CFG["accel_threshold"],"accelerating",
                           np.where(tr["leader_a_smooth"] < CFG["decel_threshold"],"decelerating",
                           np.where(tr["leader_id"]>0,"cruising","none")))
    tr["dv_to_leader"] = np.where(tr["leader_id"]>0, tr["v_along"] - tr["leader_v_along"], np.nan)
    tr["ego_class"] = tr["class"].astype(str) if "class" in tr.columns else "Car"
    for col in ["leftPrecedingId", "rightPrecedingId", "leftFollowingId", "rightFollowingId", "leftAlongsideId", "rightAlsongsideId"]:
        if col not in tr.columns:
            tr[col] = 0
    if "rightAlongsideId" not in tr.columns and "rightAlsongsideId" in tr.columns:
        tr["rightAlongsideId"] = tr["rightAlsongsideId"]
    return tr

def generate_tags(tr: pd.DataFrame) -> pd.DataFrame:
    df = tr.copy()
    tags_list = []
    is_car = df["ego_class"].str.lower().eq("car")
    is_truck = df["ego_class"].str.lower().eq("truck")
    same_lane_front = df["has_leader"]
    same_lane_rear = df["followingId"] > 0 if "followingId" in df.columns else False
    in_front_left = df["leftPrecedingId"] > 0
    in_front_right = df["rightPrecedingId"] > 0
    side_left = df["leftAlongsideId"] > 0
    side_right = df["rightAlongsideId"] > 0 if "rightAlongsideId" in df.columns else df["rightAlsongsideId"] > 0
    rear_left = df["leftFollowingId"] > 0
    rear_right = df["rightFollowingId"] > 0
    slower_than_ego = (df["dv_to_leader"] < -5.0)
    faster_than_ego = (df["dv_to_leader"] > 5.0)
    long_act = df["ego_longitudinal"]
    lat_act = df["ego_lateral"]
    for i in range(len(df)):
        row_tags = []
        if is_car.iloc[i]: row_tags.append("Car")
        elif is_truck.iloc[i]: row_tags.append("Truck")
        if same_lane_front.iloc[i]: row_tags.append("Same lane in front")
        if isinstance(same_lane_rear, pd.Series) and same_lane_rear.iloc[i]: row_tags.append("Same lane rear")
        if in_front_left.iloc[i]: row_tags.append("In front left lane")
        if in_front_right.iloc[i]: row_tags.append("In front right lane")
        if side_left.iloc[i]: row_tags.append("At side left lane")
        if side_right.iloc[i]: row_tags.append("At side right lane")
        if rear_left.iloc[i]: row_tags.append("Rear left lane")
        if rear_right.iloc[i]: row_tags.append("Rear right lane")
        if slower_than_ego.iloc[i]: row_tags.append("Leader faster (Δv<-5)")
        if faster_than_ego.iloc[i]: row_tags.append("Leader slower (Δv>5)")
        if long_act.iloc[i] == "cruising": row_tags.append("Ego cruising")
        elif long_act.iloc[i] == "accelerating": row_tags.append("Ego accelerating")
        elif long_act.iloc[i] == "decelerating": row_tags.append("Ego decelerating")
        if lat_act.iloc[i] == "following_lane": row_tags.append("Ego keeping lane")
        elif lat_act.iloc[i] == "changing_left": row_tags.append("Ego changing lane left")
        elif lat_act.iloc[i] == "changing_right": row_tags.append("Ego changing lane right")
        tags_list.append(row_tags)
    df["tags"] = tags_list
    return df

def mine_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fr = df["frameRate"].iloc[0]

    for ego_id, g in df.groupby("id", sort=False):
        g = g.sort_values("frame").reset_index(drop=True)
        keeps_lane = g["ego_lateral"].eq("following_lane")
        has_leader = g["has_leader"]
        leader_motion = g["leader_motion"]
        dv = g["dv_to_leader"]

        def collect_intervals(mask):
            intervals = []
            in_seg = False
            start = None
            for idx, val in enumerate(mask):
                if val and not in_seg:
                    in_seg = True; start = idx
                elif not val and in_seg:
                    in_seg = False; intervals.append((start, idx-1))
            if in_seg: intervals.append((start, len(mask)-1))
            return intervals

        # C1-C3
        for s,e in collect_intervals(keeps_lane & has_leader & (leader_motion=="cruising")):
            rows.append((ego_id, int(g.loc[s,"frame"]), int(g.loc[e,"frame"]), "C1_leading_cruising",
                         [int(g.loc[s,"leader_id"])] if g.loc[s,"leader_id"]>0 else [], {}))
        for s,e in collect_intervals(keeps_lane & has_leader & (leader_motion=="accelerating")):
            rows.append((ego_id, int(g.loc[s,"frame"]), int(g.loc[e,"frame"]), "C2_leading_accelerating",
                         [int(g.loc[s,"leader_id"])] if g.loc[s,"leader_id"]>0 else [], {}))
        for s,e in collect_intervals(keeps_lane & has_leader & (leader_motion=="decelerating")):
            rows.append((ego_id, int(g.loc[s,"frame"]), int(g.loc[e,"frame"]), "C3_leading_decelerating",
                         [int(g.loc[s,"leader_id"])] if g.loc[s,"leader_id"]>0 else [], {}))

        # C4 approaching slower
        for s,e in collect_intervals(keeps_lane & has_leader & (dv > CFG["dv_approach_threshold"])):
            rows.append((ego_id, int(g.loc[s,"frame"]), int(g.loc[e,"frame"]), "C4_approaching_slower",
                         [int(g.loc[s,"leader_id"])] if g.loc[s,"leader_id"]>0 else [], {}))

        # C5/C6 leader switches
        g["prev_leader"] = g["leader_id"].shift(1).fillna(0).astype(int)
        leader_switched = (g["leader_id"] != g["prev_leader"]) & (g["leader_id"] > 0)
        id_to_track = {aid: gg.set_index("frame") for aid, gg in df.groupby("id")}

        for idx in np.where(leader_switched.values)[0]:
            s_frame = int(g.loc[idx, "frame"])
            new_leader = int(g.loc[idx, "leader_id"])
            ego_lane = int(g.loc[idx, "laneId"])
            ok = False
            if new_leader in id_to_track:
                lg = id_to_track[new_leader]
                f_from = s_frame - int(fr*1.0)
                f_to = s_frame
                sub = lg.loc[lg.index.intersection(range(f_from, f_to+1))]
                if len(sub) >= 2:
                    if int(sub["laneId"].iloc[-1]) == ego_lane and int(sub["laneId"].iloc[0]) != ego_lane:
                        ok = True
            if ok:
                e_idx = min(idx + int(fr*1.0), len(g)-1)
                rows.append((ego_id, int(g.loc[idx, "frame"]), int(g.loc[e_idx, "frame"]),
                             "C5_cut_in", [new_leader], {}))

        leader_switched_or_noleader = (g["leader_id"] != g["prev_leader"])
        for idx in np.where(leader_switched_or_noleader.values)[0]:
            s_frame = int(g.loc[idx, "frame"])
            prev_leader = int(g.loc[idx, "prev_leader"])
            if prev_leader <= 0:
                continue
            ego_lane_prev = int(g.loc[max(idx-1,0), "laneId"])
            ok = False
            if prev_leader in id_to_track:
                lg = id_to_track[prev_leader]
                f_from = s_frame - int(fr*1.0)
                f_to = s_frame + int(fr*0.5)
                sub = lg.loc[lg.index.intersection(range(f_from, f_to+1))]
                if len(sub) >= 2:
                    was_in = (int(sub["laneId"].iloc[0]) == ego_lane_prev)
                    now_out = (int(sub["laneId"].iloc[-1]) != ego_lane_prev)
                    if was_in and now_out: ok = True
            if ok:
                e_idx = min(idx + int(fr*1.0), len(g)-1)
                rows.append((ego_id, int(g.loc[max(idx-1,0),"frame"]), int(g.loc[e_idx,"frame"]),
                             "C6_cut_out", [prev_leader], {}))

        # C7 lane change with vehicle behind in target
        change_idx = np.where(g["ego_lateral"].isin(["changing_left","changing_right"]).values)[0]
        cooldown = CFG["lane_change_cooldown_frames"]
        last_start = -10**9
        for idx in change_idx:
            if idx - last_start < cooldown: continue
            last_start = idx
            direction = g.loc[idx, "ego_lateral"]
            behind_id = 0
            if direction == "changing_left" and "leftFollowingId" in g.columns:
                behind_id = int(g.loc[idx,"leftFollowingId"])
            elif direction == "changing_right" and "rightFollowingId" in g.columns:
                behind_id = int(g.loc[idx,"rightFollowingId"])
            if behind_id > 0:
                e_idx = min(idx + int(fr*2.0), len(g)-1)
                rows.append(( ego_id, int(g.loc[idx,"frame"]), int(g.loc[e_idx,"frame"]),
                              "C7_ego_lane_change_with_rear_in_target", [behind_id], {} ))

        # C8 ego merges between two vehicles
        for idx in change_idx:
            direction = g.loc[idx, "ego_lateral"]
            if direction not in ["changing_left","changing_right"]: continue
            f0 = int(g.loc[idx,"frame"]); f1 = min(f0 + int(fr*2.0), int(g["frame"].iloc[-1]))
            sub = g[(g["frame"]>=f0) & (g["frame"]<=f1)]
            if len(sub)==0: continue
            if direction=="changing_left":
                pid_series = sub["leftPrecedingId"]; fid_series = sub["leftFollowingId"]
            else:
                pid_series = sub["rightPrecedingId"]; fid_series = sub["rightFollowingId"]
            pid = 0; fid = 0
            if pid_series.fillna(0).max() > 0:
                nz = pid_series.replace(0, np.nan).dropna()
                if len(nz): pid = int(nz.iloc[0])
            if fid_series.fillna(0).max() > 0:
                nz2 = fid_series.replace(0, np.nan).dropna()
                if len(nz2): fid = int(nz2.iloc[0])
            if pid>0 and fid>0:
                rows.append((ego_id, int(sub["frame"].iloc[0]), int(sub["frame"].iloc[-1]),
                             "C8_ego_merge_between_two_vehicles", [pid,fid], {}))

        # C9/C10 overtakes (approximate using relative x sign flip)
        ex = g[["frame","x"]].set_index("frame")["x"]
        for side_type, pid_col, fid_col in [("left","leftPrecedingId","leftFollowingId"), ("right","rightPrecedingId","rightFollowingId")]:
            cand_ids = pd.unique(pd.concat([g[pid_col], g[fid_col]]).fillna(0).astype(int))
            cand_ids = [cid for cid in cand_ids if cid>0]
            for cid in cand_ids:
                sub = g[keeps_lane][["frame","x"]]
                if len(sub) < 3: continue
                tg = df[df["id"]==cid][["frame","x","laneId"]].set_index("frame")
                frames_common = sub["frame"].values
                tg_sub = tg.loc[tg.index.intersection(frames_common)]
                if len(tg_sub) < 3: continue
                x_ego = sub.set_index("frame")["x"].loc[tg_sub.index]
                relx = x_ego.values - tg_sub["x"].values
                if (np.nanmin(relx) < -2.0) and (np.nanmax(relx) > 2.0):
                    s_idx = int(np.nanargmin(relx)); e_idx = int(np.nanargmax(relx))
                    start_frame = int(tg_sub.index.values[min(s_idx, e_idx)])
                    end_frame = int(tg_sub.index.values[max(s_idx, e_idx)])
                    rows.append((ego_id, start_frame, end_frame, "C9_ego_overtakes_adjacent", [int(cid)], {"side": side_type}))

        for side_type, pid_col, fid_col in [("left","leftPrecedingId","leftFollowingId"), ("right","rightPrecedingId","rightFollowingId")]:
            cand_ids = pd.unique(pd.concat([g[pid_col], g[fid_col]]).fillna(0).astype(int))
            cand_ids = [cid for cid in cand_ids if cid>0]
            for cid in cand_ids:
                sub = g[keeps_lane][["frame","x"]]
                if len(sub) < 3: continue
                tg = df[df["id"]==cid][["frame","x","laneId"]].set_index("frame")
                frames_common = sub["frame"].values
                tg_sub = tg.loc[tg.index.intersection(frames_common)]
                if len(tg_sub) < 3: continue
                x_ego = sub.set_index("frame")["x"].loc[tg_sub.index]
                relx = x_ego.values - tg_sub["x"].values
                if (np.nanmin(relx) < -2.0) and (np.nanmax(relx) > 2.0) and (relx[0] > 0 and relx[-1] < 0):
                    start_frame = int(tg_sub.index.values[0]); end_frame = int(tg_sub.index.values[-1])
                    rows.append((ego_id, start_frame, end_frame, "C10_adjacent_overtakes_ego", [int(cid)], {"side": side_type}))

    scen_df = pd.DataFrame(rows, columns=["ego_id","start_frame","end_frame","scenario_code","main_actors","extra"])
    if len(scen_df):
        scen_df = scen_df[(scen_df["end_frame"] - scen_df["start_frame"]) >= 1].reset_index(drop=True)
    return scen_df

def compute_pet_for_lane_change(df: pd.DataFrame, scen_df: pd.DataFrame) -> pd.DataFrame:
    pet_records = []
    per_id = {i: g.set_index("frame") for i, g in df.groupby("id")}
    for _, row in scen_df.iterrows():
        code = row["scenario_code"]
        if code not in ["C5_cut_in","C7_ego_lane_change_with_rear_in_target","C8_ego_merge_between_two_vehicles"]:
            continue
        ego = int(row["ego_id"]); f0 = int(row["start_frame"])
        if ego not in per_id: continue
        g = per_id[ego]
        if f0 not in g.index: continue
        x0 = g.loc[f0, "x"]
        follower_id = None
        if isinstance(row["main_actors"], list) and len(row["main_actors"])>0:
            follower_id = row["main_actors"][-1]
        if follower_id is None or follower_id == 0 or follower_id not in per_id:
            left_f = int(g.loc[f0, "leftFollowingId"]) if "leftFollowingId" in g.columns else 0
            right_f = int(g.loc[f0, "rightFollowingId"]) if "rightFollowingId" in g.columns else 0
            follower_id = left_f if left_f>0 else (right_f if right_f>0 else None)
        if follower_id is None or follower_id not in per_id: continue
        fg = per_id[follower_id]
        fut = fg.loc[fg.index >= f0]
        if len(fut)==0: continue
        window_end = f0 + int(df["frameRate"].iloc[0]*4.0)
        fut = fut.loc[fut.index <= window_end]
        if len(fut)==0: continue
        idx = int((fut["x"] - x0).abs().idxmin())
        t_pet = (idx - f0) / df["frameRate"].iloc[0]
        pet_records.append({"ego_id": ego, "start_frame": f0, "follower_id": follower_id, "PET": float(t_pet), "scenario_code": code})
    return pd.DataFrame(pet_records)

def mine_unknown_danger(df: pd.DataFrame, scen_df: pd.DataFrame, pet_df: pd.DataFrame, tracks_meta: pd.DataFrame):
    d = df.copy()
    for col in ["ttc","thw","dhw"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = d[col].replace(0, np.nan)
    d["risk_ttc"] = (d["ttc"] > 0) & (d["ttc"] < CFG["ttc_crit"]) & d["has_leader"]
    d["risk_thw"] = (d["thw"] > 0) & (d["thw"] < CFG["thw_crit"]) & d["has_leader"]
    d["risk_dhw"] = (d["dhw"] > 0) & (d["dhw"] < CFG["dhw_crit"]) & d["has_leader"]
    d["risk_pet"] = False
    if pet_df is not None and len(pet_df):
        for _, r in pet_df.iterrows():
            ego = int(r["ego_id"]); f0 = int(r["start_frame"]); pet = r["PET"]
            if pd.notna(pet) and pet < CFG["pet_crit"]:
                mask = (d["id"]==ego) & (d["frame"].between(f0-2, f0+2))
                d.loc[mask, "risk_pet"] = True
    d["risk_any"] = d[["risk_ttc","risk_thw","risk_dhw","risk_pet"]].any(axis=1)

    covered = pd.Series(False, index=d.index)
    if len(scen_df):
        for ego, gg in scen_df.groupby("ego_id"):
            mask_ego = d["id"]==ego
            for _, s in gg.iterrows():
                covered |= mask_ego & d["frame"].between(int(s["start_frame"]), int(s["end_frame"]))
    d["covered_by_known"] = covered
    d["unknown_danger"] = d["risk_any"] & (~d["covered_by_known"])

    events = []
    merge_win = CFG["unknown_merge_window_frames"]
    for ego, g in d[d["unknown_danger"]].groupby("id"):
        if g.empty: continue
        g = g.sort_values("frame")
        current_s = int(g.iloc[0]["frame"]); current_e = current_s
        for f in g["frame"].values[1:]:
            f = int(f)
            if f <= current_e + merge_win:
                current_e = f
            else:
                events.append((ego, current_s, current_e))
                current_s = f; current_e = f
        events.append((ego, current_s, current_e))

    events_df = pd.DataFrame(events, columns=["ego_id","start_frame","end_frame"])
    total_dist_m = float(tracks_meta["traveledDistance"].sum()) if "traveledDistance" in tracks_meta.columns else float('nan')
    n_unknown = int(len(events_df))
    km_per_unknown = (total_dist_m/1000.0) / n_unknown if (n_unknown>0 and not math.isnan(total_dist_m)) else float('inf')
    stats_df = pd.DataFrame([{
        "unknown_events": n_unknown,
        "total_distance_km": total_dist_m/1000.0,
        "km_per_unknown_event": km_per_unknown
    }])
    return events_df, stats_df

def compute_coverage_metrics(df: pd.DataFrame, scen_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    if len(scen_df):
        df_key = df.set_index(["id","frame"])
        tag_counts = {}
        C = sorted(scen_df["scenario_code"].unique())
        for _, row in scen_df.iterrows():
            ego = int(row["ego_id"]); s = int(row["start_frame"]); code = row["scenario_code"]
            try:
                tags = df_key.loc[(ego, s), "tags"]
            except KeyError:
                continue
            if not isinstance(tags, list):
                continue
            for t in tags:
                tag_counts.setdefault((t, code), 0)
                tag_counts[(t, code)] += 1
        L = sorted({t for (t, _) in tag_counts.keys()})
        for n in CFG["coverage_tag_n_values"]:
            if len(L)==0 or len(C)==0:
                cov = float('nan')
            else:
                tot=0
                for t in L:
                    for c in C:
                        tot += min(n, tag_counts.get((t, c), 0))
                cov = tot / (n * len(L) * len(C))
            results.append({"metric": f"CoverageTag(n={n})", "value": cov})
        # time-based
        covered = pd.Series(False, index=df.index)
        for ego, gg in scen_df.groupby("ego_id"):
            mask = (df["id"]==ego)
            for _, r in gg.iterrows():
                covered |= mask & df["frame"].between(int(r["start_frame"]), int(r["end_frame"]))
        results.append({"metric":"CoverageT(n=1)", "value": float(covered.mean())})
    else:
        results.append({"metric":"CoverageTag(n=1)", "value": float('nan')})
        results.append({"metric":"CoverageT(n=1)", "value": float('nan')})
    return pd.DataFrame(results)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CFG["fig_dpi"], bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64

def make_report(counts_df, coverage_df, unknown_df, out_path):
    fig1 = plt.figure()
    if len(counts_df):
        ax = fig1.gca()
        counts_df.plot(kind="bar", x="scenario_code", y="count", ax=ax, legend=False, rot=45)
        ax.set_xlabel("Scenario"); ax.set_ylabel("Count"); ax.set_title("Scenario counts")
    img1 = fig_to_base64(fig1)

    fig2 = plt.figure()
    if len(coverage_df):
        ax2 = fig2.gca()
        coverage_df.plot(kind="bar", x="metric", y="value", ax=ax2, legend=False, rot=45)
        ax2.set_ylim(0, 1); ax2.set_ylabel("Coverage"); ax2.set_title("Coverage Metrics")
    img2 = fig_to_base64(fig2)

    fig3 = plt.figure()
    if len(unknown_df):
        ax3 = fig3.gca()
        unknown_df["duration_frames"] = unknown_df["end_frame"] - unknown_df["start_frame"] + 1
        unknown_df["duration_frames"].plot(kind="hist", bins=20, ax=ax3)
        ax3.set_xlabel("Unknown danger event duration (frames)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Unknown danger event durations")
    img3 = fig_to_base64(fig3)

    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8" />
<title>HighD 高速场景挖掘报告</title>
<style>body{{font-family:"Segoe UI",Arial,"Microsoft YaHei",sans-serif;margin:24px}}</style>
</head><body>
<h1>HighD 高速场景挖掘与覆盖度报告</h1>
<ul>
  <li><a href="highd_tags.csv">标签（逐帧）CSV</a></li>
  <li><a href="highd_scenarios.csv">场景识别结果 CSV</a></li>
  <li><a href="highd_scenario_counts.csv">十类场景计数 CSV</a></li>
  <li><a href="highd_unknown_danger_events.csv">未知危险场景 CSV</a></li>
  <li><a href="highd_unknown_danger_stats.csv">未知危险场景统计 CSV</a></li>
  <li><a href="highd_coverage_metrics.csv">覆盖度指标 CSV</a></li>
</ul>
<h2>场景计数</h2><img src="data:image/png;base64,{img1}" />
<h2>覆盖度指标</h2><img src="data:image/png;base64,{img2}" />
<h2>未知危险场景</h2><img src="data:image/png;base64,{img3}" />
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def run(prefix: str = "18"):
    tracks, tracks_meta, rec_meta = read_highd_files(prefix)
    tr = prep_tracks(tracks, rec_meta)
    tr_tagged = generate_tags(tr)
    tr_out = tr_tagged.copy()
    tr_out["tags"] = tr_out["tags"].apply(lambda x: ";".join(x) if isinstance(x, list) else "")
    tr_out.to_csv("/mnt/data/highd_tags.csv", index=False)

    scen_df = mine_scenarios(tr_tagged)
    scen_df.to_csv("/mnt/data/highd_scenarios.csv", index=False)

    counts = scen_df["scenario_code"].value_counts().rename_axis("scenario_code").reset_index(name="count")
    counts.to_csv("/mnt/data/highd_scenario_counts.csv", index=False)

    pet_df = compute_pet_for_lane_change(tr_tagged, scen_df)
    unknown_df, unknown_stats = mine_unknown_danger(tr_tagged, scen_df, pet_df, tracks_meta)
    unknown_df.to_csv("/mnt/data/highd_unknown_danger_events.csv", index=False)
    unknown_stats.to_csv("/mnt/data/highd_unknown_danger_stats.csv", index=False)

    coverage_df = compute_coverage_metrics(tr_tagged, scen_df)
    coverage_df.to_csv("/mnt/data/highd_coverage_metrics.csv", index=False)

    make_report(counts, coverage_df, unknown_df, "/mnt/data/highd_report.html")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=str, default="18", help="HighD file prefix, e.g., 18")
    args = ap.parse_args()
    run(args.prefix)
