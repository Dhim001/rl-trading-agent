from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from ui_dashboard.components.ui import render_empty_state


def _db_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return -1


@st.cache_data(ttl=10, show_spinner=False)
def _load_tables(db_path_str: str, mtime_ns: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = mtime_ns
    db_path = Path(db_path_str)
    with sqlite3.connect(db_path) as conn:
        runs_df = pd.read_sql_query("SELECT * FROM runs ORDER BY started_at DESC", conn)
        artifacts_df = pd.read_sql_query("SELECT * FROM artifacts ORDER BY created_at DESC", conn)
        lineage_df = pd.read_sql_query("SELECT * FROM lineage ORDER BY recorded_at DESC", conn)
    return runs_df, artifacts_df, lineage_df


def _download_csv_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    data = b"" if df.empty else df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/csv",
        disabled=not bool(data),
        key=key,
    )


def _safe_parse_json(value) -> dict | list | str:
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return text


def _safe_dot_label(text: str) -> str:
    return str(text).replace("\\", "/").replace("\"", "'")


def _preview_parquet_head(path: Path, n_rows: int = 100) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.to_pandas().head(n_rows)
    except Exception:
        return pd.read_parquet(path).head(n_rows)


def _render_run_details(
    project_root: Path,
    runs_df: pd.DataFrame,
    artifacts_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
) -> None:
    st.markdown("---")
    st.markdown("### Run Detail")
    if runs_df.empty:
        render_empty_state("No runs available for detail view.")
        return

    run_options = runs_df["run_id"].astype(str).tolist()
    selected_run_id = st.selectbox("Select run_id", options=run_options, index=0, key="md_detail_run_id")
    row = runs_df[runs_df["run_id"].astype(str) == str(selected_run_id)]
    if row.empty:
        render_empty_state("Selected run was not found.")
        return
    run_row = row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workflow", str(run_row.get("workflow", "-")))
    c2.metric("Status", str(run_row.get("status", "-")))
    c3.metric("Started", str(run_row.get("started_at", "-")))
    c4.metric("Ended", str(run_row.get("ended_at", "-")))

    workflow = str(run_row.get("workflow", "")).strip()
    run_folder = project_root / "artifacts" / workflow / str(selected_run_id)
    st.markdown("**Run Folder**")
    st.code(str(run_folder), language="text")
    folder_cols = st.columns([1, 2])
    if folder_cols[0].button("Reveal Run Folder", key="md_reveal_run_folder", disabled=not run_folder.exists()):
        try:
            os.startfile(str(run_folder))  # type: ignore[attr-defined]
            st.success("Opened run folder in File Explorer.")
        except Exception as exc:
            st.warning(f"Could not open folder automatically: {exc}")
    folder_cols[1].caption(
        f"PowerShell: `ii \"{str(run_folder).replace('`', '')}\"`"
    )

    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown("**Params JSON**")
        st.json(_safe_parse_json(run_row.get("params_json")))
    with exp2:
        st.markdown("**Metrics JSON**")
        st.json(_safe_parse_json(run_row.get("metrics_json")))

    st.markdown("**Error**")
    error_text = str(run_row.get("error") or "").strip()
    if error_text:
        st.code(error_text, language="text")
    else:
        st.caption("No error recorded.")

    st.markdown("#### Linked Artifacts")
    run_artifacts = artifacts_df[artifacts_df["run_id"].astype(str) == str(selected_run_id)].copy() if not artifacts_df.empty else pd.DataFrame()
    if run_artifacts.empty:
        render_empty_state("No artifacts linked to this run.")
    else:
        st.dataframe(run_artifacts, width="stretch", hide_index=True)
        _download_csv_button(run_artifacts, "Download Run Artifacts CSV", f"{selected_run_id}_artifacts.csv", key="md_detail_artifacts_csv")

        st.markdown("**Artifact Preview**")
        artifact_kinds = sorted(run_artifacts["kind"].astype(str).unique().tolist())
        selected_kind = st.selectbox("Artifact kind", options=artifact_kinds, index=0, key="md_detail_artifact_kind")
        kind_subset = run_artifacts[run_artifacts["kind"].astype(str) == selected_kind].copy()
        artifact_options = kind_subset["path"].astype(str).tolist()
        selected_artifact = st.selectbox("Select artifact path", options=artifact_options, key="md_detail_artifact_path")
        artifact_path = project_root / selected_artifact
        st.code(str(artifact_path), language="text")
        if not artifact_path.exists():
            render_empty_state("Selected artifact path does not exist on disk.")
        else:
            suffix = artifact_path.suffix.lower()
            if suffix == ".json":
                try:
                    with open(artifact_path, encoding="utf-8") as f:
                        st.json(json.load(f))
                except Exception as exc:
                    st.code(f"Could not parse JSON: {exc}", language="text")
            elif suffix == ".csv":
                try:
                    st.dataframe(pd.read_csv(artifact_path).head(100), width="stretch", hide_index=True)
                except Exception as exc:
                    st.code(f"Could not load CSV: {exc}", language="text")
            elif suffix == ".parquet":
                try:
                    st.dataframe(_preview_parquet_head(artifact_path), width="stretch", hide_index=True)
                except Exception as exc:
                    st.code(f"Could not load Parquet: {exc}", language="text")
            else:
                st.caption("Preview supports JSON/CSV/Parquet. Use external viewer for other file types.")

    st.markdown("#### Linked Lineage Edges")
    run_lineage = lineage_df[lineage_df["run_id"].astype(str) == str(selected_run_id)].copy() if not lineage_df.empty else pd.DataFrame()
    if run_lineage.empty:
        render_empty_state("No lineage edges linked to this run.")
    else:
        st.dataframe(run_lineage, width="stretch", hide_index=True)
        _download_csv_button(run_lineage, "Download Run Lineage CSV", f"{selected_run_id}_lineage.csv", key="md_detail_lineage_csv")

    st.markdown("#### Run Lineage Graph")
    if run_lineage.empty:
        render_empty_state("No lineage graph available for this run.")
    else:
        graph_mode = st.radio(
            "Graph mode",
            options=["Compact", "Detailed"],
            index=0,
            horizontal=True,
            key="md_graph_mode",
        )
        run_label = f"Run\\n{selected_run_id}"
        lines = [
            "digraph G {",
            "rankdir=LR;",
            "node [shape=box, style=filled, fillcolor=\"#F6F8FA\"];",
            f"\"run::{_safe_dot_label(selected_run_id)}\" [label=\"{_safe_dot_label(run_label)}\", fillcolor=\"#E8F0FE\"];",
        ]
        run_node = f"run::{_safe_dot_label(selected_run_id)}"
        seen_nodes: set[str] = {run_node}

        for _, edge in run_lineage.iterrows():
            src = str(edge.get("input_path") or "").strip()
            dst = str(edge.get("output_path") or "").strip()
            relation = str(edge.get("relation") or "lineage")
            src_meta = run_artifacts[run_artifacts["path"].astype(str) == src].head(1) if not run_artifacts.empty else pd.DataFrame()
            dst_meta = run_artifacts[run_artifacts["path"].astype(str) == dst].head(1) if not run_artifacts.empty else pd.DataFrame()

            if src:
                src_id = f"src::{_safe_dot_label(src)}"
                if src_id not in seen_nodes:
                    seen_nodes.add(src_id)
                    if graph_mode == "Detailed" and not src_meta.empty:
                        row = src_meta.iloc[0]
                        kind = str(row.get("kind", "-"))
                        fmt = str(row.get("format", "-"))
                        size = str(row.get("size_bytes", "-"))
                        label = f"Input\\n{src}\\n[{kind} | {fmt} | {size} bytes]"
                    else:
                        label = f"Input\\n{src}"
                    lines.append(f"\"{src_id}\" [label=\"{_safe_dot_label(label)}\", fillcolor=\"#FFF7E6\"];")
                lines.append(f"\"{src_id}\" -> \"{run_node}\" [label=\"{_safe_dot_label(relation)}\"];")

            if dst:
                dst_id = f"dst::{_safe_dot_label(dst)}"
                if dst_id not in seen_nodes:
                    seen_nodes.add(dst_id)
                    if graph_mode == "Detailed" and not dst_meta.empty:
                        row = dst_meta.iloc[0]
                        kind = str(row.get("kind", "-"))
                        fmt = str(row.get("format", "-"))
                        size = str(row.get("size_bytes", "-"))
                        label = f"Output\\n{dst}\\n[{kind} | {fmt} | {size} bytes]"
                    else:
                        label = f"Output\\n{dst}"
                    lines.append(f"\"{dst_id}\" [label=\"{_safe_dot_label(label)}\", fillcolor=\"#E8F5E9\"];")
                lines.append(f"\"{run_node}\" -> \"{dst_id}\" [label=\"{_safe_dot_label(relation)}\"];")

        lines.append("}")
        st.graphviz_chart("\n".join(lines), width="stretch")


def render(project_root: Path) -> None:
    st.subheader("Metadata Registry")
    st.caption("Inspect run history, artifact catalog, and lineage records stored in `artifacts/metadata.db`.")

    db_path = project_root / "artifacts" / "metadata.db"
    if not db_path.exists():
        render_empty_state("No metadata database found yet.", "Run download/train/backtest/tune/paper workflows first.")
        return

    try:
        with st.spinner("Loading metadata registry..."):
            runs_df, artifacts_df, lineage_df = _load_tables(str(db_path), _db_mtime_ns(db_path))
    except Exception as exc:
        render_empty_state("Could not read metadata registry right now.", "Retry in a moment; the DB may be busy.")
        st.code(str(exc), language="text")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Runs", len(runs_df))
    m2.metric("Artifacts", len(artifacts_df))
    m3.metric("Lineage Links", len(lineage_df))

    st.markdown("### Runs")
    if runs_df.empty:
        render_empty_state("No run rows found in metadata DB.")
    else:
        filter_cols = st.columns(3)
        workflow_options = sorted([str(x) for x in runs_df["workflow"].dropna().unique().tolist()])
        status_options = sorted([str(x) for x in runs_df["status"].dropna().unique().tolist()])
        workflow_filter = filter_cols[0].multiselect("Workflow", options=workflow_options, default=workflow_options, key="md_workflow")
        status_filter = filter_cols[1].multiselect("Status", options=status_options, default=status_options, key="md_status")
        run_search = filter_cols[2].text_input("Search run_id", value="", key="md_run_search").strip().lower()

        filtered_runs = runs_df.copy()
        if workflow_filter:
            filtered_runs = filtered_runs[filtered_runs["workflow"].astype(str).isin(workflow_filter)]
        if status_filter:
            filtered_runs = filtered_runs[filtered_runs["status"].astype(str).isin(status_filter)]
        if run_search:
            filtered_runs = filtered_runs[filtered_runs["run_id"].astype(str).str.lower().str.contains(run_search, na=False)]
        st.dataframe(filtered_runs, width="stretch", hide_index=True)
        _download_csv_button(filtered_runs, "Download Runs CSV", "metadata_runs.csv", key="md_runs_csv")

    st.markdown("---")
    st.markdown("### Artifacts")
    if artifacts_df.empty:
        render_empty_state("No artifact rows found in metadata DB.")
    else:
        artifact_cols = st.columns(2)
        kind_options = sorted([str(x) for x in artifacts_df["kind"].dropna().unique().tolist()])
        kind_filter = artifact_cols[0].multiselect("Artifact kind", options=kind_options, default=kind_options, key="md_kind")
        path_search = artifact_cols[1].text_input("Search artifact path", value="", key="md_path_search").strip().lower()

        filtered_artifacts = artifacts_df.copy()
        if kind_filter:
            filtered_artifacts = filtered_artifacts[filtered_artifacts["kind"].astype(str).isin(kind_filter)]
        if path_search:
            filtered_artifacts = filtered_artifacts[
                filtered_artifacts["path"].astype(str).str.lower().str.contains(path_search, na=False)
            ]
        st.dataframe(filtered_artifacts, width="stretch", hide_index=True)
        _download_csv_button(filtered_artifacts, "Download Artifacts CSV", "metadata_artifacts.csv", key="md_artifacts_csv")

    st.markdown("---")
    st.markdown("### Lineage")
    if lineage_df.empty:
        render_empty_state("No lineage rows found in metadata DB.")
    else:
        rel_options = sorted([str(x) for x in lineage_df["relation"].dropna().unique().tolist()])
        rel_filter = st.multiselect("Relation", options=rel_options, default=rel_options, key="md_relation")
        filtered_lineage = lineage_df.copy()
        if rel_filter:
            filtered_lineage = filtered_lineage[filtered_lineage["relation"].astype(str).isin(rel_filter)]
        st.dataframe(filtered_lineage, width="stretch", hide_index=True)
        _download_csv_button(filtered_lineage, "Download Lineage CSV", "metadata_lineage.csv", key="md_lineage_csv")

    _render_run_details(project_root, runs_df, artifacts_df, lineage_df)


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)

