"""SQLite results database for optimizer outputs.

Two tables:
  - evaluations: every individual (gate_type, R1, R2, R3, V+, V-, ...) result.
    This is the per-combo cache that makes reruns instant.
  - gate_designs: the final optimized picks (best per gate type per board config).
"""

import sqlite3
import os
from datetime import datetime

from model import GateType
from .optimize import GateDesign

DB_PATH = os.path.join(os.path.dirname(__file__), "results.db")

# Shared connection — avoids open/close overhead per query
_shared_conn = None


def _connect(db_path=None):
    global _shared_conn
    path = db_path or DB_PATH
    if db_path is not None:
        # Explicit path: always create a fresh connection
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        _ensure_tables(conn)
        return conn
    if _shared_conn is None:
        _shared_conn = sqlite3.connect(path)
        _shared_conn.row_factory = sqlite3.Row
        # WAL mode for better concurrent read/write
        _shared_conn.execute("PRAGMA journal_mode=WAL")
        _ensure_tables(_shared_conn)
    return _shared_conn


def _ensure_tables(conn):
    # Per-combo evaluation cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gate_type TEXT NOT NULL,
            r1 REAL NOT NULL,
            r2 REAL NOT NULL,
            r3 REAL NOT NULL,
            v_pos REAL NOT NULL,
            v_neg REAL NOT NULL,
            v_high_target REAL NOT NULL,
            v_low_target REAL NOT NULL,
            temp_c REAL NOT NULL,
            jfet_beta REAL NOT NULL,
            jfet_vto REAL NOT NULL,

            -- Results
            power_W REAL NOT NULL,
            max_error_V REAL NOT NULL,
            v_out_high REAL,
            v_out_low REAL,
            delay_s REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_lookup
        ON evaluations (gate_type, r1, r2, r3, v_pos, v_neg,
                        v_high_target, v_low_target, temp_c,
                        jfet_beta, jfet_vto)
    """)

    # Final optimized designs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gate_designs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            gate_type TEXT NOT NULL,
            r1 REAL NOT NULL,
            r2 REAL NOT NULL,
            r3 REAL NOT NULL,
            v_pos REAL NOT NULL,
            v_neg REAL NOT NULL,
            v_high REAL NOT NULL,
            v_low REAL NOT NULL,
            swing REAL NOT NULL,
            power_mW REAL NOT NULL,
            delay_ns REAL NOT NULL,
            max_error_mV REAL NOT NULL,
            converged INTEGER NOT NULL,
            v_high_target REAL NOT NULL,
            v_low_target REAL NOT NULL,
            f_target REAL NOT NULL,
            n_fanout INTEGER NOT NULL,
            temp_c REAL NOT NULL,
            jfet_beta REAL NOT NULL,
            jfet_vto REAL NOT NULL,
            jfet_lambda REAL NOT NULL,
            jfet_is REAL NOT NULL,
            jfet_n REAL NOT NULL,
            jfet_isr REAL NOT NULL,
            jfet_nr REAL NOT NULL,
            jfet_alpha REAL NOT NULL,
            jfet_vk REAL NOT NULL,
            jfet_rd REAL NOT NULL,
            jfet_rs REAL NOT NULL,
            jfet_betatce REAL NOT NULL,
            jfet_vtotc REAL NOT NULL,
            jfet_xti REAL NOT NULL,
            jfet_eg REAL NOT NULL,
            cgd0 REAL NOT NULL,
            cgs0 REAL NOT NULL,
            notes TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_design_lookup
        ON gate_designs (gate_type, v_pos, v_neg, f_target, n_fanout, temp_c,
                         v_high_target, v_low_target)
    """)
    conn.commit()


# --- Evaluation cache (per-combo) ---

_eval_batch = []
_BATCH_SIZE = 500


def cache_evaluation(gate_type, r1, r2, r3, board, power_W, max_error_V,
                     v_out_high, v_out_low, delay_s, db_path=None):
    """Cache one R1/R2/R3 evaluation result. Batches writes for performance."""
    global _eval_batch
    _eval_batch.append((
        gate_type.value, r1, r2, r3, board.v_pos, board.v_neg,
        board.v_high, board.v_low, board.temp_c,
        board.jfet.beta, board.jfet.vto,
        float(power_W), float(max_error_V),
        float(v_out_high) if v_out_high is not None else None,
        float(v_out_low) if v_out_low is not None else None,
        float(delay_s),
    ))
    if len(_eval_batch) >= _BATCH_SIZE:
        flush_eval_cache(db_path)


def flush_eval_cache(db_path=None):
    """Write all pending evaluation results to the database."""
    global _eval_batch
    if not _eval_batch:
        return
    conn = _connect(db_path)
    conn.executemany("""
        INSERT OR IGNORE INTO evaluations (
            gate_type, r1, r2, r3, v_pos, v_neg,
            v_high_target, v_low_target, temp_c,
            jfet_beta, jfet_vto,
            power_W, max_error_V, v_out_high, v_out_low, delay_s
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, _eval_batch)
    conn.commit()
    _eval_batch = []


def lookup_evaluation(gate_type, r1, r2, r3, board, db_path=None):
    """Look up a cached evaluation. Returns (power_W, max_error_V, v_out_high, v_out_low, delay_s) or None."""
    # Flush pending writes first so we can find recently cached results
    flush_eval_cache(db_path)
    conn = _connect(db_path)
    row = conn.execute("""
        SELECT power_W, max_error_V, v_out_high, v_out_low, delay_s
        FROM evaluations
        WHERE gate_type = ? AND r1 = ? AND r2 = ? AND r3 = ?
          AND v_pos = ? AND v_neg = ?
          AND v_high_target = ? AND v_low_target = ?
          AND temp_c = ?
          AND jfet_beta = ? AND jfet_vto = ?
        LIMIT 1
    """, (
        gate_type.value, r1, r2, r3, board.v_pos, board.v_neg,
        board.v_high, board.v_low, board.temp_c,
        board.jfet.beta, board.jfet.vto,
    )).fetchone()

    if row is None:
        return None
    return (row["power_W"], row["max_error_V"], row["v_out_high"],
            row["v_out_low"], row["delay_s"])


def evaluation_count(db_path=None):
    """How many evaluations are cached."""
    conn = _connect(db_path)
    row = conn.execute("SELECT COUNT(*) as cnt FROM evaluations").fetchone()
    # shared connection stays open
    return row["cnt"]


# --- Final designs ---

def save_design(design, board_config, notes=None, db_path=None):
    """Save a final GateDesign with full board/JFET metadata."""
    conn = _connect(db_path)
    j = board_config.jfet
    c = board_config.caps

    conn.execute("""
        INSERT INTO gate_designs (
            created_at, gate_type,
            r1, r2, r3, v_pos, v_neg,
            v_high, v_low, swing, power_mW, delay_ns, max_error_mV, converged,
            v_high_target, v_low_target, f_target, n_fanout, temp_c,
            jfet_beta, jfet_vto, jfet_lambda, jfet_is, jfet_n,
            jfet_isr, jfet_nr, jfet_alpha, jfet_vk, jfet_rd, jfet_rs,
            jfet_betatce, jfet_vtotc, jfet_xti, jfet_eg,
            cgd0, cgs0, notes
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?
        )
    """, (
        datetime.now().isoformat(), design.gate_type.value,
        design.r1, design.r2, design.r3, design.v_pos, design.v_neg,
        design.v_high, design.v_low, design.swing,
        design.power_mW, design.delay_ns, design.max_error_mV,
        int(design.converged),
        board_config.v_high, board_config.v_low,
        board_config.f_target, board_config.n_fanout, board_config.temp_c,
        j.beta, j.vto, j.lmbda, j.is_, j.n, j.isr, j.nr,
        j.alpha, j.vk, j.rd, j.rs,
        j.betatce, j.vtotc, j.xti, j.eg,
        c.cgd0, c.cgs0, notes,
    ))
    conn.commit()
    # shared connection stays open


def find_design(gate_type, v_pos, v_neg, board_config, db_path=None):
    """Look up the best cached final design."""
    conn = _connect(db_path)
    j = board_config.jfet

    row = conn.execute("""
        SELECT * FROM gate_designs
        WHERE gate_type = ?
          AND v_pos = ? AND v_neg = ?
          AND v_high_target = ? AND v_low_target = ?
          AND f_target = ? AND n_fanout = ?
          AND temp_c = ?
          AND jfet_beta = ? AND jfet_vto = ? AND jfet_lambda = ?
          AND jfet_is = ? AND jfet_n = ?
          AND converged = 1
        ORDER BY power_mW ASC
        LIMIT 1
    """, (
        gate_type.value, v_pos, v_neg,
        board_config.v_high, board_config.v_low,
        board_config.f_target, board_config.n_fanout,
        board_config.temp_c,
        j.beta, j.vto, j.lmbda, j.is_, j.n,
    )).fetchone()
    # shared connection stays open

    if row is None:
        return None

    return GateDesign(
        gate_type=GateType(row["gate_type"]),
        r1=row["r1"], r2=row["r2"], r3=row["r3"],
        v_pos=row["v_pos"], v_neg=row["v_neg"],
        v_high=row["v_high"], v_low=row["v_low"],
        swing=row["swing"], power_mW=row["power_mW"],
        delay_ns=row["delay_ns"], max_error_mV=row["max_error_mV"],
        converged=bool(row["converged"]),
    )


def summary(db_path=None):
    """Print a summary of stored data."""
    conn = _connect(db_path)

    n_evals = conn.execute("SELECT COUNT(*) as cnt FROM evaluations").fetchone()["cnt"]
    n_designs = conn.execute("SELECT COUNT(*) as cnt FROM gate_designs").fetchone()["cnt"]
    print(f"Database: {n_evals} cached evaluations, {n_designs} final designs")

    rows = conn.execute("""
        SELECT gate_type, v_pos, v_neg, v_high_target, v_low_target,
               f_target, n_fanout, temp_c,
               r1, r2, r3, v_high, v_low, power_mW, max_error_mV, converged
        FROM gate_designs
        ORDER BY gate_type, power_mW
    """).fetchall()
    # shared connection stays open

    if rows:
        print(f"\n{'Type':<7} {'Rail':>8} {'R1':>7} {'R2':>7} {'R3':>7} "
              f"{'V_H':>6} {'V_L':>6} {'Err':>6} {'P':>7}")
        print("-" * 72)
        for r in rows:
            print(f"{r['gate_type']:<7} "
                  f"{r['v_pos']:+.0f}/{r['v_neg']:.0f} "
                  f"{r['r1']/1e3:>6.1f}k {r['r2']/1e3:>6.1f}k {r['r3']/1e3:>6.1f}k "
                  f"{r['v_high']:>6.3f} {r['v_low']:>6.3f} "
                  f"{r['max_error_mV']:>5.1f}mV "
                  f"{r['power_mW']:>6.2f}mW")
