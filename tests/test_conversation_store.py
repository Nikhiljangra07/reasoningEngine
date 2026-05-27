"""
ConversationStore tests.

No LLM calls, no API. Verifies the structured-storage spine:
    1. Session lifecycle (start, end, list)
    2. Iteration recording (sequence numbers, parent links)
    3. Decision attachment to iterations
    4. Turning points (creation, session linkage)
    5. Decision lineage links (with type validation)
    6. Pinning + unpinning (opt out of TTL)
    7. TTL behavior — expired entries hidden from reads, swept on demand
    8. Project scoping — two projects NEVER blend
    9. JSON persistence — round-trip integrity
    10. Tree + lineage views — render-ready output shapes

Run: PYTHONPATH=. python3 tests/test_conversation_store.py
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from src.bridge.conversation_store import ConversationStore, ExpiryAlert
from src.bridge.types import (
    DecisionLink,
    Iteration,
    Session,
    TurningPoint,
)


PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global PASSED, FAILED
    name = getattr(fn, "_test_name", fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# 1. Session lifecycle
# ---------------------------------------------------------------------------

@test("1.1 start_session creates a Session with default 30-day expiry")
async def test_start_session():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session(title="auth refactor discussion")
    assert sess.project_id == "proj-1"
    assert sess.title == "auth refactor discussion"
    assert sess.status == "active"
    assert sess.iteration_count == 0
    # expires_at should be ~30 days from now
    import time
    assert sess.expires_at is not None
    assert sess.expires_at > time.time() + (29 * 86400)


@test("1.2 end_session sets ended_at and status='ended'")
async def test_end_session():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    ended = await cs.end_session(sess.id)
    assert ended is not None
    assert ended.status == "ended"
    assert ended.ended_at is not None


@test("1.3 list_sessions returns non-expired sessions, newest first")
async def test_list_sessions():
    cs = ConversationStore(project_id="proj-1")
    import time
    s1 = await cs.start_session(title="first")
    await asyncio.sleep(0.01)
    s2 = await cs.start_session(title="second")
    out = await cs.list_sessions()
    assert len(out) == 2
    # Newest first
    assert out[0].id == s2.id
    assert out[1].id == s1.id


@test("1.4 get_session(unknown) returns None")
async def test_get_session_unknown():
    cs = ConversationStore(project_id="proj-1")
    assert await cs.get_session("S-does-not-exist") is None


# ---------------------------------------------------------------------------
# 2. Iteration recording
# ---------------------------------------------------------------------------

@test("2.1 add_iteration sequence_num auto-increments per session")
async def test_iteration_seq_num():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    i1 = await cs.add_iteration(sess.id, "hi", "hello")
    i2 = await cs.add_iteration(sess.id, "what about X?", "X is...")
    i3 = await cs.add_iteration(sess.id, "and Y?", "Y is...")
    assert i1.sequence_num == 1
    assert i2.sequence_num == 2
    assert i3.sequence_num == 3
    # Session counter reflects this
    updated = await cs.get_session(sess.id)
    assert updated.iteration_count == 3


@test("2.2 iterations_for_session returns them in sequence order")
async def test_iterations_in_order():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    await cs.add_iteration(sess.id, "1", "1r")
    await cs.add_iteration(sess.id, "2", "2r")
    iters = await cs.iterations_for_session(sess.id)
    assert [it.sequence_num for it in iters] == [1, 2]


@test("2.3 add_iteration on unknown session raises KeyError")
async def test_add_iteration_unknown_session():
    cs = ConversationStore(project_id="proj-1")
    try:
        await cs.add_iteration("S-bad", "hi", "hello")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown session")


@test("2.4 parent_iteration_id captured on branching")
async def test_iteration_branching():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    i1 = await cs.add_iteration(sess.id, "a", "ar")
    i2 = await cs.add_iteration(
        sess.id, "branched from a", "br", parent_iteration_id=i1.id,
    )
    assert i2.parent_iteration_id == i1.id


# ---------------------------------------------------------------------------
# 3. Decision attachment
# ---------------------------------------------------------------------------

@test("3.1 attach_decision adds id to iteration + increments session counter")
async def test_attach_decision():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    it = await cs.add_iteration(sess.id, "q", "r")
    await cs.attach_decision(it.id, "D-014")
    fetched = await cs.get_iteration(it.id)
    assert "D-014" in fetched.decision_ids
    updated_sess = await cs.get_session(sess.id)
    assert updated_sess.decision_count == 1


@test("3.2 attach_decision is idempotent")
async def test_attach_decision_idempotent():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    it = await cs.add_iteration(sess.id, "q", "r")
    await cs.attach_decision(it.id, "D-014")
    await cs.attach_decision(it.id, "D-014")  # no-op
    fetched = await cs.get_iteration(it.id)
    assert fetched.decision_ids.count("D-014") == 1


# ---------------------------------------------------------------------------
# 4. Turning points
# ---------------------------------------------------------------------------

@test("4.1 record_turning_point links to iteration + session")
async def test_record_tp():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    it = await cs.add_iteration(sess.id, "q", "r")
    tp = await cs.record_turning_point(
        it.id, title="pivot to JWT",
        description="user pivoted from session cookies",
        triggered_by=["D-001"],
        led_to=["D-014"],
    )
    assert tp.session_id == sess.id
    assert tp.iteration_id == it.id
    assert tp.triggered_by_decisions == ["D-001"]
    assert tp.led_to_decisions == ["D-014"]
    # Iteration knows about it
    fetched = await cs.get_iteration(it.id)
    assert tp.id in fetched.turning_point_ids
    # Session counter
    sess_fetched = await cs.get_session(sess.id)
    assert sess_fetched.turning_point_count == 1


@test("4.2 turning_points_for_session returns ordered by created_at")
async def test_tps_for_session():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    it = await cs.add_iteration(sess.id, "q", "r")
    tp1 = await cs.record_turning_point(it.id, "first")
    await asyncio.sleep(0.01)
    tp2 = await cs.record_turning_point(it.id, "second")
    tps = await cs.turning_points_for_session(sess.id)
    assert [t.id for t in tps] == [tp1.id, tp2.id]


# ---------------------------------------------------------------------------
# 5. Decision links (lineage graph)
# ---------------------------------------------------------------------------

@test("5.1 link_decisions creates a directed edge")
async def test_link_decisions():
    cs = ConversationStore(project_id="proj-1")
    link = await cs.link_decisions(
        from_decision_id="D-007",
        to_decision_id="D-014",
        link_type="leads_to",
        rationale="completing D-007 made D-014 necessary",
    )
    assert link.from_decision_id == "D-007"
    assert link.to_decision_id == "D-014"
    assert link.link_type == "leads_to"


@test("5.2 link_decisions rejects self-loops")
async def test_link_self_loop():
    cs = ConversationStore(project_id="proj-1")
    try:
        await cs.link_decisions("D-1", "D-1", "leads_to")
    except ValueError:
        return
    raise AssertionError("expected ValueError on self-loop")


@test("5.3 link_decisions rejects invalid link_type")
async def test_link_invalid_type():
    cs = ConversationStore(project_id="proj-1")
    try:
        await cs.link_decisions("D-1", "D-2", "vibes")
    except ValueError as e:
        assert "link_type" in str(e)
        return
    raise AssertionError("expected ValueError on invalid link_type")


@test("5.4 decisions_linked_from + decisions_linked_to")
async def test_lineage_queries():
    cs = ConversationStore(project_id="proj-1")
    await cs.link_decisions("D-1", "D-2", "leads_to")
    await cs.link_decisions("D-1", "D-3", "informed_by")
    await cs.link_decisions("D-4", "D-1", "depends_on")

    out_from_1 = await cs.decisions_linked_from("D-1")
    assert len(out_from_1) == 2
    targets = {l.to_decision_id for l in out_from_1}
    assert targets == {"D-2", "D-3"}

    in_to_1 = await cs.decisions_linked_to("D-1")
    assert len(in_to_1) == 1
    assert in_to_1[0].from_decision_id == "D-4"


# ---------------------------------------------------------------------------
# 6. Pinning + unpinning
# ---------------------------------------------------------------------------

@test("6.1 pin sets expires_at=None")
async def test_pin():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    assert sess.expires_at is not None
    ok = await cs.pin("sessions", sess.id)
    assert ok is True
    pinned = await cs.get_session(sess.id)
    assert pinned.expires_at is None


@test("6.2 unpin reassigns default TTL")
async def test_unpin():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    await cs.pin("sessions", sess.id)
    await cs.unpin("sessions", sess.id)
    unpinned = await cs.get_session(sess.id)
    assert unpinned.expires_at is not None


@test("6.3 pin returns False for unknown id")
async def test_pin_unknown():
    cs = ConversationStore(project_id="proj-1")
    ok = await cs.pin("sessions", "S-nope")
    assert ok is False


@test("6.4 pin rejects unknown entity_type")
async def test_pin_unknown_type():
    cs = ConversationStore(project_id="proj-1")
    try:
        await cs.pin("nonsense", "x")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# 7. TTL — filter on read + explicit sweep
# ---------------------------------------------------------------------------

@test("7.1 expired session is hidden from get_session / list_sessions")
async def test_expired_hidden():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    # Hand-edit expires_at into the past
    sess.expires_at = 1.0  # epoch 1970
    assert await cs.get_session(sess.id) is None
    assert await cs.list_sessions() == []


@test("7.2 include_expired=True bypasses the filter")
async def test_include_expired():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    sess.expires_at = 1.0
    found = await cs.get_session(sess.id, include_expired=True)
    assert found is not None
    listed = await cs.list_sessions(include_expired=True)
    assert len(listed) == 1


@test("7.3 sweep_expired removes expired entries and reports counts")
async def test_sweep():
    cs = ConversationStore(project_id="proj-1")
    s_live = await cs.start_session()
    s_dead = await cs.start_session()
    s_dead.expires_at = 1.0  # expired

    it_dead = await cs.add_iteration(s_live.id, "q", "r")
    it_dead.expires_at = 1.0  # expired iteration in a live session

    counts = await cs.sweep_expired()
    # 1 session removed, 1 iteration removed
    assert counts["sessions"] == 1
    assert counts["iterations"] == 1
    # The live session survived
    assert await cs.get_session(s_live.id) is not None
    # The dead session is gone (even from include_expired)
    assert await cs.get_session(s_dead.id, include_expired=True) is None


@test("7.4 pinned entries (expires_at=None) survive sweep")
async def test_pinned_survives_sweep():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session(expires_at=None)  # pinned at creation
    counts = await cs.sweep_expired(now=99999999999.0)  # far future
    assert counts["sessions"] == 0
    assert await cs.get_session(sess.id) is not None


# ---------------------------------------------------------------------------
# 8. Project scoping — two projects NEVER blend
# ---------------------------------------------------------------------------

@test("8.1 sessions in project A invisible to queries in project B")
async def test_project_isolation():
    shared: dict = {}
    cs_a = ConversationStore(project_id="proj-A", store=shared)
    cs_b = ConversationStore(project_id="proj-B", store=shared)
    sess_a = await cs_a.start_session(title="auth in A")
    assert await cs_b.get_session(sess_a.id) is None
    assert (await cs_b.list_sessions()) == []
    # A still sees its own
    assert await cs_a.get_session(sess_a.id) is not None


@test("8.2 sweep_expired walks every project, not just the adapter's own")
async def test_sweep_walks_all_projects():
    shared: dict = {}
    cs_a = ConversationStore(project_id="proj-A", store=shared)
    cs_b = ConversationStore(project_id="proj-B", store=shared)
    s_a = await cs_a.start_session()
    s_b = await cs_b.start_session()
    s_a.expires_at = 1.0
    s_b.expires_at = 1.0
    counts = await cs_a.sweep_expired()
    assert counts["sessions"] == 2  # both projects swept


# ---------------------------------------------------------------------------
# 9. JSON persistence
# ---------------------------------------------------------------------------

@test("9.1 round-trip: store + reload preserves everything")
async def test_persistence_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "conv.json")
        cs1 = ConversationStore(project_id="proj-1", storage_path=path)
        sess = await cs1.start_session(title="t")
        it = await cs1.add_iteration(sess.id, "u", "e", route="deep", effort="high")
        await cs1.attach_decision(it.id, "D-014")
        tp = await cs1.record_turning_point(
            it.id, "pivot", "desc",
            triggered_by=["D-007"], led_to=["D-014"],
        )
        await cs1.link_decisions("D-007", "D-014", "leads_to", "because X")

        # Fresh store, same path
        cs2 = ConversationStore(project_id="proj-1", storage_path=path)
        loaded_sess = await cs2.get_session(sess.id)
        assert loaded_sess is not None
        assert loaded_sess.title == "t"

        loaded_it = await cs2.get_iteration(it.id)
        assert loaded_it is not None
        assert loaded_it.route == "deep"
        assert "D-014" in loaded_it.decision_ids

        loaded_tp = await cs2.get_turning_point(tp.id)
        assert loaded_tp is not None
        assert loaded_tp.triggered_by_decisions == ["D-007"]

        links = await cs2.decisions_linked_from("D-007")
        assert len(links) == 1
        assert links[0].to_decision_id == "D-014"


@test("9.2 corrupted file → silent fresh start")
async def test_persistence_corrupted():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "conv.json")
        with open(path, "w") as f:
            f.write("not json }}}")
        cs = ConversationStore(project_id="proj-1", storage_path=path)
        # No exception, store is empty
        assert (await cs.list_sessions()) == []


@test("9.3 atomic write: no .tmp file remains after save")
async def test_persistence_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "conv.json")
        cs = ConversationStore(project_id="proj-1", storage_path=path)
        await cs.start_session()
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")


@test("9.4 schema drift on one entry → that entry skipped, rest load")
async def test_persistence_schema_drift():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "conv.json")
        # Hand-write a JSON file with one good session + one corrupted
        data = {
            "schema_version": 1,
            "ttl_seconds": 2592000,
            "projects": {
                "proj-1": {
                    "sessions": {
                        "S-good": {
                            "id": "S-good", "project_id": "proj-1",
                            "title": "good", "started_at": 1.0,
                            "ended_at": None, "iteration_count": 0,
                            "decision_count": 0, "turning_point_count": 0,
                            "status": "active", "expires_at": None,
                        },
                        "S-bad": {
                            "id": "S-bad", "project_id": "proj-1",
                            "title": "future schema",
                            "started_at": 1.0, "ended_at": None,
                            "iteration_count": 0, "decision_count": 0,
                            "turning_point_count": 0, "status": "active",
                            "expires_at": None,
                            "unknown_future_field": "ignore me",
                        },
                    },
                    "iterations": {}, "turning_points": {}, "decision_links": {},
                },
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)
        cs = ConversationStore(project_id="proj-1", storage_path=path)
        assert await cs.get_session("S-good") is not None
        # Bad entry skipped, not crashed
        assert await cs.get_session("S-bad", include_expired=True) is None


# ---------------------------------------------------------------------------
# 10. Tree + lineage views — the "fancy" presentation layer
# ---------------------------------------------------------------------------

@test("10.1 get_session_tree returns nested iteration+turning-point structure")
async def test_session_tree():
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session(title="auth")
    it1 = await cs.add_iteration(sess.id, "q1", "r1")
    it2 = await cs.add_iteration(sess.id, "q2", "r2")
    await cs.attach_decision(it1.id, "D-007")
    await cs.attach_decision(it2.id, "D-014")
    tp = await cs.record_turning_point(
        it2.id, "pivot to JWT",
        triggered_by=["D-007"], led_to=["D-014"],
    )
    await cs.link_decisions("D-007", "D-014", "leads_to", "after pivot")

    tree = await cs.get_session_tree(sess.id)
    assert tree["session"].id == sess.id
    assert len(tree["iterations"]) == 2
    # Iteration 2's block carries the turning point
    it2_block = next(b for b in tree["iterations"]
                     if b["iteration"].id == it2.id)
    assert len(it2_block["turning_points"]) == 1
    assert it2_block["turning_points"][0].id == tp.id
    # Decision links over the session's decisions surfaced
    assert len(tree["decision_links"]) == 1
    assert tree["decision_links"][0].from_decision_id == "D-007"


@test("10.2 get_session_tree on unknown session returns empty shape")
async def test_session_tree_unknown():
    cs = ConversationStore(project_id="proj-1")
    tree = await cs.get_session_tree("S-nope")
    assert tree["session"] is None
    assert tree["iterations"] == []
    assert tree["decision_links"] == []


@test("10.3 get_decision_lineage walks both directions up to max_depth")
async def test_lineage_walk():
    cs = ConversationStore(project_id="proj-1")
    # Build chain: A → B → C → D, plus E informs B
    await cs.link_decisions("A", "B", "leads_to")
    await cs.link_decisions("B", "C", "leads_to")
    await cs.link_decisions("C", "D", "leads_to")
    await cs.link_decisions("E", "B", "informed_by")

    out = await cs.get_decision_lineage("B", max_depth=2)
    assert out["root"] == "B"
    # Outgoing from B (depth=2): B→C (depth 1), C→D (depth 2)
    assert len(out["outgoing"]) == 2
    # Incoming to B (depth=2): A→B, E→B (both at depth 1)
    incoming_sources = {hop["link"].from_decision_id for hop in out["incoming"]}
    assert incoming_sources == {"A", "E"}


@test("10.4 lineage depth cap is honored")
async def test_lineage_depth_cap():
    cs = ConversationStore(project_id="proj-1")
    # A → B → C → D, depth=1 only sees A→B
    await cs.link_decisions("A", "B", "leads_to")
    await cs.link_decisions("B", "C", "leads_to")
    await cs.link_decisions("C", "D", "leads_to")
    out = await cs.get_decision_lineage("A", max_depth=1)
    assert len(out["outgoing"]) == 1
    assert out["outgoing"][0]["link"].to_decision_id == "B"


# ---------------------------------------------------------------------------
# 11. Custom TTL
# ---------------------------------------------------------------------------

@test("11.1 custom ttl_seconds applies to all created entities")
async def test_custom_ttl():
    cs = ConversationStore(project_id="proj-1", ttl_seconds=60)  # 1 minute
    sess = await cs.start_session()
    import time
    # Expires within ~60 seconds
    assert sess.expires_at is not None
    assert sess.expires_at <= time.time() + 60.5


@test("11.2 ttl_seconds=0 → entities created pinned (expires_at None)")
async def test_ttl_zero():
    cs = ConversationStore(project_id="proj-1", ttl_seconds=0)
    sess = await cs.start_session()
    assert sess.expires_at is None


# ---------------------------------------------------------------------------
# 12. ExpiryAlert system — 15/7/3-day + expired notifications
# ---------------------------------------------------------------------------

@test("12.1 entity > 15 days from expiry → no alert")
async def test_no_alert_far_out():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="far")
    s.expires_at = time.time() + 20 * 86400  # 20 days out
    alerts = await cs.get_expiry_alerts()
    assert alerts == []


@test("12.2 entity at 14 days remaining → 15_days tier")
async def test_alert_15_day_tier():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="14d")
    s.expires_at = time.time() + 14 * 86400
    alerts = await cs.get_expiry_alerts()
    assert len(alerts) == 1
    assert alerts[0].tier == "15_days"
    assert 13.5 < alerts[0].days_remaining < 14.5


@test("12.3 entity at 5 days remaining → 7_days tier")
async def test_alert_7_day_tier():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="5d")
    s.expires_at = time.time() + 5 * 86400
    alerts = await cs.get_expiry_alerts()
    assert len(alerts) == 1
    assert alerts[0].tier == "7_days"


@test("12.4 entity at 2 days remaining → 3_days tier")
async def test_alert_3_day_tier():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="2d")
    s.expires_at = time.time() + 2 * 86400
    alerts = await cs.get_expiry_alerts()
    assert alerts[0].tier == "3_days"


@test("12.5 entity past expiry but not yet swept → expired tier")
async def test_alert_expired_tier():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="dead")
    s.expires_at = time.time() - 86400  # 1 day past
    alerts = await cs.get_expiry_alerts()
    assert alerts[0].tier == "expired"
    assert alerts[0].days_remaining < 0


@test("12.6 pinned entities never produce alerts")
async def test_pinned_no_alert():
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="pinned", expires_at=None)
    alerts = await cs.get_expiry_alerts()
    assert alerts == []


@test("12.7 alerts sorted ascending by days_remaining (most urgent first)")
async def test_alerts_sorted():
    import time
    cs = ConversationStore(project_id="proj-1")
    now = time.time()
    s_far = await cs.start_session(title="14d")
    s_far.expires_at = now + 14 * 86400
    s_near = await cs.start_session(title="2d")
    s_near.expires_at = now + 2 * 86400
    s_mid = await cs.start_session(title="5d")
    s_mid.expires_at = now + 5 * 86400
    alerts = await cs.get_expiry_alerts()
    assert [a.title for a in alerts] == ["2d", "5d", "14d"]


@test("12.8 alerts surface iterations too, with sequence_num + snippet")
async def test_alert_iteration_title():
    import time
    cs = ConversationStore(project_id="proj-1")
    sess = await cs.start_session()
    it = await cs.add_iteration(sess.id, "should I refactor auth?", "...")
    it.expires_at = time.time() + 2 * 86400  # 2 days
    alerts = await cs.get_expiry_alerts()
    # Find the iteration alert (sess is at default 30 days, no alert)
    iter_alerts = [a for a in alerts if a.entity_type == "iterations"]
    assert len(iter_alerts) == 1
    assert "Turn 1" in iter_alerts[0].title
    assert "refactor" in iter_alerts[0].title.lower()


@test("12.9 alert user_options include pin and let_expire actions")
async def test_alert_user_options():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="x")
    s.expires_at = time.time() + 2 * 86400
    alerts = await cs.get_expiry_alerts()
    options = alerts[0].user_options
    option_ids = {o["id"] for o in options}
    assert "pin" in option_ids
    assert "let_expire" in option_ids


@test("12.10 project_only=True scopes to this adapter's project_id")
async def test_alerts_project_scoped():
    import time
    shared: dict = {}
    cs_a = ConversationStore(project_id="proj-A", store=shared)
    cs_b = ConversationStore(project_id="proj-B", store=shared)
    s_a = await cs_a.start_session(title="A")
    s_b = await cs_b.start_session(title="B")
    now = time.time()
    s_a.expires_at = now + 2 * 86400  # both expiring
    s_b.expires_at = now + 2 * 86400
    a_alerts = await cs_a.get_expiry_alerts()
    assert len(a_alerts) == 1
    assert a_alerts[0].title == "A"
    b_alerts = await cs_b.get_expiry_alerts()
    assert b_alerts[0].title == "B"


@test("12.11 project_only=False walks every project (admin view)")
async def test_alerts_global():
    import time
    shared: dict = {}
    cs_a = ConversationStore(project_id="proj-A", store=shared)
    cs_b = ConversationStore(project_id="proj-B", store=shared)
    now = time.time()
    s_a = await cs_a.start_session(title="A")
    s_a.expires_at = now + 2 * 86400
    s_b = await cs_b.start_session(title="B")
    s_b.expires_at = now + 5 * 86400
    all_alerts = await cs_a.get_expiry_alerts(project_only=False)
    assert len(all_alerts) == 2
    assert {a.title for a in all_alerts} == {"A", "B"}


@test("12.12 pin via store.pin() removes entity from alerts on next read")
async def test_pin_clears_alert():
    import time
    cs = ConversationStore(project_id="proj-1")
    s = await cs.start_session(title="rescue me")
    s.expires_at = time.time() + 2 * 86400  # would alert
    alerts_before = await cs.get_expiry_alerts()
    assert len(alerts_before) == 1
    await cs.pin("sessions", s.id)
    alerts_after = await cs.get_expiry_alerts()
    assert alerts_after == []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_start_session,
    test_end_session,
    test_list_sessions,
    test_get_session_unknown,
    test_iteration_seq_num,
    test_iterations_in_order,
    test_add_iteration_unknown_session,
    test_iteration_branching,
    test_attach_decision,
    test_attach_decision_idempotent,
    test_record_tp,
    test_tps_for_session,
    test_link_decisions,
    test_link_self_loop,
    test_link_invalid_type,
    test_lineage_queries,
    test_pin,
    test_unpin,
    test_pin_unknown,
    test_pin_unknown_type,
    test_expired_hidden,
    test_include_expired,
    test_sweep,
    test_pinned_survives_sweep,
    test_project_isolation,
    test_sweep_walks_all_projects,
    test_persistence_round_trip,
    test_persistence_corrupted,
    test_persistence_atomic,
    test_persistence_schema_drift,
    test_session_tree,
    test_session_tree_unknown,
    test_lineage_walk,
    test_lineage_depth_cap,
    test_custom_ttl,
    test_ttl_zero,
    test_no_alert_far_out,
    test_alert_15_day_tier,
    test_alert_7_day_tier,
    test_alert_3_day_tier,
    test_alert_expired_tier,
    test_pinned_no_alert,
    test_alerts_sorted,
    test_alert_iteration_title,
    test_alert_user_options,
    test_alerts_project_scoped,
    test_alerts_global,
    test_pin_clears_alert,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} conversation store tests...")
    print()
    for fn in ALL_TESTS:
        run_test(fn)
    print()
    print(f"{PASSED} passed, {FAILED} failed")
    if ERRORS:
        print()
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
