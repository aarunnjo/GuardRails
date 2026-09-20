from fireguard.store import TrustStore


def test_new_source_has_no_incidents(memory_db):
    store = TrustStore(memory_db)
    assert store.incident_count("https://example.com/a") == 0


def test_record_incident_increments_count(memory_db):
    store = TrustStore(memory_db)
    store.record_incident("src-1", "L2 detect: score 0.99")
    store.record_incident("src-1", "L2 detect: score 0.95")
    assert store.incident_count("src-1") == 2


def test_incidents_are_per_source(memory_db):
    store = TrustStore(memory_db)
    store.record_incident("src-1", "reason")
    assert store.incident_count("src-2") == 0


def test_history_is_ordered_and_readable(memory_db):
    store = TrustStore(memory_db)
    store.record_incident("src-1", "first")
    store.record_incident("src-1", "second")
    history = store.history("src-1")
    assert [r for r, _ in history] == ["first", "second"]


def test_persists_across_reconnect(tmp_path):
    path = str(tmp_path / "trust.db")
    TrustStore(path).record_incident("src-1", "reason")
    assert TrustStore(path).incident_count("src-1") == 1
