#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for LCME Memory.

Tests the full pipeline: ingestion -> neural scoring -> storage ->
retrieval (traditional + hippocampus) -> formatted output.

Run: python -m pytest tests/test_lcme.py -v
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure we import from the local package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def clean_singletons():
    """Reset all singletons before each test."""
    try:
        from lcme.core import reset_lcme
        reset_lcme()
    except Exception:
        pass
    yield
    try:
        from lcme.core import reset_lcme
        reset_lcme()
    except Exception:
        pass


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test data."""
    d = tempfile.mkdtemp(prefix="lcme_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def lcme(tmp_dir):
    """Create a LCME instance with temporary storage."""
    from lcme import LCME, LCMEConfig
    config = LCMEConfig(
        data_dir=tmp_dir,
        auto_maintenance=False,
        auto_consolidation=False,
    )
    t = LCME(config)
    return t


# =========================================================================
# 1. Basic Ingest/Retrieve Cycle
# =========================================================================

class TestBasicCycle:
    def test_ingest_returns_success(self, lcme):
        result = lcme.ingest("Alice likes Python programming.", origin="user")
        assert result["success"] is True
        assert result["event_id"]
        assert isinstance(result["claims"], int)

    def test_ingest_empty_text(self, lcme):
        result = lcme.ingest("", origin="user")
        assert result["success"] is True
        assert result["event_id"] == ""

    def test_retrieve_returns_list(self, lcme):
        lcme.ingest("Bob works at a startup in Berlin.", origin="user")
        results = lcme.retrieve("Bob Berlin")
        assert isinstance(results, list)

    def test_ingest_then_retrieve(self, lcme):
        lcme.ingest("Alice works at a startup in Berlin.", origin="user")
        results = lcme.retrieve("Alice Berlin")
        assert len(results) > 0
        # At least one result should mention Alice or Berlin
        texts = [r.get("content", "") + r.get("label", "") for r in results]
        combined = " ".join(texts).lower()
        assert "alice" in combined or "berlin" in combined


# =========================================================================
# 2. FTS5 Search
# =========================================================================

class TestFTS:
    def test_fts_search_works(self, lcme):
        lcme.ingest("Quantum computing uses qubits for calculations.", origin="user")
        results = lcme.sqlite.search_fts("quantum qubits")
        assert len(results) > 0

    def test_fts_no_results_for_unrelated(self, lcme):
        lcme.ingest("The weather is sunny today.", origin="user")
        results = lcme.sqlite.search_fts("xyznonexistent")
        assert len(results) == 0


# =========================================================================
# 3. Vector Search
# =========================================================================

class TestVectorSearch:
    def test_vector_search_returns_results(self, lcme):
        lcme.ingest("Python is a programming language.", origin="user")
        results = lcme.vectors.search("programming language")
        assert len(results) > 0

    def test_vector_similarity_ordering(self, lcme):
        lcme.ingest("Dogs are loyal pets that love walks.", origin="user")
        lcme.ingest("Quantum physics studies subatomic particles.", origin="user")
        results = lcme.vectors.search("pets and animals", limit=5)
        assert len(results) > 0
        # First result should have higher similarity
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]


# =========================================================================
# 4. Knowledge Graph Creation
# =========================================================================

class TestKnowledgeGraph:
    def test_graph_relation_creation(self, lcme):
        ok = lcme.graph.add_relation("Alice", "works_at", "Startup", confidence=0.9, origin="user")
        assert ok is True
        related = lcme.graph.get_related("Alice")
        assert len(related) > 0
        assert related[0]["relation"] == "works_at"

    def test_graph_traversal(self, lcme):
        lcme.graph.add_relation("A", "knows", "B", confidence=0.8, origin="user")
        lcme.graph.add_relation("B", "knows", "C", confidence=0.7, origin="user")
        results = lcme.graph.traverse("A", hops=2)
        node_ids = [r["node"]["id"] for r in results if "node" in r]
        assert "A" in node_ids
        assert "B" in node_ids


# =========================================================================
# 5. Claims Extraction
# =========================================================================

class TestClaimsExtraction:
    def test_claims_extracted_from_text(self, lcme):
        from lcme.ingestion import ClaimExtractor
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("Alice works at Google. Bob lives in Paris.")
        assert len(claims) >= 1
        subjects = [c.subject for c in claims]
        predicates = [c.predicate for c in claims]
        assert any("Alice" in s for s in subjects) or any("Bob" in s for s in subjects)

    def test_entities_extracted(self, lcme):
        from lcme.ingestion import ClaimExtractor
        extractor = ClaimExtractor()
        entities = extractor.extract_entities("Alice visited Berlin last summer.")
        assert any("Alice" in e for e in entities)
        assert any("Berlin" in e for e in entities)


# =========================================================================
# 6. Neural Cortex Initialization
# =========================================================================

class TestNeuralCortex:
    def test_cortex_initializes(self, tmp_dir):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from lcme.neural_cortex import LCMECortex, reset_cortex
        reset_cortex()
        db_path = Path(tmp_dir) / "lcme.db"
        model_path = Path(tmp_dir) / "models" / "cortex.pt"
        # Create minimal DB for cortex
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY)")
        conn.close()

        cortex = LCMECortex(db_path=db_path, model_path=model_path)
        assert cortex.param_count() > 0
        stats = cortex.get_stats()
        assert "total_params" in stats
        assert stats["total_params"] > 50000  # ~77K expected
        cortex.close()

    def test_cortex_score_importance(self, tmp_dir):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from lcme.neural_cortex import LCMECortex, reset_cortex
        reset_cortex()
        db_path = Path(tmp_dir) / "lcme.db"
        model_path = Path(tmp_dir) / "models" / "cortex.pt"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY)")
        conn.close()

        cortex = LCMECortex(db_path=db_path, model_path=model_path)
        score = cortex.score_importance({
            "origin": "user", "claim_count": 2, "entity_count": 1,
            "topic_count": 1, "text_length": 100, "has_user": True,
            "has_entity": True, "hour": 14,
        })
        assert 0.1 <= score <= 1.0
        cortex.close()

    def test_cortex_cold_start_blending(self, tmp_dir):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from lcme.neural_cortex import LCMECortex, reset_cortex
        reset_cortex()
        db_path = Path(tmp_dir) / "lcme.db"
        model_path = Path(tmp_dir) / "models" / "cortex.pt"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY)")
        conn.close()

        cortex = LCMECortex(db_path=db_path, model_path=model_path)
        # At cold start (0 steps), should return close to default
        blended = cortex._cold_blend(0.9, 0.5, steps=0, threshold=50)
        assert abs(blended - 0.5) < 0.01  # Should be very close to default
        cortex.close()


# =========================================================================
# 7. Hippocampus Initialization
# =========================================================================

class TestHippocampus:
    def test_hippocampus_initializes(self, tmp_dir):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from lcme.hippocampus import Hippocampus
        data_dir = Path(tmp_dir)
        lcme_db = data_dir / "lcme.db"
        # Create minimal lcme DB
        import sqlite3
        conn = sqlite3.connect(str(lcme_db))
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT, type TEXT, label TEXT, created_at TEXT, protected INT, metadata TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT)")
        conn.close()

        hippo = Hippocampus(data_dir=data_dir, lcme_db_path=lcme_db)
        stats = hippo.get_stats()
        assert stats["params"] > 200000  # ~226K expected
        assert stats["phase"] == "bootstrap"

    def test_hippocampus_not_ready_without_vectors(self, tmp_dir):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from lcme.hippocampus import Hippocampus
        data_dir = Path(tmp_dir)
        lcme_db = data_dir / "lcme.db"
        import sqlite3
        conn = sqlite3.connect(str(lcme_db))
        conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT, type TEXT, label TEXT, created_at TEXT, protected INT, metadata TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT)")
        conn.close()

        hippo = Hippocampus(data_dir=data_dir, lcme_db_path=lcme_db)
        assert hippo.is_ready() is False


# =========================================================================
# 8. Maintenance Cycle
# =========================================================================

class TestMaintenance:
    def test_maintenance_runs(self, lcme):
        stats = lcme.maintenance.run_maintenance(force=True)
        assert stats.duration_ms >= 0

    def test_unprotect_expired_nodes(self, lcme):
        from lcme.storage import Node
        from datetime import datetime, timedelta
        # Create a node with expired protection
        past = (datetime.now() - timedelta(hours=48)).isoformat()
        lcme.sqlite.add_node(Node(
            id="test_expire", type="memory", label="test",
            created_at=past, protected=True,
            metadata={"unprotect_after": past}
        ))
        lcme.maintenance._unprotect_expired_nodes()
        node = lcme.sqlite.get_node("test_expire")
        assert node.protected is False


# =========================================================================
# 9. Consolidation Cycle
# =========================================================================

class TestConsolidation:
    def test_consolidation_runs(self, lcme):
        # Ingest some data first
        lcme.ingest("Test memory for consolidation.", origin="user")
        stats = lcme.consolidate()
        assert "elapsed_s" in stats
        assert stats["consolidation_count"] == 1

    def test_consolidation_engine_stats(self, lcme):
        stats = lcme._consolidation.get_stats()
        assert "running" in stats
        assert stats["running"] is False


# =========================================================================
# 10. Memory Persistence
# =========================================================================

class TestPersistence:
    def test_vectors_save_and_load(self, tmp_dir):
        from lcme.storage import VectorStore
        data_dir = Path(tmp_dir)
        vs1 = VectorStore(data_dir, "all-MiniLM-L6-v2")
        vs1.add("node1", "hello world")
        vs1.save()

        # Create a new store instance and verify it loads
        vs2 = VectorStore(data_dir, "all-MiniLM-L6-v2")
        assert len(vs2._vectors) > 0
        assert "node1" in vs2._vectors

    def test_sqlite_persistence(self, tmp_dir):
        from lcme.storage import SQLiteStore, Node
        db_path = Path(tmp_dir) / "test.db"
        store1 = SQLiteStore(db_path)
        store1.add_node(Node(
            id="persist_test", type="test", label="persistent",
            created_at="2025-01-01T00:00:00", protected=False
        ))

        # Re-open
        store2 = SQLiteStore(db_path)
        node = store2.get_node("persist_test")
        assert node is not None
        assert node.label == "persistent"


# =========================================================================
# 11. Thread Safety
# =========================================================================

class TestThreadSafety:
    def test_concurrent_ingest(self, lcme):
        errors = []

        def ingest_worker(i):
            try:
                lcme.ingest(f"Thread {i} memory item about topic {i}.", origin="user")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=ingest_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        stats = lcme.get_stats()
        assert stats["nodes"] > 0

    def test_concurrent_ingest_retrieve(self, lcme):
        # Pre-populate
        lcme.ingest("Background memory about testing.", origin="user")
        errors = []

        def worker(i):
            try:
                if i % 2 == 0:
                    lcme.ingest(f"Concurrent item {i}", origin="user")
                else:
                    lcme.retrieve("testing")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0


# =========================================================================
# 12. Full End-to-End Pipeline
# =========================================================================

class TestEndToEnd:
    def test_full_pipeline(self, lcme):
        # Ingest
        r1 = lcme.ingest("Alice is a software engineer at Google.", origin="user")
        assert r1["success"]
        r2 = lcme.ingest("Alice lives in San Francisco.", origin="user")
        assert r2["success"]
        r3 = lcme.ingest("Bob is Alice's colleague.", origin="user")
        assert r3["success"]

        # Retrieve
        results = lcme.retrieve("Tell me about Alice")
        assert len(results) > 0

        # Context string
        ctx = lcme.get_context_string("What do we know about Alice?")
        assert len(ctx) > 0

        # Stats
        stats = lcme.get_stats()
        assert stats["nodes"] > 0
        assert stats["events"] >= 3

    def test_health_check(self, lcme):
        health = lcme.health_check()
        assert health["healthy"] is True
        assert health["sqlite"] == "ok"

    def test_protect_and_forget(self, lcme):
        lcme.ingest("Important memory to protect.", origin="user")
        stats = lcme.get_stats()
        # We know there's at least one node
        conn = lcme.sqlite._get_conn()
        rows = conn.execute("SELECT id FROM nodes WHERE type = 'memory' LIMIT 1").fetchall()
        if rows:
            node_id = rows[0][0]
            # Protect
            lcme.protect(node_id)
            node = lcme.sqlite.get_node(node_id)
            assert node.protected is True

            # Try to forget (should fail because protected)
            result = lcme.forget(node_id)
            assert result is False

            # Unprotect and forget
            lcme.unprotect(node_id)
            result = lcme.forget(node_id)
            assert result is True

    def test_context_string_format(self, lcme):
        lcme.ingest("The sky is blue on clear days.", origin="user")
        ctx = lcme.get_context_string("sky color")
        assert isinstance(ctx, str)
        # Should contain certainty markers or memory tags
        if ctx:
            assert any(marker in ctx.lower() for marker in
                       ["certain", "likely", "possible", "uncertain", "faint", "memory"])


# =========================================================================
# Run with pytest
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
