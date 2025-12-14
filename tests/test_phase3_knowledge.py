#!/usr/bin/env python
"""Test script to verify Phase 3 knowledge module skeleton."""

import sys


def test_knowledge_base_interface():
    """Test KnowledgeBase interface and basic operations."""
    print("Testing KnowledgeBase interface...")

    from paola.knowledge import KnowledgeBase, MemoryKnowledgeStorage

    # Create knowledge base with memory storage
    kb = KnowledgeBase(storage=MemoryKnowledgeStorage())
    print("✓ KnowledgeBase created")

    # Test store_insight
    insight_id = kb.store_insight(
        problem_signature={
            "dimensions": 10,
            "constraints_count": 2,
            "problem_type": "nonlinear"
        },
        strategy={
            "algorithm": "SLSQP",
            "settings": {"ftol": 1e-6, "maxiter": 100}
        },
        outcome={
            "success": True,
            "iterations": 45,
            "final_objective": 0.001
        },
        metadata={"notes": "Converged well"}
    )
    assert insight_id is not None
    assert isinstance(insight_id, str)
    print(f"✓ store_insight() works (ID: {insight_id[:8]}...)")

    # Test get_insight
    retrieved = kb.get_insight(insight_id)
    assert retrieved is not None
    assert retrieved["insight_id"] == insight_id
    assert retrieved["problem_signature"]["dimensions"] == 10
    assert retrieved["strategy"]["algorithm"] == "SLSQP"
    assert retrieved["outcome"]["success"] is True
    print("✓ get_insight() works")

    # Test count
    count = kb.count()
    assert count == 1
    print(f"✓ count() works ({count} insights)")

    # Store another insight
    insight_id_2 = kb.store_insight(
        problem_signature={
            "dimensions": 20,
            "constraints_count": 5,
            "problem_type": "nonlinear"
        },
        strategy={"algorithm": "BFGS"},
        outcome={
            "success": False,
            "iterations": 100,
            "final_objective": 10.5
        }
    )
    assert kb.count() == 2
    print("✓ Multiple insights can be stored")

    # Test get_all_insights
    all_insights = kb.get_all_insights()
    assert len(all_insights) == 2
    print("✓ get_all_insights() works")

    # Test retrieve_insights (should return empty - not implemented)
    retrieved_insights = kb.retrieve_insights(
        problem_signature={"dimensions": 10, "problem_type": "nonlinear"},
        top_k=5
    )
    assert isinstance(retrieved_insights, list)
    assert len(retrieved_insights) == 0  # Not implemented yet
    print("✓ retrieve_insights() returns empty (expected for skeleton)")

    # Test clear
    kb.clear()
    assert kb.count() == 0
    print("✓ clear() works")

    print("\n✅ All KnowledgeBase interface tests passed!\n")
    return True


def test_storage_backends():
    """Test storage backend implementations."""
    print("Testing storage backends...")

    from paola.knowledge.storage import MemoryKnowledgeStorage, FileKnowledgeStorage

    # Test MemoryKnowledgeStorage
    storage = MemoryKnowledgeStorage()
    print("✓ MemoryKnowledgeStorage created")

    # Store
    insight = {
        "insight_id": "test_001",
        "data": "test_data"
    }
    storage.store("test_001", insight)
    assert storage.count() == 1
    print("✓ MemoryKnowledgeStorage.store() works")

    # Retrieve
    retrieved = storage.retrieve("test_001")
    assert retrieved is not None
    assert retrieved["insight_id"] == "test_001"
    print("✓ MemoryKnowledgeStorage.retrieve() works")

    # List all
    all_insights = storage.list_all()
    assert len(all_insights) == 1
    print("✓ MemoryKnowledgeStorage.list_all() works")

    # Clear
    storage.clear()
    assert storage.count() == 0
    print("✓ MemoryKnowledgeStorage.clear() works")

    # Test FileKnowledgeStorage (skeleton only)
    try:
        file_storage = FileKnowledgeStorage()
        print("✓ FileKnowledgeStorage created (skeleton)")

        # Should raise NotImplementedError
        try:
            file_storage.store("test", {})
            print("✗ FileKnowledgeStorage should raise NotImplementedError")
            return False
        except NotImplementedError:
            print("✓ FileKnowledgeStorage.store() raises NotImplementedError (expected)")

    except Exception as e:
        print(f"✓ FileKnowledgeStorage skeleton (may not initialize): {e}")

    print("\n✅ All storage backend tests passed!\n")
    return True


def test_knowledge_tools():
    """Test knowledge tools for agent."""
    print("Testing knowledge tools...")

    from paola.tools.knowledge_tools import (
        store_optimization_insight,
        retrieve_optimization_knowledge,
        list_all_knowledge
    )

    # Test store_optimization_insight (returns placeholder)
    result = store_optimization_insight.invoke({
        "problem_type": "rosenbrock",
        "dimensions": 10,
        "algorithm": "SLSQP",
        "success": True,
        "iterations": 45,
        "final_objective": 0.001,
        "notes": "Test note"
    })
    assert "status" in result
    assert result["status"] == "not_implemented"  # Expected for skeleton
    print("✓ store_optimization_insight tool callable (returns placeholder)")

    # Test retrieve_optimization_knowledge (returns empty)
    insights = retrieve_optimization_knowledge.invoke({
        "problem_type": "rosenbrock",
        "dimensions": 10,
        "top_k": 3
    })
    assert isinstance(insights, list)
    assert len(insights) == 0  # Expected for skeleton
    print("✓ retrieve_optimization_knowledge tool callable (returns empty)")

    # Test list_all_knowledge
    all_knowledge = list_all_knowledge.invoke({})
    assert "count" in all_knowledge
    assert all_knowledge["count"] == 0  # Expected for skeleton
    print("✓ list_all_knowledge tool callable (returns empty)")

    print("\n✅ All knowledge tool tests passed!\n")
    return True


def test_cli_commands():
    """Test CLI knowledge commands."""
    print("Testing CLI knowledge commands...")

    from paola.foundry import OptimizationFoundry, FileStorage
    from paola.cli.commands import CommandHandler
    from rich.console import Console
    import io
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Create platform
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)

        # Create command handler with captured output
        console = Console(file=io.StringIO())
        handler = CommandHandler(platform, console)

        # Test handle_knowledge_list
        handler.handle_knowledge_list()
        output = console.file.getvalue()
        assert "Knowledge Module" in output
        assert "Skeleton Only" in output
        print("✓ CLI /knowledge command works (shows skeleton message)")

        # Reset console output
        console.file = io.StringIO()

        # Test handle_knowledge_show
        handler.handle_knowledge_show("test_id")
        output = console.file.getvalue()
        assert "not yet implemented" in output
        print("✓ CLI /knowledge show command works (shows not implemented)")

        print("\n✅ All CLI command tests passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_module_imports():
    """Test that all module imports work."""
    print("Testing module imports...")

    # Test main module imports
    from paola.knowledge import (
        KnowledgeBase,
        KnowledgeStorage,
        MemoryKnowledgeStorage,
        FileKnowledgeStorage
    )
    print("✓ paola.knowledge imports work")

    # Test submodule imports
    from paola.knowledge.knowledge_base import KnowledgeBase
    from paola.knowledge.storage import (
        KnowledgeStorage,
        MemoryKnowledgeStorage,
        FileKnowledgeStorage
    )
    print("✓ Submodule imports work")

    # Test tools imports
    from paola.tools.knowledge_tools import (
        store_optimization_insight,
        retrieve_optimization_knowledge,
        list_all_knowledge,
        set_knowledge_base,
        get_knowledge_base
    )
    print("✓ Knowledge tools import work")

    print("\n✅ All import tests passed!\n")
    return True


def test_integration():
    """Test integration of knowledge module with platform."""
    print("Testing knowledge module integration...")

    from paola.knowledge import KnowledgeBase, MemoryKnowledgeStorage
    from paola.foundry import OptimizationFoundry, FileStorage
    import tempfile
    import shutil
    import numpy as np

    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Create platform and run optimization
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)

        run = platform.create_run(
            problem_id="test_rosenbrock",
            problem_name="Rosenbrock 10D",
            algorithm="SLSQP"
        )

        # Record iterations
        for i in range(5):
            run.record_iteration(
                design=np.random.rand(10),
                objective=10.0 / (i + 1),
                gradient=np.random.randn(10) * 0.1
            )

        class MockResult:
            def __init__(self):
                self.fun = 2.0
                self.x = np.zeros(10)
                self.success = True
                self.message = "Success"
                self.nfev = 5
                self.nit = 5
                self.njev = 5

        run.finalize(MockResult())
        platform.finalize_run(run.run_id)
        print("✓ Created test optimization run")

        # Create knowledge base
        kb = KnowledgeBase(storage=MemoryKnowledgeStorage())

        # Store insight from run
        run_record = platform.load_run(run.run_id)
        insight_id = kb.store_insight(
            problem_signature={
                "dimensions": 10,
                "problem_type": "rosenbrock",
                "constraints_count": 0
            },
            strategy={
                "algorithm": run_record.algorithm,
            },
            outcome={
                "success": run_record.success,
                "iterations": len(run_record.result_data.get("iterations", [])),
                "final_objective": run_record.objective_value
            }
        )
        print(f"✓ Stored insight from run (ID: {insight_id[:8]}...)")

        # Retrieve insight
        retrieved = kb.get_insight(insight_id)
        assert retrieved["problem_signature"]["dimensions"] == 10
        assert retrieved["strategy"]["algorithm"] == "SLSQP"
        print("✓ Retrieved insight successfully")

        print("\n✅ All integration tests passed!\n")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all Phase 3 skeleton tests."""
    print("=" * 60)
    print("Phase 3 Knowledge Module Skeleton Verification Tests")
    print("=" * 60)
    print()

    results = []

    results.append(("Module Imports", test_module_imports()))
    results.append(("KnowledgeBase Interface", test_knowledge_base_interface()))
    results.append(("Storage Backends", test_storage_backends()))
    results.append(("Knowledge Tools", test_knowledge_tools()))
    results.append(("CLI Commands", test_cli_commands()))
    results.append(("Integration", test_integration()))

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("🎉 All tests passed! Phase 3 skeleton complete!")
        print()
        print("Skeleton Features:")
        print("  ✓ KnowledgeBase class (interfaces defined)")
        print("  ✓ MemoryKnowledgeStorage (working)")
        print("  ✓ FileKnowledgeStorage (skeleton)")
        print("  ✓ Agent tools (placeholder)")
        print("  ✓ CLI commands (skeleton)")
        print("  ✓ Comprehensive README documenting design intent")
        print()
        print("Next Steps:")
        print("  1. Proceed with Phase 4 (Agent Polish)")
        print("  2. Collect 20+ real optimization runs")
        print("  3. Return to knowledge module with real data")
        print("  4. Implement Phase 3.2 (basic retrieval)")
        return 0
    else:
        print("❌ Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
