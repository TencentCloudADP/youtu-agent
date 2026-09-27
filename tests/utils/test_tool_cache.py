from types import SimpleNamespace

import pytest
from sqlmodel import Session, create_engine, select

from utu.db import ToolCacheModel
from utu.utils import SQLModelUtils, tool_cache


@pytest.fixture
def cache_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'tool-cache.db'}")
    ToolCacheModel.__table__.create(engine)
    monkeypatch.setattr(SQLModelUtils, "_engine", engine)
    monkeypatch.setattr(tool_cache, "USE_CACHE", True)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(tool_cache, "time", SimpleNamespace(time=lambda: clock.now))
    yield engine, clock
    engine.dispose()


async def test_db_cache_reuses_refreshed_result_after_expiration(cache_db):
    engine, clock = cache_db
    calls = 0

    @tool_cache.create_cached_db(expire_time=10)
    async def search(query):
        nonlocal calls
        calls += 1
        return {"query": query, "call": calls}

    assert await search("first") == {"query": "first", "call": 1}
    clock.now = 1009
    assert await search("first") == {"query": "first", "call": 1}
    clock.now = 1010
    assert await search("first") == {"query": "first", "call": 2}
    clock.now = 1011
    assert await search("first") == {"query": "first", "call": 2}
    assert await search("second") == {"query": "second", "call": 3}
    assert await search("first") == {"query": "first", "call": 2}

    with Session(engine) as session:
        assert len(session.exec(select(ToolCacheModel)).all()) == 2


async def test_db_cache_uses_newest_existing_duplicate(cache_db):
    engine, clock = cache_db
    calls = 0

    @tool_cache.create_cached_db(expire_time=10)
    async def search():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await search() == {"call": 1}
    with Session(engine) as session:
        original = session.exec(select(ToolCacheModel)).one()
        refreshed = original.model_dump(exclude={"id", "timestamp"})
        refreshed.update(timestamp=1010, result={"call": 2})
        session.add(ToolCacheModel(**refreshed))
        session.commit()

    clock.now = 1011
    assert await search() == {"call": 2}
    assert calls == 1


async def test_disabled_db_cache_refreshes_existing_result(cache_db, monkeypatch):
    engine, _ = cache_db
    calls = 0

    @tool_cache.create_cached_db()
    async def search():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await search() == {"call": 1}
    monkeypatch.setattr(tool_cache, "USE_CACHE", False)
    assert await search() == {"call": 2}
    assert await search() == {"call": 3}
    monkeypatch.setattr(tool_cache, "USE_CACHE", True)
    assert await search() == {"call": 3}

    with Session(engine) as session:
        assert len(session.exec(select(ToolCacheModel)).all()) == 1
