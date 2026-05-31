import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.models.generation import JobStatus, StageStatus
from app.main import app
from app.services.generation.agentic.types import AssistantMessage
from app.services.planning import CreateAgentService, PlanningTurnOutcome
from tests.conftest import FakeModel


def _install_temp_session_store(monkeypatch, tmp_path):
    import app.services.sessions as sessions_pkg
    from app.api.v1 import chat as chat_api
    from app.api.v1 import public_shares as public_shares_api
    from app.api.v1 import sessions as sessions_api
    from app.api.v1 import workspaces as workspaces_api
    from app.api.v1 import workspace_sources as workspace_sources_api
    from app.services.sessions.store import SessionStore

    store = SessionStore(tmp_path / "zhiyan-test.db", tmp_path / "uploads")
    asyncio.run(store.init())

    monkeypatch.setattr(sessions_pkg, "session_store", store)
    monkeypatch.setattr(sessions_api, "session_store", store)
    monkeypatch.setattr(chat_api, "session_store", store)
    monkeypatch.setattr(public_shares_api, "session_store", store)
    monkeypatch.setattr(workspace_sources_api, "session_store", store)
    monkeypatch.setattr(workspaces_api, "session_store", store)
    return store


def _parse_sse_payloads(raw_text: str) -> list[dict]:
    events: list[dict] = []
    for line in raw_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_sessions_workspace_isolation_and_chat_persistence(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    client = TestClient(app)
    h1 = {"X-Workspace-Id": "ws-a"}
    h2 = {"X-Workspace-Id": "ws-b"}

    created = client.post("/api/v1/sessions", headers=h1, json={"title": "会话A"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    list_a = client.get("/api/v1/sessions", headers=h1)
    list_b = client.get("/api/v1/sessions", headers=h2)
    assert list_a.status_code == 200
    assert list_b.status_code == 200
    assert len(list_a.json()) == 1
    assert list_b.json() == []

    denied = client.get(f"/api/v1/sessions/{session_id}", headers=h2)
    assert denied.status_code == 404

    write_user = client.post(
        f"/api/v1/sessions/{session_id}/chat",
        headers=h1,
        json={"role": "user", "content": "你好", "model_meta": {}},
    )
    write_assistant = client.post(
        f"/api/v1/sessions/{session_id}/chat",
        headers=h1,
        json={"role": "assistant", "content": "您好", "model_meta": {}},
    )
    assert write_user.status_code == 200
    assert write_assistant.status_code == 200

    chat_list = client.get(f"/api/v1/sessions/{session_id}/chat", headers=h1)
    assert chat_list.status_code == 200
    records = chat_list.json()
    assert [r["role"] for r in records] == ["user", "assistant"]

    delete_resp = client.delete(f"/api/v1/sessions/{session_id}", headers=h1)
    assert delete_resp.status_code == 200

    after_delete = client.get(f"/api/v1/sessions/{session_id}", headers=h1)
    assert after_delete.status_code == 404


def test_generation_job_session_binding(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api
    from app.services.generation.job_store import GenerationJobStore

    class _NoopRunner:
        async def start_job(self, job_id: str, from_stage=None):  # noqa: ARG002
            return True

    monkeypatch.setattr(sessions_api, "job_store", GenerationJobStore(tmp_path / "jobs"))
    monkeypatch.setattr(sessions_api, "generation_runner", _NoopRunner())

    client = TestClient(app)
    h1 = {"X-Workspace-Id": "ws-a"}
    h2 = {"X-Workspace-Id": "ws-b"}

    created = client.post("/api/v1/sessions", headers=h1, json={"title": "生成会话"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    source_resp = client.post(
        "/api/v1/workspace/sources/text",
        headers=h1,
        json={"name": "素材", "content": "测试内容"},
    )
    assert source_resp.status_code == 200
    source_id = source_resp.json()["id"]
    link_resp = client.post(
        f"/api/v1/sessions/{session_id}/sources/link",
        headers=h1,
        json={"source_ids": [source_id]},
    )
    assert link_resp.status_code == 200

    create_ok = client.post(
        f"/api/v1/sessions/{session_id}/generation/jobs",
        headers=h1,
        json={
            "topic": "测试主题",
            "source_ids": [source_id],
            "num_pages": 3,
            "mode": "auto",
        },
    )
    assert create_ok.status_code == 200
    assert create_ok.json()["session_id"] == session_id
    created_job_id = create_ok.json()["job_id"]

    # Ensure source_hints is computed from source_ids and stored in the job request.
    job_detail = client.get(f"/api/v1/sessions/{session_id}/generation/jobs/{created_job_id}", headers=h1)
    assert job_detail.status_code == 200
    source_hints = job_detail.json()["request"]["source_hints"]
    assert source_hints["total_sources"] == 1
    assert source_hints["data"] == 1
    assert source_hints["images"] == 0

    detail = client.get(f"/api/v1/sessions/{session_id}", headers=h1)
    assert detail.status_code == 200
    latest_generation_job = detail.json().get("latest_generation_job")
    assert latest_generation_job is not None
    assert latest_generation_job["job_id"] == created_job_id
    assert latest_generation_job["status"] == "pending"

    create_denied = client.post(
        f"/api/v1/sessions/{session_id}/generation/jobs",
        headers=h2,
        json={
            "topic": "测试主题",
            "source_ids": [source_id],
            "num_pages": 3,
            "mode": "auto",
        },
    )
    assert create_denied.status_code == 404

    auto_session = client.post(
        "/api/v1/sessions",
        headers=h1,
        json={"title": "人工智能对未来工作影响"},
    )
    assert auto_session.status_code == 200
    auto_session_id = auto_session.json()["id"]

    create_auto = client.post(
        f"/api/v1/sessions/{auto_session_id}/generation/jobs",
        headers=h1,
        json={
            "topic": "请基于以下内容生成一个关于人工智能对未来工作影响的10页PPT，需要适合管理层汇报。",
            "num_pages": 3,
            "mode": "auto",
        },
    )
    assert create_auto.status_code == 200
    auto_session_detail = client.get(f"/api/v1/sessions/{auto_session_id}", headers=h1)
    assert auto_session_detail.status_code == 200
    assert auto_session_detail.json()["session"]["title"] == "人工智能对未来工作影响"

    auto_job_detail = client.get(
        f"/api/v1/sessions/{auto_session_id}/generation/jobs/{create_auto.json()['job_id']}",
        headers=h1,
    )
    assert auto_job_detail.status_code == 200
    assert auto_job_detail.json()["request"]["session_id"] == auto_session_id
    assert auto_job_detail.json()["request"]["topic"].startswith("请基于以下内容生成一个关于人工智能")

    existing_session = client.post("/api/v1/sessions", headers=h1, json={"title": "旧草稿"})
    assert existing_session.status_code == 200
    existing_session_id = existing_session.json()["id"]

    create_existing = client.post(
        f"/api/v1/sessions/{existing_session_id}/generation/jobs",
        headers=h1,
        json={
            "topic": "准备一个关于供应链优化的演示文稿，突出冷链、仓配协同和损耗控制。",
            "num_pages": 4,
            "mode": "auto",
        },
    )
    assert create_existing.status_code == 200
    existing_detail = client.get(f"/api/v1/sessions/{existing_session_id}", headers=h1)
    assert existing_detail.status_code == 200
    assert existing_detail.json()["session"]["title"] == "供应链优化"

    renamed = client.patch(
        f"/api/v1/sessions/{existing_session_id}",
        headers=h1,
        json={"title": "用户手动命名"},
    )
    assert renamed.status_code == 200

    create_preserved = client.post(
        f"/api/v1/sessions/{existing_session_id}/generation/jobs",
        headers=h1,
        json={
            "topic": "准备一个关于出海策略的演示文稿，强调渠道和品牌。",
            "num_pages": 4,
            "mode": "auto",
        },
    )
    assert create_preserved.status_code == 200
    preserved_detail = client.get(f"/api/v1/sessions/{existing_session_id}", headers=h1)
    assert preserved_detail.status_code == 200
    assert preserved_detail.json()["session"]["title"] == "用户手动命名"


def test_planning_turn_persists_outline_and_messages(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api

    async def _fake_handle_planning_turn(**kwargs):  # noqa: ARG001
        return PlanningTurnOutcome(
            assistant_message="我先起了一版 3 页大纲。",
            brief={
                "topic": "智能客服方案",
                "_meta": {"clarification_turns": 1, "user_turns": 1},
            },
            outline={
                "narrative_arc": "问题→方案→价值",
                "items": [
                    {
                        "slide_number": 1,
                        "title": "背景与目标",
                        "content_brief": "交代现状与目标",
                        "key_points": ["现状", "目标"],
                        "suggested_slide_role": "cover",
                    },
                    {
                        "slide_number": 2,
                        "title": "方案设计",
                        "content_brief": "说明方案模块",
                        "key_points": ["架构", "流程"],
                        "suggested_slide_role": "narrative",
                    },
                    {
                        "slide_number": 3,
                        "title": "预期价值",
                        "content_brief": "总结收益",
                        "key_points": ["效率", "满意度"],
                        "suggested_slide_role": "closing",
                    },
                ],
            },
            outline_version_increment=1,
            status="outline_ready",
            events=[
                {"type": "brief_updated", "brief": {"topic": "智能客服方案"}},
                {"type": "outline_drafted", "outline": {"items": [{"title": "背景与目标"}]}},
                {"type": "status_changed", "status": "outline_ready"},
            ],
        )

    monkeypatch.setattr(sessions_api, "handle_planning_turn", _fake_handle_planning_turn)

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-plan"}
    created = client.post("/api/v1/sessions", headers=headers, json={"title": "planning"})
    session_id = created.json()["id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/planning/turns",
        headers=headers,
        json={"message": "做一个面向客服团队的智能客服方案汇报"},
    )
    assert response.status_code == 200
    _ = response.text

    planning_detail = client.get(f"/api/v1/sessions/{session_id}/planning", headers=headers)
    assert planning_detail.status_code == 200
    planning_json = planning_detail.json()
    assert planning_json["planning_state"]["status"] == "outline_ready"
    assert planning_json["planning_state"]["outline"]["items"][1]["title"] == "方案设计"
    assert [item["role"] for item in planning_json["planning_messages"]] == [
        "assistant",
        "user",
        "assistant",
    ]

    session_detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert session_detail.status_code == 200
    assert session_detail.json()["planning_state"]["outline_version"] == 1

    outline_patch = client.patch(
        f"/api/v1/sessions/{session_id}/planning/outline",
        headers=headers,
        json={
            "outline": {
                "narrative_arc": "问题→方案→价值",
                "items": [
                    {
                        "slide_number": 1,
                        "title": "业务背景",
                        "content_brief": "重新命名后的首页",
                        "key_points": ["背景"],
                        "suggested_slide_role": "cover",
                    },
                    {
                        "slide_number": 2,
                        "title": "方案设计",
                        "content_brief": "说明方案模块",
                        "key_points": ["架构", "流程"],
                        "suggested_slide_role": "narrative",
                    },
                    {
                        "slide_number": 3,
                        "title": "预期价值",
                        "content_brief": "总结收益",
                        "key_points": ["效率", "满意度"],
                        "suggested_slide_role": "closing",
                    },
                ],
            }
        },
    )
    assert outline_patch.status_code == 200
    assert outline_patch.json()["outline_version"] == 2
    assert outline_patch.json()["outline"]["items"][0]["title"] == "业务背景"


def test_planning_confirm_starts_generation_from_approved_outline(monkeypatch, tmp_path):
    store = _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api
    real_create_generation_job_record = sessions_api.create_generation_job_record

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-plan-confirm"}
    created = client.post("/api/v1/sessions", headers=headers, json={"title": "planning"})
    session_id = created.json()["id"]

    source_resp = client.post(
        "/api/v1/workspace/sources/text",
        headers=headers,
        json={"name": "素材", "content": "智能客服建设方案"},
    )
    source_id = source_resp.json()["id"]
    link_resp = client.post(
        f"/api/v1/sessions/{session_id}/sources/link",
        headers=headers,
        json={"source_ids": [source_id]},
    )
    assert link_resp.status_code == 200

    asyncio.run(
        store.save_planning_state(
            workspace_id="ws-plan-confirm",
            session_id=session_id,
            status="outline_ready",
            brief={"topic": "智能客服方案", "_meta": {"clarification_turns": 1, "user_turns": 1}},
            outline={
                "narrative_arc": "问题→方案→价值",
                "items": [
                    {
                        "slide_number": 1,
                        "title": "业务背景",
                        "content_brief": "说明客服现状",
                        "key_points": ["背景"],
                        "suggested_slide_role": "cover",
                    },
                    {
                        "slide_number": 2,
                        "title": "方案设计",
                        "content_brief": "说明方案模块",
                        "key_points": ["架构", "流程"],
                        "suggested_slide_role": "narrative",
                    },
                    {
                        "slide_number": 3,
                        "title": "预期价值",
                        "content_brief": "总结收益",
                        "key_points": ["效率", "满意度"],
                        "suggested_slide_role": "closing",
                    },
                ],
            },
            outline_version=2,
            source_ids=[source_id],
            outline_stale=False,
            active_job_id=None,
        )
    )

    captured = {}

    async def _fake_create_generation_job_record(*, workspace_id, req, **kwargs):
        captured["workspace_id"] = workspace_id
        captured["req"] = req
        return (
            SimpleNamespace(
                job_id="job-approved-outline",
                status=JobStatus.PENDING,
                current_stage=None,
                request=SimpleNamespace(skill_id=req.skill_id),
                run_metadata=SimpleNamespace(run_id="run-approved-outline"),
                created_at="2026-03-01T00:00:00Z",
            ),
            session_id,
        )

    monkeypatch.setattr(
        sessions_api,
        "create_generation_job_record",
        _fake_create_generation_job_record,
    )

    confirm = client.post(
        f"/api/v1/sessions/{session_id}/planning/confirm",
        headers=headers,
        json={"output_mode": "html"},
    )
    assert confirm.status_code == 200
    payload = confirm.json()
    assert payload["job_id"] == "job-approved-outline"
    assert payload["status"] == "running"
    assert payload["current_stage"] == StageStatus.LAYOUT.value

    req = captured["req"]
    assert req.session_id == session_id
    assert req.topic == "智能客服方案"
    assert req.num_pages == 3
    assert req.output_mode.value == "html"
    assert req.skill_id is None
    assert req.approved_outline["items"][0]["title"] == "业务背景"

    detail = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert detail.status_code == 200
    planning_state = detail.json()["planning_state"]
    assert planning_state["status"] == "generating"
    assert planning_state["active_job_id"] == "job-approved-outline"

    monkeypatch.setattr(
        sessions_api,
        "create_generation_job_record",
        real_create_generation_job_record,
    )


def test_planning_turn_can_persist_output_mode_from_natural_language(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api

    async def _fake_handle_planning_turn(*, workspace_id, session_id, user_message):  # noqa: ANN001
        del workspace_id, session_id, user_message
        return PlanningTurnOutcome(
            assistant_message="这个场景更适合 HTML，我已经先按 HTML 路线记下来了。",
            brief={"topic": "产品发布会"},
            outline=None,
            status="collecting_requirements",
            output_mode="html",
            mode_selection_source="natural_language",
            events=[
                {
                    "type": "output_mode_selected",
                    "output_mode": "html",
                    "selection_source": "natural_language",
                    "reason": "用户明确要求 HTML",
                }
            ],
            assistant_status="ready",
        )

    monkeypatch.setattr(sessions_api, "handle_planning_turn", _fake_handle_planning_turn)

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-plan-mode"}
    created = client.post("/api/v1/sessions", headers=headers, json={"title": "planning-mode"})
    session_id = created.json()["id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/planning/turns",
        headers=headers,
        json={"message": "这个场景请用 HTML 做"},
    )
    assert response.status_code == 200

    planning_detail = client.get(f"/api/v1/sessions/{session_id}/planning", headers=headers)
    assert planning_detail.status_code == 200
    planning_state = planning_detail.json()["planning_state"]
    assert planning_state["output_mode"] == "html"
    assert planning_state["mode_selection_source"] == "natural_language"


def test_planning_turn_prompt_omits_topic_suggestion_details():
    service = CreateAgentService(session_store=None)

    prompt = service._build_turn_prompt(
        user_message="写一份 5 页演示文档",
        current_state={
            "output_mode": "slidev",
            "brief": {"topic": "Claude Code"},
            "outline": {},
            "topic_suggestions": [
                {"title": "现状问题与机会"},
                {"title": "方案设计与取舍"},
            ],
        },
        source_bundle={"sources": []},
        outline_stale=False,
    )

    assert "现状问题与机会" not in prompt
    assert "方案设计与取舍" not in prompt
    assert "pending_topic_suggestion_cards: true" in prompt


def test_planning_turn_model_input_does_not_include_topic_suggestion_text(monkeypatch, tmp_path):
    store = _install_temp_session_store(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "project_root", tmp_path)

    asyncio.run(store.ensure_workspace("ws-plan-input"))
    session = asyncio.run(store.create_session("ws-plan-input", "planning-input"))
    session_id = session["id"]
    asyncio.run(
        store.save_planning_state(
            workspace_id="ws-plan-input",
            session_id=session_id,
            status="collecting_requirements",
            brief={"topic": "Claude Code"},
            topic_suggestions=[
                {"title": "现状问题与机会", "prompt": "围绕现状问题与机会展开"},
                {"title": "方案设计与取舍", "prompt": "围绕方案设计与取舍展开"},
            ],
        )
    )

    fake_model = FakeModel(responses=[AssistantMessage(content="继续说说你最想强调的判断。")])
    monkeypatch.setattr(CreateAgentService, "_create_model_client", lambda self: fake_model)

    service = CreateAgentService(store)
    outcome = asyncio.run(
        service.handle_turn(
            workspace_id="ws-plan-input",
            session_id=session_id,
            user_message="写一份 5 页演示文档",
        )
    )

    assert outcome.assistant_message == "继续说说你最想强调的判断。"
    seen_contents = "\n".join(
        getattr(message, "content", "")
        for message in fake_model.seen_messages[0]
        if hasattr(message, "content")
    )
    assert "现状问题与机会" not in seen_contents
    assert "方案设计与取舍" not in seen_contents


def test_planning_turn_recovers_outline_from_markdown_table(monkeypatch, tmp_path):
    store = _install_temp_session_store(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "project_root", tmp_path)

    asyncio.run(store.ensure_workspace("ws-plan-recover"))
    session = asyncio.run(store.create_session("ws-plan-recover", "planning-recover"))
    session_id = session["id"]

    fake_model = FakeModel(
        responses=[
            AssistantMessage(
                content=(
                    "好，我先看完这份资料了。基于这次主题，我帮你规划 5 页结构如下：\n\n"
                    "| 页数 | 角色 | 标题 | 内容要点 |\n"
                    "| --- | --- | --- | --- |\n"
                    "| 1 | 封面 | Claude Code 提示词背后的三次转向 | 交代主题和核心判断 |\n"
                    "| 2 | 引入 | 一个你没写一个字的工作流 | 从截图提示走向 AI 自己跑通改代码 |\n"
                    "| 3 | 理论底座 | 说出来，就是做出来 | 用语言行为理论解释 prompt 为什么是行动 |\n"
                    "| 4 | 核心 | 三次哲学转向 | 从预测到理解任务再到模拟心智 |\n"
                    "| 5 | 延伸 | 隐性问题与 AI 的下一步 | 讨论隐式需求和心智闭环 |\n"
                )
            )
        ]
    )
    monkeypatch.setattr(CreateAgentService, "_create_model_client", lambda self: fake_model)

    service = CreateAgentService(store)
    outcome = asyncio.run(
        service.handle_turn(
            workspace_id="ws-plan-recover",
            session_id=session_id,
            user_message="就介绍 Claude Code prompt suggestions，写一个 5 页演示",
        )
    )

    assert outcome.status == "outline_ready"
    assert outcome.outline is not None
    assert len(outcome.outline["items"]) == 5
    assert outcome.outline["items"][0]["title"] == "Claude Code 提示词背后的三次转向"
    assert outcome.assistant_message == service._outline_ready_message()
    assert any(event.get("type") == "outline_updated" for event in outcome.events or [])

    planning_state = asyncio.run(store.get_planning_state("ws-plan-recover", session_id))
    assert planning_state is not None
    assert len(planning_state["outline"]["items"]) == 5

    planning_messages = asyncio.run(store.list_chat_messages("ws-plan-recover", session_id, limit=20))
    assistant_messages = [item["content"] for item in planning_messages if item["role"] == "assistant"]
    assert assistant_messages[-1] == service._outline_ready_message()
    assert "| 页数 |" not in assistant_messages[-1]


def test_workspace_source_link_acl_and_content_acl(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    client = TestClient(app)
    h1 = {"X-Workspace-Id": "ws-a"}
    h2 = {"X-Workspace-Id": "ws-b"}

    sess_a = client.post("/api/v1/sessions", headers=h1, json={"title": "A"}).json()["id"]
    sess_b = client.post("/api/v1/sessions", headers=h2, json={"title": "B"}).json()["id"]

    source_resp = client.post(
        "/api/v1/workspace/sources/text",
        headers=h1,
        json={"name": "跨空间素材", "content": "only ws-a can use"},
    )
    assert source_resp.status_code == 200
    source_id = source_resp.json()["id"]

    denied_link = client.post(
        f"/api/v1/sessions/{sess_b}/sources/link",
        headers=h2,
        json={"source_ids": [source_id]},
    )
    assert denied_link.status_code == 409

    fake_link = client.post(
        f"/api/v1/sessions/{sess_b}/sources/link",
        headers=h2,
        json={"source_ids": ["src-does-not-exist"]},
    )
    assert fake_link.status_code == 404

    link_ok = client.post(
        f"/api/v1/sessions/{sess_a}/sources/link",
        headers=h1,
        json={"source_ids": [source_id]},
    )
    assert link_ok.status_code == 200

    content_ok = client.get(
        f"/api/v1/sessions/{sess_a}/sources/{source_id}/content",
        headers=h1,
    )
    assert content_ok.status_code == 200
    assert "only ws-a can use" in content_ok.json()["content"]

    content_denied = client.get(
        f"/api/v1/sessions/{sess_b}/sources/{source_id}/content",
        headers=h2,
    )
    assert content_denied.status_code == 404


def test_unlink_source_is_idempotent(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-a"}

    session_resp = client.post("/api/v1/sessions", headers=headers, json={"title": "S1"})
    assert session_resp.status_code == 200
    session_id = session_resp.json()["id"]

    source_resp = client.post(
        "/api/v1/workspace/sources/text",
        headers=headers,
        json={"name": "待取消素材", "content": "unlink me"},
    )
    assert source_resp.status_code == 200
    source_id = source_resp.json()["id"]

    linked = client.post(
        f"/api/v1/sessions/{session_id}/sources/link",
        headers=headers,
        json={"source_ids": [source_id]},
    )
    assert linked.status_code == 200

    first_unlink = client.delete(
        f"/api/v1/sessions/{session_id}/sources/{source_id}/link",
        headers=headers,
    )
    assert first_unlink.status_code == 200
    assert first_unlink.json()["ok"] is True

    second_unlink = client.delete(
        f"/api/v1/sessions/{session_id}/sources/{source_id}/link",
        headers=headers,
    )
    assert second_unlink.status_code == 200
    assert second_unlink.json()["ok"] is True

    sources = client.get(f"/api/v1/sessions/{session_id}/sources", headers=headers)
    assert sources.status_code == 200
    assert sources.json() == []


def test_put_latest_presentation_workspace_isolation(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    client = TestClient(app)
    h1 = {"X-Workspace-Id": "ws-a"}
    h2 = {"X-Workspace-Id": "ws-b"}

    created = client.post("/api/v1/sessions", headers=h1, json={"title": "latest"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    presentation = {
        "presentationId": "pres-1",
        "title": "测试稿",
        "slides": [
            {
                "slideId": "slide-1",
                "layoutType": "bullet-with-icons",
                "layoutId": "bullet-with-icons",
                "contentData": {"title": "测试", "items": []},
                "components": [],
            }
        ],
    }

    ok = client.put(
        f"/api/v1/sessions/{session_id}/presentations/latest",
        headers=h1,
        json={"presentation": presentation, "source": "chat"},
    )
    assert ok.status_code == 200
    assert ok.json()["is_snapshot"] is False

    denied = client.put(
        f"/api/v1/sessions/{session_id}/presentations/latest",
        headers=h2,
        json={"presentation": presentation, "source": "chat"},
    )
    assert denied.status_code == 404




def test_workspace_source_dedup_and_cross_workspace_isolation(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)
    client = TestClient(app)
    h1 = {"X-Workspace-Id": "ws-a"}
    h2 = {"X-Workspace-Id": "ws-b"}

    first = client.post(
        "/api/v1/workspace/sources/text",
        headers=h1,
        json={"name": "文档A", "content": "same-content"},
    )
    assert first.status_code == 200
    first_payload = first.json()

    second = client.post(
        "/api/v1/workspace/sources/text",
        headers=h1,
        json={"name": "文档A-重复", "content": "same-content"},
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["id"] == first_payload["id"]
    assert second_payload["deduped"] is True

    cross = client.post(
        "/api/v1/workspace/sources/text",
        headers=h2,
        json={"name": "文档B", "content": "same-content"},
    )
    assert cross.status_code == 200
    cross_payload = cross.json()
    assert cross_payload["id"] != first_payload["id"]
    assert not cross_payload.get("deduped", False)


def test_workspace_text_source_name_falls_back_to_content(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)
    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-text-fallback"}

    response = client.post(
        "/api/v1/workspace/sources/text",
        headers=headers,
        json={"name": "   ", "content": "  第一行标题  \n第二行内容"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "第一行标题"


def test_workspace_sources_link_count_and_bulk_delete_cascade(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)
    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-a"}

    session_resp = client.post("/api/v1/sessions", headers=headers, json={"title": "S1"})
    assert session_resp.status_code == 200
    session_id = session_resp.json()["id"]

    source_resp = client.post(
        "/api/v1/workspace/sources/text",
        headers=headers,
        json={"name": "可删除素材", "content": "delete me"},
    )
    assert source_resp.status_code == 200
    source_id = source_resp.json()["id"]

    link_resp = client.post(
        f"/api/v1/sessions/{session_id}/sources/link",
        headers=headers,
        json={"source_ids": [source_id]},
    )
    assert link_resp.status_code == 200

    listed = client.get(
        "/api/v1/workspace/sources",
        headers=headers,
        params={"sort": "linked_desc"},
    )
    assert listed.status_code == 200
    rows = listed.json()
    row = next(item for item in rows if item["id"] == source_id)
    assert row["linked_session_count"] == 1

    deleted = client.post(
        "/api/v1/workspace/sources/bulk-delete",
        headers=headers,
        json={"source_ids": [source_id]},
    )
    assert deleted.status_code == 200
    assert source_id in deleted.json()["deleted_ids"]

    session_sources = client.get(f"/api/v1/sessions/{session_id}/sources", headers=headers)
    assert session_sources.status_code == 200
    assert session_sources.json() == []


def test_workspace_upload_rejects_dangerous_filename(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)
    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-upload-security"}

    traversal = client.post(
        "/api/v1/workspace/sources/upload",
        headers=headers,
        files={"file": ("../../evil.txt", b"evil", "text/plain")},
    )
    assert traversal.status_code == 400

    absolute = client.post(
        "/api/v1/workspace/sources/upload",
        headers=headers,
        files={"file": ("C:/Users/Public/evil.txt", b"evil", "text/plain")},
    )
    assert absolute.status_code == 400

    listed = client.get("/api/v1/workspace/sources", headers=headers)
    assert listed.status_code == 200
    assert listed.json() == []


def test_workspace_source_file_endpoint_serves_binary_asset(monkeypatch, tmp_path):
    store = _install_temp_session_store(monkeypatch, tmp_path)
    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-image-preview"}
    source_id = "src-image-preview"
    asyncio.run(store.ensure_workspace("ws-image-preview"))
    asset_dir = store.uploads_dir / source_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "preview.png"
    image_bytes = bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
        "0000000A49444154789C6360000002000154A24F5D0000000049454E44AE426082"
    )
    image_path.write_bytes(image_bytes)

    asyncio.run(
        store.create_workspace_source(
            workspace_id="ws-image-preview",
            source_type="file",
            name="preview.png",
            file_category="image",
            size=len(image_bytes),
            status="ready",
            content_hash="hash-image-preview",
            preview_snippet=None,
            storage_path=str(image_path.resolve()),
            parsed_content="",
            metadata={},
            source_id=source_id,
        )
    )

    response = client.get(
        f"/api/v1/workspace/sources/{source_id}/file",
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == image_bytes

    denied = client.get(
        f"/api/v1/workspace/sources/{source_id}/file",
        headers={"X-Workspace-Id": "ws-other"},
    )
    assert denied.status_code == 404


def test_workspaces_current_and_owner_unique_index(monkeypatch, tmp_path):
    store = _install_temp_session_store(monkeypatch, tmp_path)
    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-current"}

    current = client.get("/api/v1/workspaces/current", headers=headers)
    assert current.status_code == 200
    payload = current.json()
    assert payload["id"] == "ws-current"

    with sqlite3.connect(store._db_path) as conn:  # noqa: SLF001
        conn.execute(
            """
            UPDATE workspaces
            SET owner_type='user', owner_id='u-1'
            WHERE id=?
            """,
            ("ws-current",),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO workspaces(id, owner_type, owner_id, created_at, last_seen_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                ("ws-other", "user", "u-1", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )

def test_slidev_latest_presentation_endpoints(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api

    async def _fake_prepare_slidev_deck_artifact(**kwargs):  # noqa: ANN003
        return {
            "title": "Slidev 测试演示",
            "markdown": "---\ntitle: Slidev 测试演示\n---\n\n# 封面\n\n---\n\n# 收尾\n",
            "meta": {
                "title": "Slidev 测试演示",
                "slide_count": 2,
                "slides": [
                    {"index": 0, "slide_id": "slide-1", "title": "封面", "role": "cover", "layout": "cover"},
                    {"index": 1, "slide_id": "slide-2", "title": "收尾", "role": "closing", "layout": "center"},
                ],
                "selected_style_id": "tech-launch",
                "validation": {"ok": True, "issues": []},
                "review": {"issues": []},
            },
            "presentation": {
                "presentationId": "pres-slidev-test",
                "title": "Slidev 测试演示",
                "slides": [
                    {
                        "slideId": "slide-1",
                        "layoutType": "blank",
                        "layoutId": "blank",
                        "contentData": {"title": "封面"},
                        "components": [],
                    },
                    {
                        "slideId": "slide-2",
                        "layoutType": "blank",
                        "layoutId": "blank",
                        "contentData": {"title": "收尾"},
                        "components": [],
                    },
                ],
            },
            "selected_style_id": "tech-launch",
            "selected_style": {"name": "tech-launch", "theme": "seriph"},
            "selected_theme": {"theme": "seriph"},
        }

    async def _fake_build_slidev_spa(*, out_dir, **kwargs):  # noqa: ANN003
        build_out_dir = Path(out_dir)
        build_out_dir.mkdir(parents=True, exist_ok=True)
        (build_out_dir / "index.html").write_text("<html><body>slidev deck</body></html>", encoding="utf-8")
        assets_dir = build_out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "entry.js").write_text("console.log('slidev');", encoding="utf-8")

    monkeypatch.setattr(sessions_api, "prepare_slidev_deck_artifact", _fake_prepare_slidev_deck_artifact)
    monkeypatch.setattr(sessions_api, "build_slidev_spa", _fake_build_slidev_spa)

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-slidev"}
    created = client.post("/api/v1/sessions", headers=headers, json={"title": "slidev latest"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    presentation = {
        "presentationId": "pres-slidev-test",
        "title": "Slidev 测试演示",
        "slides": [
            {
                "slideId": "slide-1",
                "layoutType": "blank",
                "layoutId": "blank",
                "contentData": {"title": "封面"},
                "components": [],
            },
            {
                "slideId": "slide-2",
                "layoutType": "blank",
                "layoutId": "blank",
                "contentData": {"title": "收尾"},
                "components": [],
            },
        ],
    }

    saved = client.put(
        f"/api/v1/sessions/{session_id}/presentations/latest",
        headers=headers,
        json={
            "presentation": presentation,
            "source": "chat",
            "output_mode": "slidev",
            "slidev_deck": {
                "markdown": "# placeholder",
                "selected_style_id": "tech-launch",
                "meta": {"slides": [{"title": "封面"}, {"title": "收尾"}]},
            },
        },
    )
    assert saved.status_code == 200

    latest = client.get(f"/api/v1/sessions/{session_id}/presentations/latest", headers=headers)
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["output_mode"] == "slidev"
    assert latest_payload["artifacts"]["slidev_deck"]["selected_style_id"] == "tech-launch"
    assert latest_payload["artifacts"]["slidev_build"]["slide_count"] == 2

    slidev = client.get(f"/api/v1/sessions/{session_id}/presentations/latest/slidev", headers=headers)
    assert slidev.status_code == 200
    slidev_payload = slidev.json()
    assert slidev_payload["meta"]["slide_count"] == 2
    assert slidev_payload["meta"]["slides"][0]["title"] == "封面"
    assert slidev_payload["build_url"].endswith("/presentations/latest/slidev/build")

    markdown = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/markdown",
        headers=headers,
    )
    assert markdown.status_code == 200
    assert "# 封面" in markdown.text

    meta = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/meta",
        headers=headers,
    )
    assert meta.status_code == 200
    assert meta.json()["selected_style_id"] == "tech-launch"

    build_entry = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/build",
        headers=headers,
    )
    assert build_entry.status_code == 200
    assert "slidev deck" in build_entry.text

    build_entry_without_header = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/build",
    )
    assert build_entry_without_header.status_code == 200
    assert "slidev deck" in build_entry_without_header.text

    build_asset = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/build/assets/entry.js",
        headers=headers,
    )
    assert build_asset.status_code == 200
    assert "console.log('slidev')" in build_asset.text

    build_asset_without_header = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/build/assets/entry.js",
    )
    assert build_asset_without_header.status_code == 200
    assert "console.log('slidev')" in build_asset_without_header.text


def test_slidev_latest_presentation_survives_build_failure(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api

    async def _fake_prepare_slidev_deck_artifact(**kwargs):  # noqa: ANN003
        return {
            "title": "Slidev 构建失败保留 artifact",
            "markdown": "---\ntitle: Slidev 构建失败保留 artifact\n---\n\n# 封面\n",
            "meta": {
                "title": "Slidev 构建失败保留 artifact",
                "slide_count": 1,
                "slides": [
                    {"index": 0, "slide_id": "slide-1", "title": "封面", "role": "cover", "layout": "cover"},
                ],
                "selected_style_id": "tech-launch",
                "validation": {"ok": True, "issues": []},
                "review": {"issues": []},
            },
            "presentation": {
                "presentationId": "pres-slidev-test",
                "title": "Slidev 构建失败保留 artifact",
                "slides": [
                    {
                        "slideId": "slide-1",
                        "layoutType": "blank",
                        "layoutId": "blank",
                        "contentData": {"title": "封面"},
                        "components": [],
                    }
                ],
            },
            "selected_style_id": "tech-launch",
            "selected_style": {"name": "tech-launch", "theme": "seriph"},
            "selected_theme": {"theme": "seriph"},
        }

    async def _fake_build_slidev_spa(**kwargs):  # noqa: ANN003
        raise RuntimeError("Slidev build failed: broken css")

    monkeypatch.setattr(sessions_api, "prepare_slidev_deck_artifact", _fake_prepare_slidev_deck_artifact)
    monkeypatch.setattr(sessions_api, "build_slidev_spa", _fake_build_slidev_spa)

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-slidev-fail"}
    created = client.post("/api/v1/sessions", headers=headers, json={"title": "slidev fail"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    presentation = {
        "presentationId": "pres-slidev-test",
        "title": "Slidev 构建失败保留 artifact",
        "slides": [
            {
                "slideId": "slide-1",
                "layoutType": "blank",
                "layoutId": "blank",
                "contentData": {"title": "封面"},
                "components": [],
            }
        ],
    }

    saved = client.put(
        f"/api/v1/sessions/{session_id}/presentations/latest",
        headers=headers,
        json={
            "presentation": presentation,
            "source": "chat",
            "output_mode": "slidev",
            "slidev_deck": {
                "markdown": "# placeholder",
                "selected_style_id": "tech-launch",
                "meta": {"slides": [{"title": "封面"}]},
            },
        },
    )
    assert saved.status_code == 200

    latest = client.get(f"/api/v1/sessions/{session_id}/presentations/latest", headers=headers)
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["artifact_status"] == "ready"
    assert latest_payload["render_status"] == "failed"
    assert latest_payload["render_available"] is False
    assert "slidev_build" not in latest_payload["artifacts"]

    slidev = client.get(f"/api/v1/sessions/{session_id}/presentations/latest/slidev", headers=headers)
    assert slidev.status_code == 200
    slidev_payload = slidev.json()
    assert slidev_payload["build_url"] is None
    assert slidev_payload["render_status"] == "failed"
    assert "broken css" in slidev_payload["render_error"]

    build_entry = client.get(
        f"/api/v1/sessions/{session_id}/presentations/latest/slidev/build",
        headers=headers,
    )
    assert build_entry.status_code == 404


def test_session_share_link_rejects_slidev(monkeypatch, tmp_path):
    _install_temp_session_store(monkeypatch, tmp_path)

    from app.api.v1 import sessions as sessions_api

    async def _fake_prepare_slidev_deck_artifact(**kwargs):  # noqa: ANN003
        return {
            "title": "Slidev 分享限制",
            "markdown": "---\ntitle: Slidev 分享限制\n---\n\n# 封面\n",
            "meta": {
                "title": "Slidev 分享限制",
                "slide_count": 1,
                "slides": [
                    {"index": 0, "slide_id": "slide-1", "title": "封面", "role": "cover", "layout": "cover"},
                ],
                "selected_style_id": "tech-launch",
                "validation": {"ok": True, "issues": []},
                "review": {"issues": []},
            },
            "presentation": {
                "presentationId": "pres-slidev-share",
                "title": "Slidev 分享限制",
                "slides": [
                    {
                        "slideId": "slide-1",
                        "layoutType": "blank",
                        "layoutId": "blank",
                        "contentData": {"title": "封面"},
                        "components": [],
                    }
                ],
            },
            "selected_style_id": "tech-launch",
            "selected_style": {"name": "tech-launch", "theme": "seriph"},
            "selected_theme": {"theme": "seriph"},
        }

    async def _fake_build_slidev_spa(*, out_dir, **kwargs):  # noqa: ANN003
        build_out_dir = Path(out_dir)
        build_out_dir.mkdir(parents=True, exist_ok=True)
        (build_out_dir / "index.html").write_text("<html><body>slidev deck</body></html>", encoding="utf-8")

    monkeypatch.setattr(sessions_api, "prepare_slidev_deck_artifact", _fake_prepare_slidev_deck_artifact)
    monkeypatch.setattr(sessions_api, "build_slidev_spa", _fake_build_slidev_spa)

    client = TestClient(app)
    headers = {"X-Workspace-Id": "ws-slidev-share"}
    created = client.post("/api/v1/sessions", headers=headers, json={"title": "Slidev 分享"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    presentation = {
        "presentationId": "pres-slidev-share",
        "title": "Slidev 分享限制",
        "slides": [
            {
                "slideId": "slide-1",
                "layoutType": "blank",
                "layoutId": "blank",
                "contentData": {"title": "封面"},
                "components": [],
            }
        ],
    }

    saved = client.put(
        f"/api/v1/sessions/{session_id}/presentations/latest",
        headers=headers,
        json={
            "presentation": presentation,
            "source": "chat",
            "output_mode": "slidev",
            "slidev_deck": {
                "markdown": "# placeholder",
                "selected_style_id": "tech-launch",
                "meta": {"slides": [{"title": "封面"}]},
            },
        },
    )
    assert saved.status_code == 200

    share = client.post(f"/api/v1/sessions/{session_id}/share-link", headers=headers)
    assert share.status_code == 422
    assert "Slidev" in share.json()["detail"]
