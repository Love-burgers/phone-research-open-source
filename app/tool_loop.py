"""Shared bounded LangGraph tool loop for chat and research adapters."""

from copy import deepcopy
import json
import logging
from time import monotonic

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

class AgentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_tool_calls: int = Field(default=8, ge=1, le=18)
    max_model_calls: int = Field(default=6, ge=2, le=12)
    max_total_tokens: int = Field(default=24000, ge=1000, le=128000)
    max_output_tokens: int = Field(default=1200, ge=128, le=2400)
    max_context_characters: int = Field(default=24000, ge=1000, le=128000)
    max_no_progress: int = Field(default=3, ge=1, le=6)
    deadline_seconds: int = Field(default=120, ge=5, le=300)
    max_search_calls: int = Field(default=3, ge=0, le=10)
    max_fetch_calls: int = Field(default=4, ge=0, le=12)


class ExecutionLimit(ValueError):
    """A configured execution boundary, distinct from a provider outage."""


class BudgetedJsonClient:
    """Share the final answer/report call's budget and observable failure handling."""
    available = True

    def __init__(self, client, limits, state, events, budget, charge, clock=monotonic):
        self.client, self.limits, self.state, self.events = client, limits, state, events
        self.budget, self.charge, self.clock = budget, charge, clock

    def complete_json(self, messages):
        reason, reservation = self.budget(self.state, messages)
        if reason or self.state["model_calls"] >= self.limits.max_model_calls:
            self.events.append({"action": "report", "status": "skipped", "reason": reason or "model_call_limit"})
            raise ExecutionLimit(reason or "model_call_limit")
        event, started = {"action": "report", "status": "ok"}, self.clock()
        attempted = True
        try:
            return self.client.complete_json(messages, max_output_tokens=self.limits.max_output_tokens)
        except ExecutionLimit as error:
            attempted = False
            event.update(status='skipped',reason=str(error))
            raise
        except Exception as error:
            event.update(status="error", error_type=type(error).__name__)
            logger.warning("model_request_failed stage=report error_type=%s", type(error).__name__)
            raise
        finally:
            if attempted:
                self.state.update(self.charge(self.state, reservation))
            event.update(model_usage=getattr(self.client, "last_usage", None),
                         elapsed_seconds=round(self.clock() - started, 4))
            self.events.append(event)


def build_tool_loop(*, client, limits, schemas, initialize, execute, finish, clock=monotonic, validate_finish=None, prepare_messages=None):
    def budget(state, messages, tools=(), *, reserve_final=False):
        if clock() - state["started_at"] >= limits.deadline_seconds:
            return "deadline", 0
        serialized = json.dumps([m.model_dump() if hasattr(m, "model_dump") else m for m in messages],
                                ensure_ascii=False) + json.dumps(tools, ensure_ascii=False)
        if len(serialized) > limits.max_context_characters:
            return "context_limit", 0
        # Byte length is a conservative request-size guard, not a tokenizer or price quote.
        reservation = len(serialized.encode("utf-8")) + limits.max_output_tokens
        cushion = min(limits.max_total_tokens//4,limits.max_output_tokens+4000) if reserve_final and state.get('sources') else 0
        if state["charged_tokens"] + reservation + cushion > limits.max_total_tokens:
            return "token_budget", 0
        return "", reservation

    def charge(state, reservation):
        usage = getattr(client, "last_usage", None)
        used = usage.get("total_tokens") if usage else None
        return {"model_calls": state["model_calls"] + 1,
                "charged_tokens": state["charged_tokens"] + (used if isinstance(used, int) and used >= 0 else reservation),
                "usage": [*state["usage"], usage]}

    def agent(state):
        if state["model_calls"] >= limits.max_model_calls - 1:
            return {"stop_reason": "model_call_limit"}
        active_schemas = schemas(state) if callable(schemas) else schemas
        messages = prepare_messages(state) if prepare_messages else state["messages"]
        reason, reservation = budget(state, messages, active_schemas, reserve_final=True)
        if reason:
            return {"stop_reason": reason}
        started = clock()
        try:
            response = client.invoke_tools(messages, active_schemas, max_output_tokens=limits.max_output_tokens)
            if response.invalid_tool_calls:
                raise ValueError("malformed_tool_calls")
            calls = response.tool_calls
            ids = [call.get("id") for call in calls]
            if any(not item for item in ids) or len(set(ids)) != len(ids):
                raise ValueError("invalid_tool_ids")
            event = {"action": "decision", "content": response.content, "tool_calls": calls,
                     "elapsed_seconds": round(clock() - started, 4)}
            feedback = validate_finish(state, response) if not calls and validate_finish else None
            if feedback:
                event["validation_feedback"] = feedback
            return {**charge(state, reservation), "messages": [*messages, response],
                    "actions": [*state["actions"], event], "nodes": [*state["nodes"], "agent"],
                    "stop_reason": "" if calls else "model_finished", **({
                        "messages": [*messages, response, HumanMessage(content=feedback)],
                        "stop_reason": "", "finish_retries": state.get("finish_retries", 0) + 1
                    } if feedback else {})}
        except ExecutionLimit as error:
            return {'stop_reason':str(error),'actions':[*state['actions'],
                {'action':'decision','status':'skipped','reason':str(error)}], 'nodes':[*state['nodes'],'agent']}
        except Exception as error:
            logger.warning("model_request_failed stage=decision error_type=%s", type(error).__name__)
            return {**charge(state, reservation), "stop_reason": "model_error",
                    "actions": [*state["actions"], {"action": "decision", "status": "error",
                        "error_type": type(error).__name__}], "nodes": [*state["nodes"], "agent"]}

    def guarded_tools(state):
        updated = deepcopy(state)
        updated["nodes"] = [*state["nodes"], "tools"]
        batch_progress = False
        for call in state["messages"][-1].tool_calls:
            started = clock()
            if updated["tool_calls_used"] >= limits.max_tool_calls:
                result = {"status": "tool_call_limit"}
            elif clock() - state["started_at"] >= limits.deadline_seconds:
                result = {"status": "deadline"}
            else:
                updated["tool_calls_used"] += 1
                try:
                    result = execute(updated, call["name"], call["args"])
                except Exception as error:
                    result = {"status": "tool_error", "error_type": type(error).__name__}
                    logger.warning("tool_failed tool=%s error_type=%s", call["name"], type(error).__name__)
            updated["messages"].append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                                   tool_call_id=call["id"], name=call["name"]))
            updated["actions"].append({"action": "tool", "tool_call_id": call["id"], "name": call["name"],
                                       "arguments": call["args"], "result": result,
                                       "elapsed_seconds": round(clock() - started, 4)})
            batch_progress = batch_progress or bool(result.get("added_evidence", 0) or result.get("added_candidates", 0))
        # A model may query three candidates at once. Let it observe that batch
        # and choose recovery; three empty siblings are one failed decision.
        updated["no_progress"] = 0 if batch_progress else state["no_progress"] + 1
        if updated["tool_calls_used"] >= limits.max_tool_calls:
            updated["stop_reason"] = "tool_call_limit"
        elif updated["no_progress"] >= limits.max_no_progress:
            updated["stop_reason"] = "no_progress"
        return updated


    def wrapped(node):
        def run(state):
            return {**state, **node(state)}
        return run

    def finalize(state):
        return finish(state, budget, charge)

    graph = StateGraph(dict)
    for name, node in (("initialize", initialize), ("agent", agent), ("tools", guarded_tools), ("report", finalize)):
        graph.add_node(name, wrapped(node))
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "agent")
    graph.add_conditional_edges("agent", lambda state: "report" if state["stop_reason"] else
                               "tools" if state["messages"][-1].type == "ai" else "agent")
    graph.add_conditional_edges("tools", lambda state: "report" if state["stop_reason"] else "agent")
    graph.add_edge("report", END)
    return graph.compile()
